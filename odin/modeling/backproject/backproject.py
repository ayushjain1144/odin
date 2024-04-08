import torch
import numpy as np
from torch_scatter import scatter_mean
from torch.nn import functional as F
import libs.pointops2.functions.pointops as pointops
from scipy.spatial.transform import Rotation as R

import math


import ipdb
st = ipdb.set_trace


def get_rotation_transform():
    # sample random rotation between 0 and pi/24
    angle = np.random.uniform(-np.pi / 24, np.pi / 24)
    r_x = R.from_euler('x', angle, degrees=False)

    angle = np.random.uniform(-np.pi / 24, np.pi / 24)
    r_y = R.from_euler('y', angle, degrees=False)

    angle = np.random.uniform(-np.pi, np.pi)
    r_z = R.from_euler('z', angle, degrees=False)

    mat = r_z.as_matrix() @ r_y.as_matrix() @ r_x.as_matrix()

    return mat

def rotation_augmentation_fast(pc, scannet_pc=None):
    mat = torch.from_numpy(get_rotation_transform()).to(pc)
    if scannet_pc is not None:
        center_point = scannet_pc.mean(dim=0)
        scannet_pc = (scannet_pc - center_point) @ mat.T + center_point
    else:
        center_point = pc.mean(dim=0)
    pc = (pc - center_point) @ mat.T + center_point
    return pc, scannet_pc

def scale_augmentations_fast(pc, scannet_pc=None):
    transform = torch.diag(torch.distributions.uniform.Uniform(0.9,1.1).sample([3,]).to(torch.float32))
    pc = pc @ transform
    if scannet_pc is not None:
        scannet_pc = scannet_pc @ transform
    return pc, scannet_pc


def augment_depth_numpy(xyz, scannet_pc=None, do_rot_scale=False):
    """
    xyz: B X V X H X W X 3
    """
    B, V, H, W, _ = xyz.shape
    assert B == 1

    # convert to a pointcloud
    xyz = xyz.reshape(B, V*H*W, 3)
    
    # mean center
    if scannet_pc is not None:
        mean = scannet_pc.mean(1, keepdims=True)
        scannet_pc -= mean
    else:
        mean = xyz.mean(1, keepdims=True)

    xyz -= mean
    
    # # add unform noise between xyz.min(1) and xyz.max(1)
    if scannet_pc is not None:
        noise = np.random.uniform(scannet_pc.min(1, keepdims=True),
            scannet_pc.max(1, keepdims=True)) / 2.0
        scannet_pc += noise
    else:
        noise = np.random.uniform(xyz.min(1, keepdims=True),
            xyz.max(1, keepdims=True)) / 2.0

    xyz += noise

    if scannet_pc is not None:
        scannet_pc = scannet_pc[0]

    xyz = xyz[0]

    xyz = torch.from_numpy(xyz)
    if scannet_pc is not None:
        scannet_pc = torch.from_numpy(scannet_pc)
        
    if do_rot_scale:
        xyz, scannet_pc = rotation_augmentation_fast(xyz, scannet_pc)
        xyz, scannet_pc = scale_augmentations_fast(xyz, scannet_pc)

    xyz = xyz.reshape(B, V, H, W, 3)
    return xyz, scannet_pc

  
def unproject(intrinsics, poses, depths):
    """
    Inputs:
        intrinsics: B X V X 3 X 3
        poses: B X V X 4 X 4 (torch.tensor)
        depths: B X V X H X W (torch.tensor)
    
    Outputs:
        world_coords: B X V X H X W X 3 (all valid 3D points)
        valid: B X V X H X W (bool to indicate valid points)
                can be used to index into RGB images
                to get N X 3 valid RGB values
    """
    B, V, H, W = depths.shape
    fx, fy, px, py = intrinsics[..., 0, 0][..., None], intrinsics[..., 1, 1][..., None], intrinsics[..., 0, 2][..., None], intrinsics[..., 1, 2][..., None]

    y = torch.arange(0, H).to(depths.device)
    x = torch.arange(0, W).to(depths.device)
    y, x = torch.meshgrid(y, x)

    x = x[None, None].repeat(B, V, 1, 1).flatten(2)
    y = y[None, None].repeat(B, V, 1, 1).flatten(2)
    z = depths.flatten(2)
    x = (x - px) * z / fx
    y = (y - py) * z / fy
    cam_coords = torch.stack([
        x, y, z, torch.ones_like(x)
    ], -1)

    world_coords = (poses @ cam_coords.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
    world_coords = world_coords[..., :3] / world_coords[..., 3][..., None]

    world_coords = world_coords.reshape(B, V, H, W, 3)

    return world_coords

def backproject_depth(depths, poses, intrinsics=None):
    B, V, H, W = depths.shape
    if intrinsics is None:
        print("CAUTION!!! USING DEFAULT INTRINSICS oF SCANNET!! MIGHT BE BAD!!")
        intrinsics = torch.from_numpy(
            get_scannet_intrinsic([H, W])[:3, :3]).reshape(1, 1, 3, 3).repeat(B, V, 1, 1).cuda().to(depths.dtype)
    xyz = unproject(intrinsics, poses, depths)
    return xyz


def interpolate_depth(xyz, multi_scale_features, method="nearest"):
    multi_scale_xyz = []
    B, V, H, W, _ = xyz.shape
    for feat in multi_scale_features:
        h, w = feat.shape[2:]
        xyz_ = torch.nn.functional.interpolate(
            xyz.reshape(B*V, H, W, 3).permute(0, 3, 1, 2), size=(h, w),
            mode=method).permute(0, 2, 3, 1).reshape(B, V, h, w, 3)
        multi_scale_xyz.append(xyz_.float())    
    return multi_scale_xyz


def make_xyz_for_rgb(h, w):
    """
    Input: h, w
    Output: xyz: h, w, 3
    """
    y = torch.arange(0, h).to(torch.float32)
    x = torch.arange(0, w).to(torch.float32)
    y, x = torch.meshgrid(y, x)
    xyz = torch.stack([
        x, y, torch.ones_like(x)
    ], -1)
    return xyz

def backprojector_dataloader(
    multi_scale_features, depths, poses,
    intrinsics=None, augment=False,
    method='nearest', scannet_pc=None, padding=None,
    do_rot_scale=False):
    """
    Inputs:
        multi_scale_features: list
            [B*V, 256, 15, 20], [B*V, 256, 30, 40], [B*V, 256, 60, 80]
        depths: tensor [B, 5, 480, 640]
        poses: tensor [B, 5, 4, 4]
        mask_features: [B, 5, 256, 120, 160]
        intrinsics: tensor [B, 5, 4, 4]

    Outputs:
        list: []
            B, V, H, W, 3
    """
    xyz = backproject_depth(
        depths[None], poses[None], intrinsics[None])
    
    if augment:
        new_xyz, scannet_pc = augment_depth_numpy(
            xyz.numpy(),
            scannet_pc[None].numpy() if scannet_pc is not None else None,
            do_rot_scale=do_rot_scale)

    else:
        new_xyz = xyz
    
    if padding is not None:
        new_xyz = F.pad(new_xyz.permute(0, 1, 4, 2, 3), (0, padding[1], 0, padding[0]), mode='constant', value=0).permute(0, 1, 3, 4, 2)

    multi_scale_xyz = interpolate_depth(
        new_xyz, multi_scale_features,
        method=method)

    multi_scale_xyz = [xyz.squeeze(0) for xyz in multi_scale_xyz]
    
    return multi_scale_xyz, scannet_pc, new_xyz


def multiscsale_voxelize(multi_scale_xyz, voxel_size):
    """
    Inputs: 
        multi_scale_xyz: list of tensors [B, V, H, W, 3]
        voxel_size: list of floats of len multi_scale_xyz

    Outputs:
        N=V*H*W
        multi_scale_unqiue_idx list of tensors [B, N]
        multi_scale_p2v: list of tensors [B, N]
        multi_scale_padding_mask: list of tensors [B, N]
    """
    multi_scale_p2v = []
    assert len(multi_scale_xyz) == len(voxel_size)
    for i, xyz in enumerate(multi_scale_xyz):
        B, V, H, W, _ = xyz.shape
        xyz = xyz.reshape(B, V*H*W, 3)
        point_to_voxel = voxelization(xyz, voxel_size[i])
        multi_scale_p2v.append(point_to_voxel)
    return multi_scale_p2v


def ravel_hash_vec(arr):
    """
    Ravel the coordinates after subtracting the min coordinates.
    """
    assert len(arr.shape) == 3
    arr -= arr.min(1, keepdims=True)[0].to(torch.long)
    arr_max = arr.max(1, keepdims=True)[0].to(torch.long) + 1

    keys = torch.zeros(arr.shape[0], arr.shape[1], dtype=torch.long).to(arr.device)

    # Fortran style indexing
    for j in range(arr.shape[2] - 1):
        keys += arr[..., j]
        keys *= arr_max[..., j + 1]
    keys += arr[..., -1]
    return keys

def voxelization(xyz, voxel_size):
    """
    Inputs:
        xyz: tensor [B, N, 3]
        voxel_size: float
    Outputs: 
        point_to_voxel_all: tensor [B, N], is the mapping from original point cloud to voxel
    """
    B, N, _ = xyz.shape
    xyz = xyz / voxel_size
    xyz = torch.round(xyz).long()
    xyz = xyz - xyz.min(1, keepdim=True)[0]

    keys = ravel_hash_vec(xyz)

    point_to_voxel = torch.stack(
        [torch.unique(keys[b], return_inverse=True)[1] for b in range(B)], 0)
    return point_to_voxel


def prepare_feats_for_pointops(xyz, shape, feats=None, voxelize=False, p2v=None):
    bs, v = shape

    if len(xyz.shape) == 5:
        b, v, h, w, _ = xyz.shape
        if feats is not None:
            feats = feats.reshape(bs, v, feats.shape[1], h, w).permute(0, 1, 3, 4, 2).flatten(1, 3) # B, VHW, F
        xyz = xyz.reshape(bs, v, h, w, 3).flatten(1, 3) # B, VHW, 3
        
    if voxelize:
        if feats is not None:
            feats = torch.cat(
                [scatter_mean(feats[b], p2v[b], dim=0) for b in range(len(feats))]) # bn, F
        xyz = torch.cat(
            [scatter_mean(xyz[b], p2v[b], dim=0) for b in range(len(xyz))])
        batch_offset = ((p2v).max(1)[0] + 1).cumsum(0).to(torch.int32)
    else:
        # queryandgroup expects N, F and N, 3 with additional batch offset
        xyz = xyz.flatten(0, 1).contiguous()
        if feats is not None:
            feats = feats.flatten(0, 1).contiguous()
        batch_offset = (torch.arange(bs, dtype=torch.int32, device=xyz.device) + 1) * v * h * w

    return feats, xyz, batch_offset
    

def voxel_map_to_source(voxel_map, poin2voxel):
    """
    Input:
        voxel_map (B, N1, C)
        point2voxel (B, N)
    Output:
        src_new (B, N, C)
    """
    bs, n, c = voxel_map.shape
    src_new = torch.stack([voxel_map[i, poin2voxel[i]] for i in range(bs)])
    return src_new


def interpolate_feats_3d(
    source_feats, source_xyz,
    source_p2v, target_xyz,
    target_p2v, shape, num_neighbors=3, voxelize=False, 
    return_voxelized=False
    ):
    """
    Inputs:
        source_feats: tensor [B*V, C, H1, W1] or B, N, C
        source_xyz: tensor [B, V, H1, W1, 3] or B, N, 3
        source_p2v: tensor [B, N1]
        target_xyz: tensor [B, V, H2, W2, 3] or B, N2, 3
        target_p2v: tensor [B, N2]
    Outputs:
        target_feats: tensor [BV, C, H2, W2] or B, C, N2
    """
    source_feats_pops, source_xyz_pops, source_batch_offset = prepare_feats_for_pointops(
        xyz=source_xyz, shape=shape, feats=source_feats,
        voxelize=voxelize, p2v=source_p2v
    )
    _, target_xyz_pops, target_batch_offset = prepare_feats_for_pointops(
        xyz=target_xyz, shape=shape, feats=None,
        voxelize=voxelize, p2v=target_p2v)
    target_feats = pointops.interpolation(
        source_xyz_pops, target_xyz_pops, source_feats_pops,
        source_batch_offset, target_batch_offset, k=num_neighbors).to(source_feats) # bn, C

    # undo voxelization
    if voxelize and not return_voxelized:
        out_new = []
        idx = 0
        for i, b in enumerate(target_batch_offset):
            out_new.append(target_feats[idx:b][target_p2v[i]])
            idx = b
        del target_feats
        output = torch.stack(out_new, 0)
        if len(target_xyz.shape) == 5:
            bs, v, h, w, _ = target_xyz.shape
            output = output.reshape(bs, v, h, w, -1).flatten(0, 1).permute(0, 3, 1, 2) # BV, C, H, W
        else:
            output = output.permute(0, 2, 1)

    else:
        # just batch it
        if len(target_batch_offset) != 1:
            max_batch_size = (target_batch_offset[1:] - target_batch_offset[:-1]).max()
            max_batch_size = max(max_batch_size, target_batch_offset[0])
        else:    
            max_batch_size = target_batch_offset[0]
        output = torch.zeros(
            len(target_batch_offset), max_batch_size, target_feats.shape[1], device=target_feats.device)
        idx = 0
        for i, b in enumerate(target_batch_offset):
            output[i, :b - idx] = target_feats[idx:b]
            idx = b
        output = output.permute(0, 2, 1)
        if not return_voxelized and len(source_feats.shape) == 4:
            output = output.reshape(
                output.shape[0], output.shape[1], target_xyz.shape[2], target_xyz.shape[3])
       
    return output


def get_scannet_intrinsic(image_size, intrinsic_file=None):
    scannet_intrinsic = np.array([[577.871,   0.       , 319.5, 0.],
                                  [  0.       , 577.871, 239.5, 0.],
                                  [  0.       ,   0.       ,   1., 0. ],
                                  [0., 0., 0., 1.0]
                                ])
    scannet_intrinsic[0] /= 480 / image_size[0]
    scannet_intrinsic[1] /= 640 / image_size[1]
    return scannet_intrinsic

def get_ai2thor_intrinsics(image_size, intrinsic_file=None):
    fov = 90
    hfov = float(fov) * np.pi / 180.
    H, W = image_size[:2]
    if H != W:
        assert False, "Ai2thor only supports square images"
    intrinsics = np.array([
            [(W/2.)*1 / np.tan(hfov / 2.), 0., 0., 0.],
            [0., (H/2.)*1 / np.tan(hfov / 2.), 0., 0.],
            [0., 0.,  1, 0],
            [0., 0., 0, 1]])
    intrinsics[0,2] = W/2.
    intrinsics[1,2] = H/2.
    return intrinsics



def get_s3dis_intrinsics(image_size, intrinsic_file=None):
    H, W = image_size[:2]
    
    intrinsics = np.loadtxt(intrinsic_file).reshape(3, 3).astype(np.float32)
    intrinsics_ = np.eye(4).astype(np.float32)
    intrinsics_[:3, :3] = intrinsics    
    intrinsics = intrinsics_
    
    intrinsics[0] /= 1080 / H
    intrinsics[1] /= 1080 / W
    
    return intrinsics


def get_matterport_intrinsics(image_size, intrinsic_file=None):
    H, W = image_size[:2]
    
    intrinsics = np.loadtxt(intrinsic_file).reshape(3, 3).astype(np.float32)
    intrinsics_ = np.eye(4).astype(np.float32)
    intrinsics_[:3, :3] = intrinsics    
    intrinsics = intrinsics_
    
    intrinsics[0] /= 512 / H
    intrinsics[1] /= 640 / W
    
    return intrinsics