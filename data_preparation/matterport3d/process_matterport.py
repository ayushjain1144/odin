import numpy as np
import os
import random
from PIL import Image
from natsort import natsorted
from glob import glob
import multiprocessing as mp

import torch

import ipdb
st = ipdb.set_trace

DATA_PATH = '/projects/katefgroup/language_grounding/SEMSEG_100k/matterport_2d/'
HOUSE_SEGMENTATIONS = "/projects/katefgroup/language_grounding/SEMSEG_100k/matterport"
PC_PROCESSED = "/projects/katefgroup/language_grounding/mask3d_processed/matterport"
FRAMES_PROCESSED = "/projects/katefgroup/language_grounding/SEMSEG_100k/matterport_frames"


if not os.path.exists(PC_PROCESSED):
    os.makedirs(PC_PROCESSED)
    

def load_matrix_from_txt(path, shape=(4, 4)):
    with open(path) as f:
        txt = f.readlines()
    txt = ''.join(txt).replace('\n', ' ')
    matrix = [float(v) for v in txt.split()]
    return np.array(matrix).reshape(shape)


def region_to_camera_mapping_fn(pano_file):
    pano_region = open(pano_file).readlines()
    region_to_camera_mapping = {}
    for line in pano_region:
        idx, camera, region, _ = line.split(' ')
        if region in region_to_camera_mapping:
            region_to_camera_mapping[region].append(camera)
        else:
            region_to_camera_mapping[region] = [camera]
            
    if '-1' in region_to_camera_mapping:
        del region_to_camera_mapping['-1']
        
    return region_to_camera_mapping
            
            
def process_pc(pc_path):
    pc, color, labels = torch.load(pc_path)
    color = (color + 1) * 127.5
    
    segments = torch.ones_like(labels)
    normals = torch.ones_like(pc)
    
    points = torch.cat(
        [pc, color, normals, segments, labels], dim=-1)
    
    return points.numpy()


def fix_depth(depth):
    depth = depth / 4.0
    return depth

def fix_intrinsics(intrinsics):
    intrinsics[..., 1, 2] = 512 - intrinsics[..., 1, 2]
    return intrinsics

def load_image(path):
    image = Image.open(path)
    return np.array(image)


def load_raw_data(poses_path, depths_path, rgbs_path, intrinsics_path, cam_uiud, output_scene_name):
    poses_path = [pose_path for pose_path in natsorted(glob(os.path.join(poses_path, "**"))) for cam in cam_uiud if cam in pose_path]
    depths_path = [fix_depth(load_image(depth_path)) for depth_path in natsorted(glob(os.path.join(depths_path, "**"))) for cam in cam_uiud if cam in depth_path]
    rgbs_path = [rgb_path for rgb_path in natsorted(glob(os.path.join(rgbs_path, "**"))) for cam in cam_uiud if cam in rgb_path]
    intrinsics = [fix_intrinsics(load_matrix_from_txt(intrinsic_path, shape=(3, 3))) for intrinsic_path in natsorted(glob(os.path.join(intrinsics_path, "**"))) for cam in cam_uiud if cam in intrinsic_path]
    
    OUTPUT_DIR = f"{FRAMES_PROCESSED}/{output_scene_name}"
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # color dir
    COLOR_DIR = f"{OUTPUT_DIR}/color"
    if not os.path.exists(COLOR_DIR):
        os.makedirs(COLOR_DIR)
        
    # depth dir
    DEPTH_DIR = f"{OUTPUT_DIR}/depth"
    if not os.path.exists(DEPTH_DIR):
        os.makedirs(DEPTH_DIR)
        
    # pose dir
    POSE_DIR = f"{OUTPUT_DIR}/pose"
    if not os.path.exists(POSE_DIR):
        os.makedirs(POSE_DIR)
    
    # intrinsic dir
    INTRINSIC_DIR = f"{OUTPUT_DIR}/intrinsic"
    if not os.path.exists(INTRINSIC_DIR):
        os.makedirs(INTRINSIC_DIR)
        
    for i, (pose, depth, rgb, intrinsic) in enumerate(zip(poses_path, depths_path, rgbs_path, intrinsics)):
        # save color by copying the file
        os.system(f"cp {rgb} {COLOR_DIR}/{i}.jpg")
        
        # save depth by saving the numpy array
        image = Image.fromarray(depth.astype(np.uint16))
        image.save(f"{DEPTH_DIR}/{i}.png")
        # np.save(f"{DEPTH_DIR}/{i}.png", depth)
        
        # save pose by copying the file
        os.system(f"cp {pose} {POSE_DIR}/{i}.txt")
        
        # save intrinsic by saving the numpy array
        file = open(f"{INTRINSIC_DIR}/{i}.txt", "w")
        for item in intrinsic:
            for i in range(len(item)):
                file.write(str(item[i]) + " ")
            file.write("\n")
        file.close()
        # np.save(f"{INTRINSIC_DIR}/{i}.txt", intrinsic)
        
        
def process_scene(scene):
    poses_path = os.path.join(DATA_PATH, scene, 'pose')
    depths_path = os.path.join(DATA_PATH, scene, 'depth')
    rgb_path = os.path.join(DATA_PATH, scene, 'color')
    intrinsic_path = os.path.join(DATA_PATH, scene, 'intrinsic')
    
    house_segmentation_path = os.path.join(HOUSE_SEGMENTATIONS, scene, scene, 'house_segmentations', 'panorama_to_region.txt')
    if not os.path.exists(house_segmentation_path):
        # unzip house_segmentations
        os.system(f"unzip {os.path.join(HOUSE_SEGMENTATIONS, scene, 'house_segmentations.zip')} -d {os.path.join(HOUSE_SEGMENTATIONS, scene)}")
        assert os.path.exists(house_segmentation_path)
        
    region_to_camera_mapping = region_to_camera_mapping_fn(house_segmentation_path)
    
    for region in region_to_camera_mapping:
        print(region, len(region_to_camera_mapping[region]))
        region_to_camera_mapping[region] = list(set(region_to_camera_mapping[region]))
        
    for region in region_to_camera_mapping:
        print(region, len(region_to_camera_mapping[region]))
        
        output_scene_name = f"{scene}_region{region}"
        
        load_raw_data(poses_path, depths_path, rgb_path, intrinsic_path, region_to_camera_mapping[region], output_scene_name)
        


if __name__=="__main__":
    
    scene_list = os.listdir(DATA_PATH)
    with mp.Pool(processes=32) as pool:
        pool.map(process_scene, scene_list)
    # scene_list  = ['17DRP5sb8fy']
    # for scene in scene_list:
        