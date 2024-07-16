# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from typing import Tuple
import numpy as np
from pathlib import Path
import ast

import torch
from torch import nn
from torch.nn import functional as F
from pytorch3d.ops import knn_points


from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.layers import ShapeSpec

from .modeling.criterion import ODINSetCriterion
from .modeling.matcher import ODINHungarianMatcher
from .utils.memory import retry_if_cuda_oom
from .utils.util_video_to_3d import convert_video_instances_to_3d

from odin.global_vars import LEARNING_MAP, LEARNING_MAP20, SCANNET200_LEARNING_MAP, \
    S3DIS_NAME_MAP, \
    MATTERPORT_LEARNING_MAP, LEARNING_MAP_INV, SCANNET200_LEARNING_MAP_INV, \
    MATTERPORT_ALL_CLASSES_TO_21

from odin.data_video.sentence_utils import convert_od_to_grounding_simple, convert_grounding_to_od_logits

from odin.modeling.backproject.backproject import \
    multiscsale_voxelize, interpolate_feats_3d, voxelization, voxel_map_to_source, \
    make_xyz_for_rgb
from odin.utils import vis_utils
from odin.modeling.backbone.resnet import build_resnet_backbone_custom
from torch_scatter import scatter_mean, scatter_min

from detectron2.utils.comm import get_world_size
from odin.utils.misc import is_dist_avail_and_initialized
from transformers import RobertaTokenizerFast, AutoTokenizer

from detectron2.data import MetadataCatalog
import detectron2.utils.comm as comm
import wandb

logger = logging.getLogger(__name__)

import ipdb 
st = ipdb.set_trace


@META_ARCH_REGISTRY.register()
class ODIN(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # video
        num_frames,
        decoder_3d,
        supervise_sparse,
        eval_sparse,
        cfg
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        if cfg.MODEL.FREEZE_BACKBONE:
            panet_resnet_layers = ['cross_view_attn', 'res_to_trans', 'trans_to_res']
            panet_swin_layers = ['cross_view_attn', 'cross_layer_norm', 'res_to_trans', 'trans_to_res']

            if cfg.MODEL.BACKBONE.NAME == "build_resnet_backbone":
                backbone_panet_layers = panet_resnet_layers
            elif cfg.MODEL.BACKBONE.NAME == "D2SwinTransformer":
                backbone_panet_layers = panet_swin_layers
            else:
                raise NotImplementedError

            for name, param in backbone.named_parameters():
                if any([layer in name for layer in backbone_panet_layers]):
                    print(f'Not freezing {name}')
                    continue
                else:
                    param.requires_grad = False
        
        if cfg.USE_WANDB:
            if not is_dist_avail_and_initialized() or comm.is_main_process():
                name = cfg.OUTPUT_DIR.split('/')[-1] if cfg.WANDB_NAME is None else cfg.WANDB_NAME
                wandb.init(
                    project="odin", sync_tensorboard=True,
                    name=name, id=name,
                    resume="allow"
                )
            
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.num_feature_levels = 3

        self.decoder_3d = decoder_3d
        self.supervise_sparse = supervise_sparse
        self.eval_sparse = eval_sparse
        self.cfg = cfg

        self.categories = {k: v for k, v in enumerate(self.metadata.thing_classes)}

        if self.cfg.MODEL.OPEN_VOCAB:
            self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
            

    @classmethod
    def from_config(cls, cfg):
        if cfg.MODEL.BACKBONE.NAME == "build_resnet_backbone":
            backbone = build_resnet_backbone_custom(cfg, ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN)))
        else:
            backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        # building criterion
        matcher = ODINHungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS_2D,
            supervise_sparse=cfg.MODEL.SUPERVISE_SPARSE,
            cfg=cfg
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        criterion = ODINSetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS_2D,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            supervise_sparse=cfg.MODEL.SUPERVISE_SPARSE,
            cfg=cfg
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": True,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "num_frames": cfg.INPUT.SAMPLING_FRAME_NUM,
            "decoder_3d": cfg.MODEL.DECODER_3D,
            "supervise_sparse": cfg.MODEL.SUPERVISE_SPARSE,
            "eval_sparse": cfg.TEST.EVAL_SPARSE,
            "cfg": cfg
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def load_3d_data(self, batched_inputs, images_shape, decoder_3d):
        depths, poses, segments, valids, intrinsics = None, None, None, None, None
        multiview_data = None
        bs, v = images_shape[:2]
        h, w = batched_inputs[0]['images'][0].shape[-2:]

        if self.supervise_sparse or self.eval_sparse:
            valids = torch.stack(
                [torch.stack(video["valids"]).to(self.device) for video in batched_inputs]
            ).reshape(bs, v, h, w)
            # pad the depth image with size divisibility
            if self.size_divisibility > 1:
                pad_h = int(np.ceil(h / self.size_divisibility) * self.size_divisibility - h)
                pad_w = int(np.ceil(w / self.size_divisibility) * self.size_divisibility - w)
                valids = F.pad(valids, (0, pad_w, 0, pad_h), mode="constant", value=0)

        if decoder_3d:
            depths = torch.stack(
                [torch.stack(video["depths"]).to(self.device) for video in batched_inputs]
            )
            intrinsics =  torch.stack(
                [torch.stack(video["intrinsics"]).to(self.device) for video in batched_inputs]
            )
            b, v, h, w = depths.shape
            # pad the depth image with size divisibility
            if self.size_divisibility > 1:
                pad_h = int(np.ceil(h / self.size_divisibility) * self.size_divisibility - h)
                pad_w = int(np.ceil(w / self.size_divisibility) * self.size_divisibility - w)
                depths = F.pad(depths, (0, pad_w, 0, pad_h), mode="constant", value=0)
            assert list(depths.shape[-2:]) == images_shape[-2:], "depth and image size mismatch"
             
            poses = torch.stack(
                [torch.stack(video["poses"]).to(self.device) for video in batched_inputs]
            )
            
            multiview_data = {}
            multiview_data["multi_scale_xyz"] = [
                torch.stack([batched_inputs[i]["multi_scale_xyz"][j] for i in range(bs)], dim=0).to(self.device) for j in range(len(batched_inputs[0]["multi_scale_xyz"]))
            ]

            voxel_size = self.cfg.INPUT.VOXEL_SIZE[::-1]

            if self.cfg.INPUT.VOXELIZE:
                multiview_data["multi_scale_p2v"] = multiscsale_voxelize(
                    multiview_data["multi_scale_xyz"], voxel_size
                )
        return depths, poses, valids, segments, intrinsics, multiview_data

    def load_scannet_data(self, batched_inputs, multiview_data, scannet_pc_list=None, do_knn=False, images=None, shape=None):
        scennet_pc_processed = []
        scannet_labels_processed = []
        scannet_segments_processed = []
        scannet_idxs = []
        scannet_p2vs = []
        scannet_gt_instances = []
        scannet_color_processed = []

        for i, batched_input in enumerate(batched_inputs):
            if 'scannet200' in batched_input['dataset_name']:
                learning_map = SCANNET200_LEARNING_MAP
            elif 'matterport' in batched_input['dataset_name']:
                learning_map = MATTERPORT_LEARNING_MAP
            elif 'scannet' in batched_input['dataset_name']:
                learning_map = LEARNING_MAP20 if self.cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES == 20 else LEARNING_MAP
            elif 's3dis' in batched_input['dataset_name']:
                learning_map = {k:k for k in S3DIS_NAME_MAP.keys()}
            elif 'ai2thor' in batched_input['dataset_name']:
                # we don't need learning map for ai2thor
                pass
            else:
                raise NotImplementedError
  
            if scannet_pc_list is not None:
                scannet_pc = scannet_pc_list[i].float()
            else:
                scannet_pc = batched_input["scannet_coords"].to(self.device).float()
            scannet_labels = batched_input['scannet_labels'].to(self.device)
            scannet_segments = batched_input['scannet_segments'].to(self.device)
            scannet_colors = batched_input['scannet_color'].to(self.device)
            scannet_colors = (scannet_colors - self.pixel_mean.squeeze()[None]) / self.pixel_std.squeeze()[None]

            if do_knn:
                assert self.training, "knn is only used during training - at test time we use full pc"
                our_pc = multiview_data["multi_scale_xyz"][3][i].to(self.device).flatten(0, 2)
                dists_idxs = knn_points(scannet_pc[None], our_pc[None])
                dists_close = dists_idxs.dists.squeeze() < self.cfg.KNN_THRESH
                
                if dists_close.sum() == 0:
                    print(f"scene_name: {batched_input['file_name'].split('/')[-3]}")

                scennet_pc_processed.append(scannet_pc[dists_close])
                scannet_labels_processed.append(scannet_labels[dists_close])
                segments = scannet_segments[dists_close]
                scannet_color_processed.append(scannet_colors[dists_close])
                scannet_segments_processed.append(segments)
                scannet_idxs.append(dists_close)
            else:
                scennet_pc_processed.append(scannet_pc)
                scannet_labels_processed.append(scannet_labels)
                scannet_color_processed.append(scannet_colors)
                scannet_segments_processed.append(batched_input['scannet_segments'].to(self.device))

            # voxelization
            if len(scennet_pc_processed[-1]) == 0:
                scennet_pc_processed[-1] = torch.zeros(1, 3, device=self.device, dtype=torch.float32)
                scannet_p2vs.append(torch.zeros(1, device=self.device, dtype=torch.int64))
                scannet_color_processed[-1] = torch.zeros(1, 3, device=self.device, dtype=torch.float32)
            else:
                scannet_p2v = voxelization(scennet_pc_processed[-1][None], 0.02)[0]
                scannet_p2vs.append(scannet_p2v)

            # extract masks and labels
            target_dict = {}

            if 'test' not in batched_input['dataset_name']:
                unique_instances = torch.unique(scannet_labels_processed[-1][:, 1])
                if len(unique_instances) > 0 and unique_instances[0] == -1:
                    unique_instances = unique_instances[1:]
                
                num_unique_instances = len(unique_instances)
                scannet_masks = []
                scannet_classes = []

                for k in range(num_unique_instances):
                    scannet_mask = scannet_labels_processed[-1][:, 1] == unique_instances[k]
                    class_label = scannet_labels_processed[-1][:, 0][scannet_mask][0]
                    if class_label.item() not in learning_map:
                        # print(f"{class_label.item()} not in learning map")
                        continue
                    class_label = learning_map[class_label.item()]
                    if class_label == 0:
                        continue
                    
                    if self.cfg.MATTERPORT_ALL_CLASSES_TO_21:
                        class_label = MATTERPORT_ALL_CLASSES_TO_21[class_label]

                    scannet_masks.append(scannet_mask)
                    scannet_classes.append(class_label - 1)

                if len(scannet_masks) == 0:
                    print("no valid masks, recovering...")
                    scannet_masks = torch.zeros((1, scennet_pc_processed[-1].shape[0]), device=self.device)
                    scannet_classes = torch.ones((1), device=self.device, dtype=torch.int64) * len(batched_input['all_classes']) - 1
                    if self.cfg.MODEL.OPEN_VOCAB:
                        positive_map = torch.zeros((1, self.cfg.MODEL.MAX_SEQ_LEN), device=self.device)
                        positive_map[:, -1] = 1.0
                        positive_map_od = torch.ones((self.cfg.MODEL.MAX_SEQ_LEN)) * -1
                        tokens_positive = [[]]
                        text_caption = ""

                else:
                    scannet_masks = torch.stack(scannet_masks, dim=0)
                    scannet_classes = torch.tensor(scannet_classes, device=self.device, dtype=torch.int64)

                    if self.cfg.MODEL.OPEN_VOCAB:
                        positive_map, positive_map_od, tokens_positive, text_caption = convert_od_to_grounding_simple(
                            scannet_classes.tolist(), batched_input['all_classes'], disable_shuffle=self.cfg.DISABLE_SHUFFLE or (not self.training), add_detection_prompt=False,
                            separation_tokens=". ", tokenizer=self.tokenizer, max_query_len=self.cfg.MODEL.MAX_SEQ_LEN,
                        ) 
                        positive_map = positive_map.to(self.device)

            # voxelized segments
            if self.cfg.USE_SEGMENTS:
                if len(scannet_segments_processed[-1]) == 0:
                    scannet_segments = torch.zeros(1, device=self.device)
                else:
                    scannet_segments = scannet_segments_processed[-1]
                scannet_p2v = scannet_p2vs[-1]
                scannet_segments = scatter_min(
                    scannet_segments, scannet_p2vs[-1], dim=0
                )[0]
                
                unique_segments, inverse_indices = torch.unique(scannet_segments, return_inverse=True)
                new_segments = torch.arange(unique_segments.shape[0], device=self.device)[inverse_indices]
                segments = new_segments
                scannet_segments_processed[-1] = segments

                if 'test' not in batched_input['dataset_name']:
                    # voxelized masks
                    segment_mask = scatter_mean(
                        scannet_masks.float(), scannet_p2v[None], dim=1
                    )
                    # segment masks
                    segment_mask = scatter_mean(
                        segment_mask, segments[None], dim=1
                    ) > 0.5
                    voxel_masks = segment_mask
            else:
                if 'test' not in batched_input['dataset_name']:
                    voxel_masks = scatter_mean(
                        scannet_masks.float(), scannet_p2vs[-1][None], dim=1
                    ) > 0.5
            
            if 'test' not in batched_input['dataset_name']:
                target_dict = {
                    "masks": scannet_masks.float(),
                    "labels": scannet_classes,
                    "p2v": scannet_p2vs[-1],
                    "max_valid_points": scannet_masks.shape[1],
                    "segments": segments if self.cfg.USE_SEGMENTS else None,
                    "segment_mask": voxel_masks if self.cfg.USE_SEGMENTS else None,
                    "voxel_masks": voxel_masks,
                    "positive_map": positive_map if self.cfg.MODEL.OPEN_VOCAB else None, # self.cfg.MODEL.MAX_SEQ_LEN
                    "tokens_positive": tokens_positive if self.cfg.MODEL.OPEN_VOCAB else None, # list
                    "positive_map_od": positive_map_od if self.cfg.MODEL.OPEN_VOCAB else None, # N X self.cfg.MODEL.MAX_SEQ_LEN
                    "text_caption": text_caption if self.cfg.MODEL.OPEN_VOCAB else None,
                }
            else:
                target_dict = {
                    "p2v": scannet_p2vs[-1],
                    "max_valid_points": scannet_p2vs[-1].shape[0],
                    "segments": segments if self.cfg.USE_SEGMENTS else None,
                }
            scannet_gt_instances.append(target_dict)

        # get max points
        valid_points = [x.shape[0] for x in scennet_pc_processed]
        max_points = max(valid_points)

        # batch the points and labels
        scannet_pc_batched = torch.zeros(len(scennet_pc_processed), max_points, 3, device=self.device)
        scannet_color_batched = torch.zeros(len(scennet_pc_processed), max_points, 3, device=self.device)
        scannet_p2v_batched = torch.zeros(len(scannet_p2vs), max_points, dtype=torch.int64, device=self.device)

        valid_segment_points = [x.shape[0] for x in scannet_segments_processed]
        max_segment_points = max(valid_segment_points)
        voxel_sizes = np.array([x.max().item() for x in scannet_p2vs])
        max_voxel_size = voxel_sizes == voxel_sizes.max()
        max_voxel_valid_points =  np.array(valid_points)[np.nonzero(max_voxel_size)[0]]

        if (max_voxel_valid_points < max_points).any():
            max_segment_points += 1 # extra segment due to voxel padding
        scannet_segments_batched = torch.zeros(len(scannet_segments_processed), max_segment_points, dtype=torch.int64, device=self.device)

        for j, (pc, color, p2v, segments) in enumerate(zip(scennet_pc_processed, scannet_color_processed, scannet_p2vs, scannet_segments_processed)):
            # st()
            scannet_pc_batched[j, :pc.shape[0]] = pc
            scannet_pc_batched[j, pc.shape[0]:] = -10

            scannet_color_batched[j, :color.shape[0]] = color
            scannet_color_batched[j, color.shape[0]:] = -10

            scannet_segments_batched[j, :segments.shape[0]] = segments
            if segments.shape[0] != 0:
                scannet_segments_batched[j, segments.shape[0]:] = segments.max() + 1
            
            scannet_p2v_batched[j, :p2v.shape[0]] = p2v
            if p2v.shape[0] != 0:
                scannet_p2v_batched[j, p2v.shape[0]:] = p2v.max() + 1

        return scannet_pc_batched, scannet_p2v_batched, scannet_gt_instances, scannet_idxs, scannet_segments_batched, scannet_color_batched
                
    def adjust_masks_for_highres(self, multiview_data, targets, idxs, shape, segments):
        bs, v, H_padded, W_padded = shape
        if self.training:
            for i in range(len(targets)):
                # handle cases where there are no valid masks
                if targets[i]['masks'].shape[0] == 0:
                    # print("no valid masks, recovering...")
                    targets[i]['masks'] = torch.zeros(1, targets[i]['masks'].shape[1], targets[i]['masks'].shape[2], targets[i]['masks'].shape[3], device=targets[i]['masks'].device, dtype=torch.bool)
                    targets[i]['labels'] = torch.ones(1, device=targets[i]['masks'].device, dtype=torch.long) * (self.cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES)
                
                if self.supervise_sparse:
                    targets[i]['valids'] = F.interpolate(
                        targets[i]['valids'][None].float(),
                        size=(H_padded // 4, W_padded // 4),
                        mode='nearest',
                    )[0]
                    if self.cfg.INPUT.VOXELIZE:
                        p2v = multiview_data['original_color_p2v'][i] if (self.cfg.HIGH_RES_INPUT and self.cfg.HIGH_RES_SUBSAMPLE) else multiview_data['multi_scale_p2v'][-1][i]
                        targets[i]['valids'] = scatter_mean(
                            targets[i]['valids'].flatten().float(),
                            p2v,
                            dim=0,
                        ) > 0.5
                        if targets[i]['valids'].sum() == 0:
                            targets[i]['valids'][0] = True
                    else:
                        targets[i]['valids'] = (targets[i]['valids'] > 0.5)
                        
                if self.cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS != -1 and self.cfg.HIGH_RES_SUBSAMPLE and self.cfg.INPUT.VOXELIZE:
                    mask = F.interpolate(
                        targets[i]['masks'].float(),
                        size=(H_padded // 4, W_padded // 4),
                        mode='nearest',
                    )
                    targets[i]['voxel_masks'] = scatter_mean(
                        mask.flatten(1),
                        multiview_data['original_color_p2v'][i][None],
                        dim=1,
                    )[:, idxs[i]] > 0.5
                    if self.supervise_sparse:
                        targets[i]['valids'] = targets[i]['valids'][idxs[i]]
                        if targets[i]['valids'].sum() == 0:
                            targets[i]['valids'][0] = True
                    
                else:
                    if self.cfg.INPUT.VOXELIZE:
                        targets[i]['voxel_masks'] = F.interpolate(
                            targets[i]['masks'].float(),
                            size=(H_padded // 4, W_padded // 4),
                            mode='nearest',
                        ).flatten(1)
                        targets[i]['voxel_masks'] = scatter_mean(
                            targets[i]['voxel_masks'].float(),
                            multiview_data['multi_scale_p2v'][-1][i][None],
                            dim=1,
                        ) > 0.5
                            
                        if self.cfg.USE_SEGMENTS:
                            segment_mask = scatter_mean(
                                targets[i]['voxel_masks'].float(), segments[i][:targets[i]['voxel_masks'].shape[1]][None], dim=1
                            ) > 0.5
                            targets[i]['segment_mask'] = segment_mask
                    else:
                        targets[i]['masks'] = F.interpolate(
                            targets[i]['masks'].float(),
                            size=(H_padded // 4, W_padded // 4),
                            mode='nearest',
                        ) > 0.5
        return targets
        
    def get_color_features(self, multiview_data, images, features, shape):
        bs, v, H_padded, W_padded = shape
        idxs = None

        if self.cfg.HIGH_RES_INPUT and self.cfg.INPUT.VOXELIZE and self.cfg.HIGH_RES_SUBSAMPLE and self.training and self.cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS != -1:
            color_features = features['res2']
            color_features = scatter_mean(
                color_features.reshape(bs, v, -1, color_features.shape[-2], color_features.shape[-1]).permute(0, 1, 3, 4, 2).flatten(1, 3),
                multiview_data['multi_scale_p2v'][-1],
                dim=1,
            )
            color_features_xyz = multiview_data['multi_scale_xyz'][-1]
            color_features_p2v = multiview_data['multi_scale_p2v'][-1]

            original_p2v = multiview_data['multi_scale_p2v'][-1]
            multiview_data['original_color_p2v'] = multiview_data['multi_scale_p2v'][-1]
            
            color_features_xyz = scatter_mean(
                color_features_xyz.flatten(1, 3),
                color_features_p2v,
                dim=1,
            )
            color_features_p2v = torch.arange(color_features_xyz.shape[1], device=color_features_xyz.device).unsqueeze(0).repeat(bs, 1)
            multiview_data['multi_scale_xyz'][-1] = color_features_xyz[:, None, None]
            multiview_data['multi_scale_p2v'][-1] = color_features_p2v

            idxs = []
            new_color_features = []
            new_color_features_xyz = []
            new_color_features_p2v = []
            for i in range(bs):
                # idx = torch.randperm(color_features_p2v[i].max() + 1, device=color_features_xyz.device)[:self.cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS]
                idx = torch.randint(0, original_p2v[i].max() + 1, (self.cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,), device=color_features_xyz.device)
                new_color_features.append(color_features[i, idx])
                new_color_features_xyz.append(color_features_xyz[i, idx])
                new_color_features_p2v.append(torch.arange(idx.shape[0], device=color_features_xyz.device))
                idxs.append(idx)
            color_features = torch.stack(new_color_features) # b, n, c
            color_features_xyz = torch.stack(new_color_features_xyz) # b, n, 3
            color_features_p2v = torch.stack(new_color_features_p2v)
            multiview_data['multi_scale_xyz'][-1] = color_features_xyz[:, None, :, None]
            multiview_data['multi_scale_p2v'][-1] = color_features_p2v
            features['res2'] = color_features.permute(0, 2, 1)[..., None]

        return idxs
    
    def upsample_pred_masks(
            self, mask_pred_results, batched_inputs,
            multiview_data, shape, downsample=False, interp="bilinear"
        ):
        bs, v, H_padded, W_padded = shape
        assert bs == 1
        if interp == "trilinear":
            target_xyz = batched_inputs['original_xyz'][None].to(self.device)
            if downsample:
                target_xyz = F.interpolate(
                    target_xyz[0].permute(0, 3, 1, 2),
                    scale_factor=0.5,
                    mode='nearest'
                ).permute(0, 2, 3, 1).reshape(bs, v, target_xyz.shape[2] // 2, target_xyz.shape[3] // 2, target_xyz.shape[4])
            target_p2v = torch.arange(
                target_xyz.flatten(1, 3).shape[1], device=self.device)[None]
            source_xyz = multiview_data['multi_scale_xyz']
            source_p2v = multiview_data['multi_scale_p2v']

            mask_pred_results = mask_pred_results[:, source_p2v][None]

            mask_pred_results = mask_pred_results.permute(0, 2, 1)
            source_xyz = source_xyz.flatten(0, 2)[None]
            source_p2v = source_p2v[None]
            B, _, Q = mask_pred_results.shape

            mask_pred_results = interpolate_feats_3d(
                mask_pred_results,
                source_xyz, source_p2v,
                target_xyz, target_p2v,
                shape=[bs, v],
                num_neighbors=self.cfg.INTERP_NEIGHBORS,
                voxelize=True
            ).reshape(target_xyz.shape[1], Q, target_xyz.shape[-3],
                target_xyz.shape[-2]).permute(1, 0, 2, 3).to(mask_pred_results.dtype)
        elif interp == "bilinear":
            Q, N, H, W = mask_pred_results.shape
            img_size = (H_padded, W_padded)
            if downsample:
                img_size = (img_size[0] // 2, img_size[1] // 2)

            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(img_size[0], img_size[1]),
                mode="bilinear",
                align_corners=False,
            )
        else:
            raise NotImplementedError(f"interp must be either trilinear or bilinear, got {interp}")

        return mask_pred_results

    def export_pred_benchmark(self, processed_results, scene_name, dataset_name):
        # for instance segmentation
        root_path = self.cfg.EXPORT_BENCHMARK_PATH
        if 'scannet200' in dataset_name:
            learning_map_inv = SCANNET200_LEARNING_MAP_INV
        else:
            learning_map_inv = LEARNING_MAP_INV
            
        base_path = f"{root_path}/instance_evaluation"
        pred_mask_path = f"{base_path}/pred_mask"

        Path(pred_mask_path).mkdir(parents=True, exist_ok=True)

        file_name = scene_name
        with open(f"{base_path}/{file_name}.txt", "w") as fout:
            real_id = -1
            pred_classes = processed_results['instances_3d']["pred_classes"].cpu().numpy()
            scores = processed_results['instances_3d']["pred_scores"].cpu().numpy()
            pred_masks = processed_results['instances_3d']["pred_masks"].cpu().numpy()
            for instance_id in range(len(pred_classes)):
                real_id += 1
                pred_class = pred_classes[instance_id]
                pred_class =  learning_map_inv[pred_class]
                score = scores[instance_id]
                mask = pred_masks[:, instance_id].astype("uint8")

                np.savetxt(f"{pred_mask_path}/{file_name}_{real_id}.txt", mask, fmt="%d")
                fout.write(f"pred_mask/{file_name}_{real_id}.txt {pred_class} {score}\n")
        
        # for semantic segmentation
        base_path = f"{root_path}/semantic_evaluation"
        Path(base_path).mkdir(parents=True, exist_ok=True)
        
        pred_mask_path = f"{base_path}/{scene_name}.txt"
        
        with open(pred_mask_path, "w") as fout:
            pred_mask = processed_results['semantic_3d'].cpu().numpy()
            pred_mask = np.array([learning_map_inv[x + 1] for x in pred_mask])
            np.savetxt(pred_mask_path, pred_mask, fmt="%d")     
            
        torch.cuda.empty_cache()
     
    def eval_ghost(
            self, mask_cls_results, mask_pred_results, batched_inputs,
            scannet_gt_target_dicts, scannet_p2v, num_classes, 
            scannet_idxs, segments
        ):
        processed_results = []
        
        if self.cfg.USE_SEGMENTS:
            mask_pred_results = voxel_map_to_source(
                mask_pred_results.permute(0, 2, 1),
                segments
            ).permute(0, 2, 1)
            
        pred_masks = mask_pred_results
        for i, pred_mask in enumerate(pred_masks):

            pred_mask = pred_mask[:, scannet_p2v[i]]

            # remove padding
            max_valid_point = scannet_gt_target_dicts[i]['max_valid_points']
            pred_mask = pred_mask[:, :max_valid_point]

            if self.cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
                processed_3d = self.inference_scannet_ghost(
                    pred_mask, mask_cls_results[i], num_classes=num_classes)
                    
                if 'test' not in batched_inputs[i]['dataset_name']:
                    processed_3d["scannet_gt_masks"] = scannet_gt_target_dicts[i]['masks']
                    processed_3d["scannet_gt_classes"] = scannet_gt_target_dicts[i]['labels'] + 1
                    processed_3d["max_valid_points"] = max_valid_point
                processed_3d = {"instances_3d": processed_3d}

            if self.cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
                semantic_r = retry_if_cuda_oom(self.inference_scannet_ghost_semantic)(
                mask_cls_results[i], pred_mask)
                processed_3d["semantic_3d"] = semantic_r
            
            if self.cfg.MATTERPORT_ALL_CLASSES_TO_21:
                matterport_all_classes_to_21 = torch.tensor(list(MATTERPORT_ALL_CLASSES_TO_21.values()), device=pred_mask.device)
                processed_3d['instances_3d']['pred_classes'] = matterport_all_classes_to_21[processed_3d['instances_3d']['pred_classes'] - 1]
                processed_3d["semantic_3d"] = matterport_all_classes_to_21[processed_3d["semantic_3d"]] - 1
                
            processed_results.append(processed_3d)
        
            if self.cfg.EXPORT_BENCHMARK_DATA:
                self.export_pred_benchmark(
                    processed_results[-1], batched_inputs[i]['file_name'].split('/')[-3], dataset_name=batched_inputs[i]['dataset_name'])
                return None
                
            if self.cfg.VISUALIZE:
                self.visualize_pred_on_scannet(
                        batched_inputs[i], processed_results[i], scannet_gt_target_dicts,
                        index=i, scannet_idxs=scannet_idxs[i] if len(scannet_idxs) > 0 else None,
                    )
        torch.cuda.empty_cache()
        return processed_results
       
    def eval_normal(
            self, mask_cls_results, mask_pred_results, batched_inputs,
            images, shape, targets, num_classes, decoder_3d,
            multiview_data
        ):
        bs, v, H_padded, W_padded = shape
        processed_results = []

        for i, (mask_cls_result, mask_pred_result, input_per_image, image_size) in enumerate(zip(
            mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
        )):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            shape_ = [1, v, H_padded, W_padded]
            
            multiview_data_ = None
            if multiview_data is not None:
                multiview_data_ = {}
                multiview_data_['multi_scale_xyz'] = multiview_data['multi_scale_xyz'][-1][i]
                if self.cfg.INPUT.VOXELIZE:
                    multiview_data_['multi_scale_p2v'] = multiview_data['multi_scale_p2v'][-1][i]

            if self.eval_sparse:
                valids = input_per_image.get("valids")
                valids = torch.stack(valids).reshape(v, height, width)

            processed_results.append({})

            if self.cfg.EVAL_PER_IMAGE and decoder_3d:
                instance_r = self.inference_2d_per_image(
                    mask_cls_result, mask_pred_result,
                    (height, width), shape_,
                    batched_inputs[i],
                    multiview_data=multiview_data_
                )
                processed_results[-1]['instances'] = instance_r
                
            if self.cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON and decoder_3d:
                semantic_r = self.inference_video_semantic(
                    mask_cls_result, mask_pred_result, image_size,
                    valids if self.eval_sparse else None,
                    batched_inputs[i], multiview_data_, shape_
                )
                processed_results[-1]["semantic_3d"] = semantic_r
                
            output_img_size = None
            
            if 'coco' in input_per_image['dataset_name']:
                output_img_size = [input_per_image.get("height"), input_per_image.get('width')]
                height, width = image_size
            
            if self.cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
                instance_r = self.inference_video(
                    mask_cls_result, mask_pred_result,
                    height, width,
                    valids if self.eval_sparse else None,
                    decoder_3d=decoder_3d, num_classes=num_classes,
                    batched_inputs=batched_inputs[i], multiview_data=multiview_data_,
                    shape=shape_, 
                    output_img_size=output_img_size)

                if decoder_3d:
                    processed_results[-1]["instances_3d"] = instance_r["3d"]
                
                if not decoder_3d:
                    processed_results[-1]["instances"] = instance_r["2d"]

            if self.cfg.VISUALIZE_3D and decoder_3d:
                self.visualize_pred_on_ours(
                    i, images, [bs, v, H_padded, W_padded],
                    input_per_image, processed_results[-1], 
                    targets, valids
                )   
    
        return processed_results
          
    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one scene. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """    
        images = []
        for video in batched_inputs:
            for image in video["images"]:
                images.append(image.to(self.device))
        bs = len(batched_inputs)
        v = len(batched_inputs[0]["images"])
        
        # important to check this when joint joint training
        decoder_3d = torch.tensor(sum([batched_input["decoder_3d"] for batched_input in batched_inputs]), device=self.device)
        if self.cfg.MULTI_TASK_TRAINING and self.training:
            eff_bs = len(batched_inputs)
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(decoder_3d)
                eff_bs *= get_world_size()
            decoder_3d = decoder_3d.item()
            assert decoder_3d == eff_bs or decoder_3d == 0, f"All videos must have the same decoder_3d value {decoder_3d}"


        decoder_3d = decoder_3d > 0

        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        H_padded, W_padded = images.tensor.shape[-2:]

        depths, poses, intrinsics, valids, segments = None, None, None, None, None
        
        depths, poses, valids, segments, intrinsics, multiview_data = self.load_3d_data(
            batched_inputs,
            images_shape=[bs, v, H_padded, W_padded],
            decoder_3d=decoder_3d
        )        

        if self.cfg.MODEL.CROSS_VIEW_BACKBONE and decoder_3d:
            features = self.backbone(
                images.tensor,
                multiview_data['multi_scale_xyz'],
                shape=[bs, v, H_padded, W_padded],
                multiview_data=multiview_data, 
                decoder_3d=decoder_3d
            )
        else:
            features = self.backbone(images.tensor, decoder_3d=decoder_3d)
        
        # check for nans in features
        if torch.isnan(features['res4']).any():
            st()
        
        idxs = None
        if decoder_3d and not self.cfg.USE_GHOST_POINTS:
            idxs = self.get_color_features(
                multiview_data, images, features, shape=[bs, v, H_padded, W_padded]
            )

        scannet_pc, scannet_gt_target_dicts, scannet_p2v, \
            scannet_segments_batched, scannet_color_batched = None, None, None, None, None
        if self.cfg.USE_GHOST_POINTS  and decoder_3d:
            # scannet_pc = multiview_data['scannet_pc']
            full_scene_dataset = 'single' in batched_inputs[0]['dataset_name']
            scannet_pc, scannet_p2v, scannet_gt_target_dicts, scannet_idxs, \
                scannet_segments_batched, scannet_color_batched = self.load_scannet_data(
                batched_inputs, multiview_data, do_knn=self.training or not full_scene_dataset,
                images=images, shape=[bs, v, H_padded, W_padded]
            )

        # mask classification target
        if self.cfg.USE_GHOST_POINTS and decoder_3d:
            targets = scannet_gt_target_dicts
        else:
            targets = self.prepare_targets(
                batched_inputs, images, valids,
            )
            
        if decoder_3d and not self.cfg.USE_GHOST_POINTS:
            targets = self.adjust_masks_for_highres(
                multiview_data, targets, idxs, shape=[bs, v, H_padded, W_padded],
                segments=segments
            )
        
        captions = None
        num_classes = None
        if self.cfg.MODEL.OPEN_VOCAB:
            captions = [targets[i]['text_caption'] for i in range(len(targets))]
            num_classes = max([batched_input['num_classes'] for batched_input in batched_inputs]) + 1
        
        scene_names = [batched_inputs[i]['file_name'].split('/')[-3] for i in range(bs)]
        
        outputs = self.sem_seg_head(
            features,
            shape=[bs, v, H_padded, W_padded],
            multiview_data=multiview_data,
            scannet_pc=scannet_pc,
            scannet_p2v=scannet_p2v,
            segments=scannet_segments_batched if self.cfg.USE_GHOST_POINTS else segments,
            decoder_3d=decoder_3d, 
            captions=captions,
            positive_map_od=[targets[i]['positive_map_od'] for i in range(len(targets))] if self.cfg.MODEL.OPEN_VOCAB and not self.cfg.DETIC else None,
            num_classes=num_classes, 
            scene_names=scene_names
        )
        
        if self.training:
            # bipartite matching-based loss
            losses = self.criterion(outputs, targets, decoder_3d=decoder_3d)
        
            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            if self.cfg.EVAL_PER_IMAGE:
                torch.cuda.empty_cache()
                
            assert len(set([len(batched_input['all_classes']) for batched_input in batched_inputs])) == 1, "all targets should have the same number of classes"
            num_classes = max([batched_input['num_classes'] for batched_input in batched_inputs])
            
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            del outputs
                
            # Processing for ghost points
            if self.cfg.USE_GHOST_POINTS and decoder_3d:
                processed_results = self.eval_ghost(
                    mask_cls_results, mask_pred_results, batched_inputs,
                    scannet_gt_target_dicts, scannet_p2v, num_classes,
                    scannet_idxs, scannet_segments_batched
                )
                return processed_results
            
            # Normal Processing
            processed_results = self.eval_normal(
                mask_cls_results, mask_pred_results, batched_inputs,
                images, [bs, v, H_padded, W_padded], targets,
                num_classes, decoder_3d, multiview_data
            )
                
            return processed_results

    def visualize_pred_on_scannet(
            self, input_per_image, processed_result,
            gt_targets, index, scannet_idxs=None
    ):
        pc = input_per_image['scannet_coords'].cpu().numpy()
        if scannet_idxs is not None:
            pc = pc[scannet_idxs.cpu().numpy()]

        color = (input_per_image['scannet_color'] / 255.0).cpu().numpy()
        color = np.clip(color, 0, 1)
        if scannet_idxs is not None:
            color = color[scannet_idxs.cpu().numpy()]

        scene_name = input_per_image['file_name'].split('/')[-3]
        pred_scores = processed_result["instances_3d"]['pred_scores']
        pred_masks = processed_result["instances_3d"]['pred_masks']
        pred_labels = processed_result["instances_3d"]['pred_classes']

        # sort by scores in ascending order
        sort_idx = torch.argsort(pred_scores)
        pred_masks = pred_masks.permute(1, 0)[sort_idx].cpu().numpy()
        pred_labels = pred_labels[sort_idx].cpu().numpy()

        # threshold by scores > 0.5
        pred_scores = pred_scores[sort_idx].cpu().numpy()
        conf = pred_scores > 0.5
        pred_masks = pred_masks[conf]
        pred_labels = pred_labels[conf]

        gt_masks = gt_targets[index]['masks'].cpu().numpy()
        if "max_valid_points" in gt_targets[index]:
            max_valid_point = gt_targets[index]["max_valid_points"]
            gt_masks = gt_masks[:, :max_valid_point]

        gt_labels = gt_targets[index]['labels'].cpu().numpy()

        valids = np.ones_like(pc[:, 0]).astype(bool)

        dataset_name = input_per_image['dataset_name']

        vis_utils.plot_3d_offline(
            pc, color, masks=pred_masks, valids=valids,
            labels=pred_labels,
            gt_masks=gt_masks, gt_labels=gt_labels, scene_name=scene_name,
            data_dir=self.cfg.VISUALIZE_LOG_DIR,
            mask_classes=self.cfg.SKIP_CLASSES, dataset_name=dataset_name,
        )

    def visualize_pred_on_ours(
        self, index, images,
        shape,
        input_per_image, processed_results, targets, valids):

        bs, v, H_padded, W_padded = shape
        our_pc = input_per_image['original_xyz']
        if valids is not None:
            our_pc = our_pc[0][valids]
        else:
            if self.cfg.HIGH_RES_INPUT:
                our_pc = F.interpolate(
                    our_pc.float().permute(0, 3, 1, 2), scale_factor=0.5, mode="nearest"
                ).permute(0, 2, 3, 1).reshape(-1, 3)
                our_pc = our_pc.cpu().numpy()

        vis_images = images.tensor * self.pixel_std + self.pixel_mean
        vis_images = vis_images.view(bs, v, 3, H_padded, W_padded)[index]
        if self.cfg.HIGH_RES_INPUT:
            vis_images = F.interpolate(
                vis_images, scale_factor=0.5, mode="bilinear"
            )
        
        if valids is not None:
            color = vis_images.permute(0, 2, 3, 1)[valids].cpu().numpy() / 255.0
        else:
            color = vis_images.permute(0, 2, 3, 1).reshape(-1, 3).cpu().numpy() / 255.0
        color = np.clip(color, 0, 1)

        scene_name = input_per_image['file_name'].split('/')[-3]
       
        pred_scores = processed_results["instances_3d"]['pred_scores']
        pred_masks = processed_results["instances_3d"]['pred_masks']
        pred_labels = processed_results["instances_3d"]['pred_classes']

        sort_idx = torch.argsort(pred_scores)
        pred_masks = pred_masks.permute(1, 0)[sort_idx].cpu().numpy()
        pred_labels = pred_labels[sort_idx].cpu().numpy()

        # select confident predictions
        pred_scores = pred_scores[sort_idx].cpu().numpy()

        conf = pred_scores > 0.5
        pred_masks = pred_masks[conf]
        pred_labels = pred_labels[conf]


        gt_masks = targets[index]['masks']
        if valids is not None:
            gt_masks = gt_masks[:, valids].cpu().numpy()
        else:
            if self.cfg.HIGH_RES_INPUT:
                gt_masks = F.interpolate(
                    gt_masks.float(), scale_factor=0.5, mode="nearest"
                ) > 0.5
                gt_masks = gt_masks.flatten(1)
            gt_masks = gt_masks.cpu().numpy()
        gt_labels = targets[index]['labels'].cpu().numpy()

        valids = np.zeros_like(our_pc[:, 0]).astype(bool)

        valid_idx = np.random.choice(
            np.arange(valids.shape[0]), 200000)
        valids[valid_idx] = True

        dataset_name = input_per_image['dataset_name']

        vis_utils.plot_3d_offline(
            our_pc, color, masks=pred_masks, valids=valids,
            labels=pred_labels,
            gt_masks=gt_masks, gt_labels=gt_labels, scene_name=scene_name,
            data_dir=self.cfg.VISUALIZE_LOG_DIR, 
            mask_classes=self.cfg.SKIP_CLASSES, dataset_name=dataset_name,
        )     

    def prepare_targets(
            self, targets, images, valids=None
        ):
        if type(images) == ImageList:
            h_pad, w_pad = images.tensor.shape[-2:]
        else:
            h_pad, w_pad = images.shape[-2:]
        gt_instances = []
        for targets_per_video in targets:
            num_frames = len(targets_per_video["images"])            
        
        for i, targets_per_video in enumerate(targets):
            target_dict = convert_video_instances_to_3d(
                targets_per_video["instances_all"], num_frames, h_pad, w_pad, self.device,
                multiplier=targets_per_video['multiplier']
            )
            
            if self.cfg.MODEL.OPEN_VOCAB:
                positive_map, positive_map_od, tokens_positive, text_caption = convert_od_to_grounding_simple(
                        target_dict['labels'].tolist(), targets[i]['all_classes'], disable_shuffle=self.cfg.DISABLE_SHUFFLE or (not self.training), add_detection_prompt=False,
                        separation_tokens=". ", tokenizer=self.tokenizer, max_query_len=self.cfg.MODEL.MAX_SEQ_LEN,
                    ) 
                positive_map = positive_map.to(self.device)
                target_dict['positive_map'] = positive_map
                target_dict['tokens_positive'] = tokens_positive
                target_dict['positive_map_od'] = positive_map_od
                target_dict['text_caption'] = text_caption
            
            if valids is not None:
                target_dict.update({"valids": valids[i]})
                
            gt_instances.append(target_dict)

        return gt_instances

    def prepare_2d(
        self, pred_masks, img_size,
        labels_per_image, scores_per_image, 
        batched_inputs=None, multiview_data=None, shape=None, decoder_3d=False,
        output_img_size=None):
        pred_masks = self.upsample_pred_masks(
            pred_masks, batched_inputs, multiview_data, shape,
            downsample=False, interp="trilinear" if decoder_3d else "bilinear"
        )
        
        context_img_id = pred_masks.shape[1] // 2
        pred_masks = pred_masks[:, context_img_id]
        
        pred_masks = pred_masks[:, :img_size[0], :img_size[1]]
        
        if output_img_size is not None:
            pred_masks = F.interpolate(
                pred_masks[None], size=output_img_size, mode="bilinear"
            )[0]

        masks = pred_masks > 0.        
        image_size = masks.shape[-2:]
        
        mask_scores_per_image = (pred_masks.sigmoid().flatten(1) * masks.flatten(1)).sum(1) / (masks.flatten(1).sum(1) + 1e-6)

        result_2d = Instances(image_size)
        result_2d.pred_masks = masks
        result_2d.pred_boxes = Boxes(torch.zeros(masks.size(0), 4))
        mask_scores_per_image = (pred_masks.sigmoid().flatten(1) * result_2d.pred_masks.flatten(1)).sum(1) / (result_2d.pred_masks.flatten(1).sum(1) + 1e-6)
        result_2d.scores = scores_per_image * mask_scores_per_image
        result_2d.pred_classes = labels_per_image
        return result_2d

    def prepare_3d(
        self, pred_masks, output_height,
        output_width, labels_per_image, scores_per_image,
        valids=None, batched_inputs=None, multiview_data=None, shape=None
        ):
        
        pred_masks = self.upsample_pred_masks(
            pred_masks, batched_inputs, multiview_data, shape,
            downsample=self.cfg.HIGH_RES_INPUT, interp="trilinear"
        )

        if valids is not None:
            # downsample valids
            if self.size_divisibility > 1:
                h, w = output_height, output_width
                pad_h = int(np.ceil(h / self.size_divisibility) * self.size_divisibility - h)
                pad_w = int(np.ceil(w / self.size_divisibility) * self.size_divisibility - w)
                valids = F.pad(valids, (0, pad_w, 0, pad_h), mode="constant", value=0)
            H, W = pred_masks.shape[-2:]
            valids = F.interpolate(
                valids.float().unsqueeze(0), size=(H, W), mode='nearest').squeeze(0).bool()
        
        if valids is not None:
            pred_masks = pred_masks[:, valids]

        masks = pred_masks > 0.
        mask_scores_per_image = (pred_masks.sigmoid().flatten(1) * masks.flatten(1)).sum(1) / (masks.flatten(1).sum(1) + 1e-6)
               
        # add +1 to labels as mask3d evals from 1-18
        result_3d = {
            "pred_classes": labels_per_image + 1,
            "pred_masks": masks.flatten(1).permute(1, 0),
            "pred_scores": scores_per_image * mask_scores_per_image
        }
        return result_3d
        
    def inference_2d_per_image(self, pred_cls, pred_masks, img_size,
        shape, batched_inputs, multiview_data=None):
        """
        pred_cls: 100 X 19
        pred_masks: 100 X 5 X 480 X 640
        """
        pred_masks = self.upsample_pred_masks(
            pred_masks, batched_inputs, multiview_data, shape,
            downsample=False, interp="trilinear"
        )
        test_topk_per_image = self.cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 
        if not self.cfg.MODEL.OPEN_VOCAB:
            scores = F.softmax(pred_cls, dim=-1)[:, :-1]
        else:
            scores = pred_cls[:, :-1]

        num_classes = self.sem_seg_head.num_classes

        if self.cfg.SKIP_CLASSES is not None:
            num_classes -= len(self.cfg.SKIP_CLASSES)
            skip_classes = torch.tensor(self.cfg.SKIP_CLASSES, device=self.device) - 1

            # +1 for background class
            keep_class_mask = torch.ones(self.cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES, device=self.device)
            keep_class_mask[skip_classes] = 0
            scores = scores[:, keep_class_mask.bool()]

        labels = torch.arange(num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // num_classes
        
        num_images = pred_masks.shape[1]
        result_2ds = []
        for i in range(num_images):
            pred_masks_ = pred_masks[:, i][topk_indices]
            pred_masks_ = pred_masks_[:, :img_size[0], :img_size[1]]
            masks = pred_masks_ > 0.        
            image_size = masks.shape[-2:]
            
            assert image_size[0] == img_size[0] and image_size[1] == img_size[1], "image size should be the same"
            
            result_2d = Instances(image_size)
            result_2d.pred_masks = masks
            result_2d.pred_boxes = Boxes(torch.zeros(masks.size(0), 4))
            mask_scores_per_image = (pred_masks_.sigmoid().flatten(1) * result_2d.pred_masks.flatten(1)).sum(1) / (result_2d.pred_masks.flatten(1).sum(1) + 1e-6)
            result_2d.scores = (scores_per_image * mask_scores_per_image).cpu()
            result_2d.pred_classes = labels_per_image.cpu()
            result_2d.pred_masks = result_2d.pred_masks.cpu()
            result_2ds.append(result_2d)
        return result_2ds

    def inference_video(
        self, pred_cls, pred_masks,
        output_height, output_width, valids=None,
        decoder_3d=False, num_classes=None,
        batched_inputs=None, multiview_data=None, shape=None,
        output_img_size=None
    ):
        """
        pred_cls: 100 X 19
        pred_masks: 100 X 5 X 480 X 640
        """
        test_topk_per_image = self.cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 
        if not self.cfg.MODEL.OPEN_VOCAB or self.cfg.NON_PARAM_SOFTMAX:
            scores = F.softmax(pred_cls, dim=-1)[:, :-1]
        else:
            scores = pred_cls[:, :-1]
            
        assert scores.min() >= 0 and scores.max() <= 1, "scores should be between 0 and 1"

        skip_classes = self.cfg.SKIP_CLASSES if decoder_3d else self.cfg.SKIP_CLASSES_2D

        if skip_classes is not None:
            skip_classes = torch.tensor(skip_classes, device=self.device) - 1

            # +1 for background class
            keep_class_mask = torch.ones(num_classes, device=self.device)
            keep_class_mask[skip_classes] = 0
            scores = scores[:, keep_class_mask.bool()]
            num_classes -= len(skip_classes)

        labels = torch.arange(num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // num_classes
        pred_masks = pred_masks[topk_indices]

        results = {}
        if not decoder_3d:
            results["2d"] = self.prepare_2d(
                pred_masks, (output_height, output_width), labels_per_image, scores_per_image,
                batched_inputs, multiview_data, shape, decoder_3d, output_img_size)

        if decoder_3d:
            results["3d"] = self.prepare_3d(
                pred_masks, output_height, output_width,
                labels_per_image, scores_per_image, valids,
                batched_inputs, multiview_data, shape)    
        return results
        
        
    def inference_video_semantic(
            self, mask_cls, mask_pred, image_size=None,
            valids=None, batched_inputs=None, multiview_data=None,
            shape=None):
        """
        pred_cls: 100 X 19
        pred_masks: 100 X 5 X 480 X 640
        """
        mask_pred = self.upsample_pred_masks(
            mask_pred, batched_inputs, multiview_data, shape,
            downsample=self.cfg.HIGH_RES_INPUT, interp="trilinear"
        )
            
        if not self.cfg.MODEL.OPEN_VOCAB or self.cfg.NON_PARAM_SOFTMAX:
            mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        else:
            mask_cls = mask_cls[..., :-1]
            
        assert mask_cls.min() >= 0 and mask_cls.max() <= 1, "mask_cls should be between 0 and 1"
        mask_pred = mask_pred[:, :, :image_size[0], :image_size[1]]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qvhw->cvhw", mask_cls, mask_pred).max(0)[1]
        if valids is not None:
            if self.size_divisibility > 1:
                h, w = image_size[0], image_size[1]
                pad_h = int(np.ceil(h / self.size_divisibility) * self.size_divisibility - h)
                pad_w = int(np.ceil(w / self.size_divisibility) * self.size_divisibility - w)
                valids = F.pad(valids, (0, pad_w, 0, pad_h), mode="constant", value=0)
            H, W = mask_pred.shape[-2:]
            valids = F.interpolate(
                valids.float().unsqueeze(0), size=(H, W), mode='nearest').squeeze(0).bool()
            semseg = semseg[valids]
        return semseg.reshape(-1)

    def inference_scannet_ghost(self, pred_masks, pred_cls, num_classes):
        """
        pred_cls: 100 X 19
        pred_masks: 100 X 5 X 480 X 640
        """
        test_topk_per_image = self.cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 
        if not self.cfg.MODEL.OPEN_VOCAB or self.cfg.NON_PARAM_SOFTMAX:
            scores = F.softmax(pred_cls, dim=-1)[:, :-1]
        else:
            scores = pred_cls[:, :-1]
            
        assert scores.min() >= 0 and scores.max() <= 1, "scores should be between 0 and 1"

        # num_classes = self.sem_seg_head.num_classes
        if num_classes == 20:
            # because we skip floor and wall for evaluation

            # -1 to 0 index it
            skip_classes = torch.tensor(self.cfg.SKIP_CLASSES, device=self.device) - 1

            # +1 for background class
            keep_class_mask = torch.ones(num_classes, device=self.device)
            keep_class_mask[skip_classes] = 0
            scores = scores[:, keep_class_mask.bool()]
            num_classes = 18
            
        labels = torch.arange(num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(test_topk_per_image, sorted=False)

        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // num_classes
        pred_masks = pred_masks[topk_indices]

        masks = pred_masks > 0.
        mask_scores_per_image = (pred_masks.sigmoid().flatten(1) * masks.flatten(1)).sum(1) / (masks.flatten(1).sum(1) + 1e-6)

        result_3d = {
            "pred_classes": labels_per_image + 1,
            "pred_masks": masks.flatten(1).permute(1, 0),
            "pred_scores": scores_per_image * mask_scores_per_image
        }
        return result_3d

    def inference_scannet_ghost_semantic(
            self, mask_cls, mask_pred):
        """
        pred_cls: 100 X 19
        pred_masks: 100 X 5 X 480 X 640
        """
        if not self.cfg.MODEL.OPEN_VOCAB or self.cfg.NON_PARAM_SOFTMAX:
            mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        else:
            mask_cls = mask_cls[..., :-1]
            
        assert mask_cls.min() >= 0 and mask_cls.max() <= 1, "mask_cls should be between 0 and 1"
        
        # mask_cls = mask_cls.sigmoid()[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qn->cn", mask_cls, mask_pred).max(0)[1]
        return semseg.reshape(-1)
    
 
