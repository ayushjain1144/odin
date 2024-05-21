import copy
import logging
import numpy as np
import pycocotools.mask as mask_util
from imageio import imread
import torch
from operator import itemgetter
from natsort import natsorted
import yaml

from detectron2.data import DatasetCatalog
from detectron2.config import configurable
from detectron2.structures import (
    BitMasks,
    Instances,
    polygons_to_bitmask
)

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from odin.modeling.backproject.backproject import get_scannet_intrinsic, \
    backprojector_dataloader, get_ai2thor_intrinsics, \
    get_s3dis_intrinsics, get_matterport_intrinsics
from PIL import Image
from copy import deepcopy
from torchvision import transforms as pt_transforms
from fvcore.transforms.transform import NoOpTransform
from  odin.global_vars import AI2THOR_CLASS_ID_MULTIPLIER, \
    ALFRED_CLASS_ID_MULTIPLIER, SCANNET200_MISSING_IDS_FROM_VAL_TEST



import matplotlib.pyplot as plt

import ipdb
st = ipdb.set_trace

__all__ = ["ScannetDatasetMapper"]




def custom_image_augmentations(is_train):
    if is_train:
        return pt_transforms.Compose([
            pt_transforms.ToPILImage(),
            pt_transforms.RandomApply([pt_transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        ])
    else:
        return None


def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    if is_train:
        image_size = cfg.INPUT.IMAGE_SIZE
        min_scale = cfg.INPUT.MIN_SCALE
        max_scale = cfg.INPUT.MAX_SCALE

        augmentation = []
        depth_augmentation = []

        augmentation.extend([
            T.RandomApply(T.ResizeScale(
                min_scale=min_scale, max_scale=max_scale, target_height=image_size, target_width=image_size
            )),
            T.FixedSizeCrop(crop_size=(image_size, image_size)),
        ])

    else:
        augmentation = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST_SAMPLING
            ),
        ]
        depth_augmentation = None

    return augmentation, depth_augmentation

class ScannetDatasetMapper:
    """
    A callable which takes a dataset dict in Scannet Dataset format,
    and map it into a format used by the model.
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        dataset_name: str,
        *,
        augmentations,
        depth_augmentations,
        image_format: str,
        use_instance_mask: bool = False,
        frame_left: int = 2,
        frame_right: int = 2,
        num_classes: int = 18,
        size_divisibility,
        decoder_3d,
        inpaint_depth,
        cfg,
        supervise_sparse,
        eval_sparse,
        filter_empty_annotations,
        sample_chunk_aug,
        dataset_dict=None,
        force_decoder_2d=False
    ):
        # fmt: off
        self.is_train               = is_train
        self.tfm_gens               = augmentations
        self.depth_tfm_gems         = depth_augmentations
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.frame_right            = frame_right
        self.frame_left             = frame_left
        self.num_classes            = num_classes
        self.size_divisibility      = size_divisibility
        self.decoder_3d             = decoder_3d
        self.inpaint_depth          = inpaint_depth
        self.cfg = cfg  
        self.supervise_sparse       = supervise_sparse
        self.eval_sparse            = eval_sparse
        self.filter_empty_annotations = filter_empty_annotations
        self.sample_chunk_aug           = sample_chunk_aug
        self.dataset_name  = dataset_name
        self.num_frames = cfg.INPUT.SAMPLING_FRAME_NUM
        
        if not decoder_3d:
            self.frame_left = cfg.INPUT.FRAME_LEFT_2D
            self.frame_right = cfg.INPUT.FRAME_RIGHT_2D
            self.num_frames = cfg.INPUT.SAMPLING_FRAME_NUM_2D
        
        print("frame left: ", self.frame_left)
        print("frame right: ", self.frame_right)
        
        # careful: this assumes that all classes you want to predict
        # are in thing classes (instead of stuff classes)
        self.class_names = {k: v for k, v in enumerate(MetadataCatalog.get(dataset_name).thing_classes)}
        self.num_classes = len(self.class_names)
        
        if self.cfg.MODEL.OPEN_VOCAB and not self.cfg.DETIC:
            # add "__background__" class
            self.class_names[self.num_classes] = "__background__"

        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

        if dataset_dict is not None:
            self.context_dataset_dicts = dataset_dict
        else:
            self.context_dataset_dicts = DatasetCatalog.get(dataset_name)

        self.image_augs = custom_image_augmentations(is_train)
        
        # multiplier
        if 'ai2thor' in dataset_name:
            self.multiplier = AI2THOR_CLASS_ID_MULTIPLIER
            self.get_intrinsics = get_ai2thor_intrinsics
        elif 'alfred' in dataset_name:
            self.multiplier = ALFRED_CLASS_ID_MULTIPLIER
            self.get_intrinsics = get_ai2thor_intrinsics
        elif 's3dis' in dataset_name:
            self.multiplier = 1000
            self.get_intrinsics = get_s3dis_intrinsics
        elif 'matterport' in dataset_name:
            self.multiplier = 1000
            self.get_intrinsics = get_matterport_intrinsics
        else:
            self.multiplier = 1000
            self.get_intrinsics = get_scannet_intrinsic

        # path to SEMSEG_100k
        if 'scannet200' in dataset_name:
            label_db_filepath = self.cfg.SCANNET200_DATA_DIR
        elif 'scannet' in dataset_name:
            label_db_filepath = self.cfg.SCANNET_DATA_DIR
        elif 's3dis' in dataset_name:
            label_db_filepath = self.cfg.S3DIS_DATA_DIR
        elif 'matterport' in dataset_name:
            label_db_filepath = self.cfg.MATTERPORT_DATA_DIR

        if (self.cfg.USE_GHOST_POINTS) and 'ai2thor' not in dataset_name:
            # label_db_filepath = f"{data_dir}/train_validation_database.yaml"
            with open(label_db_filepath) as f:
                data = yaml.load(f)

            self.scannet_data = {}
            for item in data:
                if 's3dis' in dataset_name:
                    area, room = item['raw_filepath'].split('/')[-2:]
                    scene_name = f'{area}_{room}'
                    scene_name = scene_name.lower()
                elif 'matterport' in dataset_name:
                    scene_name, region_name = item['filepath'].split('/')[-1].split('.')[0].split('_')
                    scene_name = f"{scene_name}_region{region_name}"
                else:
                    scene_name = item['raw_filepath'].split('/')[-2]
                self.scannet_data[scene_name] = item
            

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs, depth_augs = [], []
        if cfg.INPUT.STRONG_AUGS:
            augs, depth_augs = build_transform_gen(cfg, is_train)

        frame_right = cfg.INPUT.FRAME_RIGHT
        frame_left = cfg.INPUT.FRAME_LEFT

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "depth_augmentations": depth_augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "frame_right": frame_right,
            "frame_left": frame_left,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
            "decoder_3d": cfg.MODEL.DECODER_3D,
            "inpaint_depth": cfg.INPUT.INPAINT_DEPTH,
            "cfg": cfg,
            "supervise_sparse": cfg.MODEL.SUPERVISE_SPARSE,
            "eval_sparse": cfg.TEST.EVAL_SPARSE,
            "filter_empty_annotations": cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            "sample_chunk_aug": cfg.INPUT.SAMPLE_CHUNK_AUG
        }

        return ret


    def get_multiview_xyz(self, dataset_dict):
        scales = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}
        v = len(dataset_dict["images"])
        h, w = dataset_dict["new_image_shape"]
        pad_h = int(np.ceil(h / self.size_divisibility) * self.size_divisibility - h)
        pad_w = int(np.ceil(w / self.size_divisibility) * self.size_divisibility - w)
        H_padded = h + pad_h
        W_padded = w + pad_w
        features = {
            k: torch.zeros(v, 1, H_padded//s, W_padded//s) for k, s in scales.items()
        }
        depths = torch.stack(dataset_dict["depths"])
        poses = torch.stack(dataset_dict["poses"])
        
        # poses = poses
        intrinsics = torch.stack(dataset_dict["intrinsics"])
        augment = self.is_train and self.cfg.INPUT.AUGMENT_3D
        if 'scannet_coords' in dataset_dict:
            scannet_pc = dataset_dict['scannet_coords']
        else:
            scannet_pc = None

        multi_scale_xyz, scannet_pc, original_xyz = backprojector_dataloader(
            list(features.values()), depths, poses,
            intrinsics, augment=augment, 
            method=self.cfg.MODEL.INTERPOLATION_METHOD, 
            scannet_pc=scannet_pc, padding=(pad_h, pad_w), 
            do_rot_scale=self.cfg.DO_ROT_SCALE)
        multi_scale_xyz = multi_scale_xyz[::-1]
        dataset_dict['multi_scale_xyz'] = multi_scale_xyz
        if scannet_pc is not None:
            dataset_dict['scannet_coords'] = scannet_pc
        dataset_dict['original_xyz'] = original_xyz[0]
        
        return dataset_dict

    
    def get_current_image_id(self, dataset_dict, context):
        idx = None
        for i, context_img in enumerate(context):
            if dataset_dict['file_name'] == context_img['file_name']:
                return i
        return idx


    def convert_to_video_dict(self, dataset_dict, context):
        record = {}
        record["height"] = dataset_dict["height"]
        record["width"] = dataset_dict["width"]
        record["file_names"] = []
        record["depth_file_names"] = []
        record["pose_file_names"] = []
        record["valid_file_names"] = []
        record["segment_file_names"] = []
        record["annotations"] = []
        record["image_ids"] = []
        
        context = natsorted(context, key=itemgetter(*['file_name']))

        sample_factor = 1
        if self.sample_chunk_aug and self.is_train:
            sample_factor = np.random.randint(1, self.cfg.INPUT.CHUNK_AUG_MAX)

        if self.frame_left > 0:
            idx = self.get_current_image_id(dataset_dict, context)
            if self.cfg.INPUT.UNIFORM_SAMPLE:
                # sample self.frame_left images uniformly from left side 
                sample_factor = len(context) // (self.frame_left * 2)

            left_ids = [(idx - sample_factor*(i+1)) % len(context) for i in range(self.frame_left)]

            if len(left_ids) == 1:
                left_contexts = [itemgetter(*left_ids)(context)]
            else:
                left_contexts = list(itemgetter(*left_ids)(context))
                left_contexts.reverse()

            for lc in left_contexts:
                record["file_names"].append(lc['file_name'])
                record["valid_file_names"].append(lc['valid_file'])
                record["segment_file_names"].append(lc['segment_file'])
                record["depth_file_names"].append(lc['depth_file'])
                record["pose_file_names"].append(lc['pose_file'])
                record["annotations"].append(lc['annotations'])
                record["image_ids"].append(lc["image_id"])

        record["file_names"].append(dataset_dict['file_name'])
        record["valid_file_names"].append(dataset_dict['valid_file'])
        record["segment_file_names"].append(dataset_dict['segment_file'])
        record["depth_file_names"].append(dataset_dict['depth_file'])
        record["pose_file_names"].append(dataset_dict['pose_file'])
        record["annotations"].append(dataset_dict['annotations'])   
        record["image_ids"].append(dataset_dict["image_id"])

        if self.frame_right > 0:
            right_ids = [(idx + sample_factor*(i+1)) % len(context) for i in range(self.frame_right)]
            if len(right_ids) == 1:
                right_contexts = [itemgetter(*right_ids)(context)]
            else:
                right_contexts = list(itemgetter(*right_ids)(context))

            for rc in right_contexts:
                record["file_names"].append(rc['file_name'])
                record["valid_file_names"].append(rc['valid_file'])
                record["segment_file_names"].append(rc['segment_file'])
                record["depth_file_names"].append(rc['depth_file'])
                record["pose_file_names"].append(rc['pose_file'])
                record["annotations"].append(rc['annotations'])
                record["image_ids"].append(rc["image_id"])

        record["length"] = len(record["file_names"])
        record["image_id"] = dataset_dict['image_id']

        if self.filter_empty_annotations:
            video_instance_ids = []
            for anno in record['annotations']:
                anno_instance_ids = [ann['semantic_instance_id_scannet'] for ann in anno]
                video_instance_ids.extend(anno_instance_ids)
            if len(video_instance_ids) == 0:
                return None

        return record


    def subsample_frames(self, dataset_dict, **kwargs):
        num_sample_frames = self.num_frames
        if not self.is_train and self.cfg.MAX_FRAME_NUM != -1:
            num_sample_frames = self.cfg.MAX_FRAME_NUM
        
        if len(dataset_dict['file_names']) <= num_sample_frames and not self.is_train:
            return dataset_dict

        keys = ["file_names", "depth_file_names", "pose_file_names", "annotations", "segment_file_names", "valid_file_names", "image_ids"]

        # random sample frames with a seed
        if not self.is_train:
            np.random.seed(42)

        if self.cfg.SAMPLING_STRATEGY == "random":
            sample_ids = np.random.choice(
                len(dataset_dict['file_names']), num_sample_frames, replace=True)
        elif self.cfg.SAMPLING_STRATEGY == "consecutive":
            if not self.is_train:
                # print(f"Keeping {num_sample_frames} frames from {len(dataset_dict['file_names'])}")
                # uniform sampling
                sample_factor = len(dataset_dict['file_names']) // num_sample_frames
                start_point = 0
            else:
                if self.sample_chunk_aug:
                    sample_factor = np.random.randint(1, self.cfg.INPUT.CHUNK_AUG_MAX)
                else:
                    sample_factor = 1
                # sample_factor = np.random.randint(1, self.cfg.INPUT.CHUNK_AUG_MAX)
                start_point = np.random.randint(0, len(dataset_dict['file_names']))
            sample_ids = [(start_point + i*sample_factor) % len(dataset_dict['file_names']) for i in range(num_sample_frames)]
        
        for key in keys:
            dataset_dict[key] = [dataset_dict[key][i] for i in sample_ids]

        dataset_dict["length"] = len(dataset_dict["file_names"])
        return dataset_dict


    def __call__(self, dataset_dict, **kwargs):
        """
        Args:
            dataset_dict (dict): Metadata of one video
        
        Returns:
            dict: a format that builtin models in ...
        """
        dataset_dict = copy.deepcopy(dataset_dict)

        if not 'single' in self.dataset_name:
            if self.num_frames > 1:
                context = [self.context_dataset_dicts[context_id] for context_id in dataset_dict['context_ids']]
            else:
                context = [dataset_dict]
            dataset_dict = self.convert_to_video_dict(dataset_dict, context)
        
        # subsample frames
        if 'single' in self.dataset_name and (self.is_train or self.cfg.MAX_FRAME_NUM != -1):
            dataset_dict = self.subsample_frames(dataset_dict, **kwargs)

        if dataset_dict is None and self.filter_empty_annotations:
            # print("found empty video annotation") # for debug
            return self.__call__(self.context_dataset_dicts[0])

        video_length = dataset_dict["length"]
        selected_idx = range(video_length)
        eval_idx = video_length // 2

        video_annos = dataset_dict.pop("annotations", None)
        file_names = dataset_dict.pop("file_names", None)
        image_ids = dataset_dict.pop("image_ids", None)
        decoder_3d = self.decoder_3d
        
        if decoder_3d:
            depth_file_names = dataset_dict.pop("depth_file_names", None)
            if self.inpaint_depth:
                depth_file_names = [
                    depth_file_names[i].replace("depth", self.cfg.DEPTH_PREFIX) for i in range(len(depth_file_names))
                ]
            pose_file_names = dataset_dict.pop("pose_file_names", None)
            
            if 's3dis' in self.dataset_name or 'matterport' in self.dataset_name:
                instrinsic_file_names = [
                    pose_file_names[i].replace("pose", "intrinsic") for i in range(len(pose_file_names))
                ]
            else:
                instrinsic_file_names = None
        dataset_dict["decoder_3d"] = decoder_3d
        dataset_dict["images"] = []
        dataset_dict["padding_masks"] = []
        dataset_dict["image"] = None
        dataset_dict["instances"] = None
        dataset_dict["instances_all"] = []
        dataset_dict["file_names"] = []
        dataset_dict["image_ids"] = []
        dataset_dict["file_name"] = None
        dataset_dict["valid_class_ids"] = np.arange(len(self.class_names))

        if self.supervise_sparse or self.eval_sparse:
            dataset_dict["valids"] = []
            if 'scannet' in self.dataset_name:
                valid_file_names = dataset_dict.pop("valid_file_names", None)
            
        if self.cfg.USE_SEGMENTS and not self.cfg.USE_GHOST_POINTS and decoder_3d:
            segment_file_names = dataset_dict.pop("segment_file_names")
            dataset_dict["segments"] = []

        dataset_dict['multiplier'] = self.multiplier

        # some parameters for augmentations
        dataset_dict["do_camera_drop"] = self.cfg.INPUT.CAMERA_DROP and self.is_train
        dataset_dict["max_frames"] = video_length
        dataset_dict["use_ghost"] = self.cfg.USE_GHOST_POINTS

        if decoder_3d:
            dataset_dict["depths"] = []
            dataset_dict["poses"] = []
            dataset_dict["intrinsics"] = []

            dataset_dict["depth_file_names"] = []
            dataset_dict["pose_file_names"] = []
        
        for frame_idx in selected_idx:
            if eval_idx == frame_idx:
                dataset_dict["file_name"] = file_names[frame_idx]
            dataset_dict["file_names"].append(file_names[frame_idx])
            dataset_dict["image_ids"].append(image_ids[frame_idx])

            if decoder_3d:
                dataset_dict["depth_file_names"].append(depth_file_names[frame_idx])
                dataset_dict["pose_file_names"].append(pose_file_names[frame_idx])

            image = utils.read_image(file_names[frame_idx], format=self.image_format)
            if self.is_train and self.cfg.INPUT.COLOR_AUG:
                image = np.asarray(self.image_augs(image))
            size = image.shape
            utils.check_image_size(dataset_dict, image)
            padding_mask = np.ones(image.shape[:2])

            aug_input = T.AugInput(image)
            aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
            image = aug_input.image


            # get padding mask for segmentation
            padding_mask = transforms.apply_segmentation(padding_mask)
            padding_mask = ~ padding_mask.astype(bool)

            if decoder_3d:
                depth = imread(depth_file_names[frame_idx]).astype(np.float32)
                depth = depth / 1000.0
                if self.cfg.IGNORE_DEPTH_MAX != -1:
                    depth[depth > self.cfg.IGNORE_DEPTH_MAX] = 0.0
                if self.cfg.INPUT.STRONG_AUGS:
                    depth_transforms = deepcopy(transforms)
                    if self.is_train:
                        depth_transforms[-1].pad_value = 0.0
                        depth_transforms[0].interp = Image.NEAREST
                        depth = T.apply_transform_gens(depth_transforms, T.AugInput(depth))[0].image

                        # get camera intrinsics
                        if isinstance(depth_transforms[0], NoOpTransform):
                            intrinsics_ = self.get_intrinsics(size, intrinsic_file=instrinsic_file_names[frame_idx] if instrinsic_file_names is not None else None)
                        else:
                            intrinsics_ = self.get_intrinsics((depth_transforms[0].new_h, depth_transforms[0].new_w), intrinsic_file=instrinsic_file_names[frame_idx] if instrinsic_file_names is not None else None)

                        # for crop transform
                        intrinsics_[0, 2] -= depth_transforms[1].x0
                        intrinsics_[1, 2] -= depth_transforms[1].y0
                    else:
                        depth_transforms[0].interp = Image.NEAREST
                        depth = T.apply_transform_gens(depth_transforms, T.AugInput(depth))[0].image
                        intrinsics_ = self.get_intrinsics((depth_transforms[0].new_h, depth_transforms[0].new_w), intrinsic_file=instrinsic_file_names[frame_idx] if instrinsic_file_names is not None else None)

                else:
                    intrinsics_ = self.get_intrinsics(depth.shape, intrinsic_file=instrinsic_file_names[frame_idx] if instrinsic_file_names is not None else None)
                    
                pose = np.loadtxt(pose_file_names[frame_idx])

            # NOTE copy() is to prevent annotations getting changed from applying augmentations
            if not (self.cfg.USE_GHOST_POINTS and decoder_3d) or 'ai2thor' in self.dataset_name:
                _frame_annos = []
                for anno in video_annos[frame_idx]:
                    _anno = {}
                    for k, v in anno.items():
                        _anno[k] = copy.deepcopy(v)
                    _frame_annos.append(_anno)

                # USER: Implement additional transformations if you have other types of data
                annos = [
                    utils.transform_instance_annotations(obj, transforms, image.shape[:2])
                    for obj in _frame_annos
                    if obj.get("iscrowd", 0) == 0
                ]

                if len(annos):
                    assert "segmentation" in annos[0]
                segms = [obj["segmentation"] for obj in annos]
                masks = []
                for segm in segms:
                    if isinstance(segm, list):
                        # polygon
                        masks.append(polygons_to_bitmask(segm, *image.shape[:2]))
                    elif isinstance(segm, dict):
                        # COCO RLE
                        masks.append(mask_util.decode(segm))
                    elif isinstance(segm, np.ndarray):
                        assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
                            segm.ndim
                        )
                        if self.is_train:
                            segm[segm > 1] = 0
                        # mask array
                        masks.append(segm)
                    else:
                        raise ValueError(
                            "Cannot convert segmentation of type '{}' to BitMasks!"
                            "Supported types are: polygons as list[list[float] or ndarray],"
                            " COCO-style RLE as a dict, or a binary segmentation mask "
                            " in a 2D numpy array of shape HxW.".format(type(segm))
                        )

                masks = [torch.from_numpy(np.ascontiguousarray(x)) for x in masks]
                classes = [int(obj["category_id"]) for obj in annos]
                classes = torch.tensor(classes, dtype=torch.int64)
                instance_ids = [int(obj["semantic_instance_id_scannet"]) for obj in annos]
                instance_ids = torch.tensor(instance_ids, dtype=torch.int64)

            image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
            if decoder_3d:
                depth = torch.as_tensor(np.ascontiguousarray(depth))
                pose = torch.from_numpy(pose).float()
                intrinsics_ = torch.from_numpy(intrinsics_).float()
            if self.supervise_sparse or self.eval_sparse:
                if decoder_3d:
                    if 'scannet' in self.dataset_name:
                        valid = Image.open(valid_file_names[frame_idx])
                        valid = np.asarray(valid, dtype=np.float32)
                        valid = transforms.apply_segmentation(valid)
                        valid = torch.from_numpy(valid).bool()
                    else:
                        valid = depth > 0.0
                else:
                    valid = torch.ones(image.shape[-2:], dtype=bool)
                # valid = torch.from_numpy(valid).bool()
                
            if decoder_3d and self.cfg.USE_SEGMENTS and not self.cfg.USE_GHOST_POINTS:
                segment = Image.open(segment_file_names[frame_idx])
                segment = np.asarray(segment, dtype=np.float32)
                segment = transforms.apply_segmentation(segment)
                dataset_dict["segments"].append(
                    torch.from_numpy(segment).to(torch.int64)
                )

            image_shape = (image.shape[-2], image.shape[-1])
            dataset_dict["new_image_shape"] = image_shape
            if decoder_3d:
                depth_shape = (depth.shape[-2], depth.shape[-1])
                assert depth_shape == image_shape
        
            dataset_dict["images"].append(image)
            dataset_dict["padding_masks"] = torch.as_tensor(np.ascontiguousarray(padding_mask))
            if decoder_3d:
                dataset_dict["depths"].append(depth)
                dataset_dict["poses"].append(pose)

                dataset_dict["intrinsics"].append(intrinsics_)

            if self.supervise_sparse or self.eval_sparse:
                dataset_dict["valids"].append(valid)

            if frame_idx == eval_idx:
                dataset_dict["image"] = image

            dataset_dict['original_all_classes'] = copy.copy(self.class_names)
            all_classes = copy.copy(self.class_names)
            
            if self.cfg.MODEL.OPEN_VOCAB and 'scannet200' in self.dataset_name:
                for id in SCANNET200_MISSING_IDS_FROM_VAL_TEST:
                    del all_classes[id - 1]
                    
            if not (self.cfg.USE_GHOST_POINTS and decoder_3d) or 'ai2thor' in self.dataset_name:
                instances = Instances(image_shape)
                instances.gt_classes = classes
                
                instances.instance_ids = instance_ids
                if len(masks) == 0:
                    instances.gt_masks = torch.zeros((0, image.shape[-2], image.shape[-1]))
                else:
                    masks = BitMasks(torch.stack(masks))
                    instances.gt_masks = masks.tensor
                dataset_dict["instances_all"].append(instances)
                if frame_idx == eval_idx:
                    dataset_dict["instances"] = instances
                    
            dataset_dict["all_classes"] = all_classes
                
        if self.cfg.USE_GHOST_POINTS and decoder_3d and 'ai2thor' not in self.dataset_name:
            assert 'scannet' in self.dataset_name or 's3dis' in self.dataset_name or 'matterport' in self.dataset_name
            scene_name = dataset_dict['file_name'].split('/')[-3]
            
            if 's3dis' in self.dataset_name:
                scene_name = scene_name.lower()
            filepath = self.scannet_data[scene_name]['filepath']
            
            if 's3dis' in self.dataset_name:
                sub1, sub2 = filepath.split('s3dis')
                filepath = f"{sub1}/SEMSEG_100k/s3dis/{sub2}"
            points = np.load(filepath)
            coordinates, color, _, segments, labels = (
                points[:, :3],
                points[:, 3:6],
                points[:, 6:9],
                points[:, 9],
                points[:, 10:12],
            )
            
            if 's3dis' in self.dataset_name:
                # s3dis from mask3d follow different convention
                labels[:, 0] += 1
                
                # their segments is all 1 and will lead to bugs if USE_SEGMENTS=True
                segments = np.arange(len(labels))
                
            dataset_dict["scannet_coords"] = torch.from_numpy(coordinates)
            dataset_dict["scannet_color"] = torch.from_numpy(color)
            dataset_dict["scannet_labels"] = torch.from_numpy(labels)
            dataset_dict["scannet_segments"] = torch.from_numpy(segments)

        dataset_dict["dataset_name"] = self.dataset_name
        dataset_dict['num_classes'] = self.num_classes
        
        # # backproject and multiview
        if decoder_3d:
            dataset_dict = self.get_multiview_xyz(dataset_dict)
            
        return dataset_dict