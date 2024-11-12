# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/d2/detr/dataset_mapper.py
import copy
import logging

import numpy as np
import torch
import random

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.catalog import MetadataCatalog

from pycocotools import mask as coco_mask

import ipdb
st = ipdb.set_trace

__all__ = ["COCOInstanceNewBaselineDatasetMapper"]


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks



def build_transform_gen_3D(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
        
    This is a special transform function which makes the 2D scale the same as 3D scale
    so 50% of the time we train on 3D scale and use this function
    50% of the time we train on 2D scale and use the build_transform_gen function
    """
    # assert is_train, "Only support training augmentation"
    image_size = cfg.INPUT.IMAGE_SIZE
    min_scale = cfg.INPUT.MIN_SCALE
    max_scale = cfg.INPUT.MAX_SCALE
    # print(image_size, min_scale, max_scale)

    augmentation = []

    if is_train:
        if cfg.INPUT.RANDOM_FLIP != "none":
            augmentation.append(
                T.RandomFlip(
                    horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                    vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
                )
            )

        augmentation.extend([
            T.ResizeScale(
                min_scale=min_scale, max_scale=max_scale, target_height=image_size, target_width=image_size
            ),
            T.FixedSizeCrop(crop_size=(image_size, image_size)),
        ])
    else:
        pass

    return augmentation


def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    image_size = cfg.INPUT.IMAGE_SIZE_2D
    min_scale = cfg.INPUT.MIN_SCALE
    max_scale = cfg.INPUT.MAX_SCALE

    augmentation = []

    if is_train:
        if cfg.INPUT.RANDOM_FLIP != "none":
            augmentation.append(
                T.RandomFlip(
                    horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                    vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
                )
            )

        augmentation.extend([
            T.ResizeScale(
                min_scale=min_scale, max_scale=max_scale, target_height=image_size, target_width=image_size
            ),
            T.FixedSizeCrop(crop_size=(image_size, image_size)),
        ])
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST_2D
        max_size = cfg.INPUT.MAX_SIZE_TEST_2D
        sample_style = "choice"
        augmentation = [T.ResizeShortestEdge(min_size, max_size, sample_style)]

    return augmentation


# This is specifically designed for the COCO dataset.
class COCOInstanceNewBaselineDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer.

    This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        tfm_gens,
        tfm_gens_3D,
        cfg,
        image_format,
        dataset_name,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.tfm_gens = tfm_gens
        self.tfm_gens_3D = tfm_gens_3D
        self.cfg = cfg
        self.dataset_name = dataset_name
        logging.getLogger(__name__).info(
            "[COCOInstanceNewBaselineDatasetMapper] Full TransformGens used in training: {}".format(str(self.tfm_gens))
        )
        self.class_names = {k: v for k, v in enumerate(MetadataCatalog.get(dataset_name).thing_classes)}
        self.num_classes = len(self.class_names)
        if self.cfg.MODEL.OPEN_VOCAB:
            # add "__background__" class
            self.class_names[self.num_classes] = "__background__"
        
        self.img_format = image_format
        self.is_train = is_train
    
    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)
        tfm_gens_3D = build_transform_gen_3D(cfg, is_train)

        ret = {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "tfm_gens_3D": tfm_gens_3D,
            "image_format": cfg.INPUT.FORMAT,
            "cfg": cfg,
            
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        # TODO: get padding mask
        # by feeding a "segmentation mask" to the same transforms
        padding_mask = np.ones(image.shape[:2])

        if self.cfg.AUGMENT_WITH_3D_SCALE and self.is_train:
            if random.random() > 0.5:
                tfm_gens = self.tfm_gens_3D
            else:
                tfm_gens = self.tfm_gens
            image, transforms = T.apply_transform_gens(tfm_gens, image)
        else:
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        # the crop transformation has default padding value 0 for segmentation
        padding_mask = transforms.apply_segmentation(padding_mask)
        padding_mask = ~ padding_mask.astype(bool)

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))

        # if not self.is_train:
        #     # USER: Modify this if you want to keep them for some reason.
        #     dataset_dict.pop("annotations", None)
        #     return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                # Let's always keep mask
                # if not self.mask_on:
                #     anno.pop("segmentation", None)
                anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            try:
                # NOTE: does not support BitMask due to augmentation
                # Current BitMask cannot handle empty objects
                instances = utils.annotations_to_instances(annos, image_shape)
                # After transforms such as cropping are applied, the bounding box may no longer
                # tightly bound the object. As an example, imagine a triangle object
                # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
                # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
                # the intersection of original bounding box and the cropping box.
                # instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
                # Need to filter empty instances first (due to augmentation)
                instances = utils.filter_empty_instances(instances)
            except:
                print("No instances found in {}".format(dataset_dict["file_name"]))
            # Generate masks from polygon
            h, w = instances.image_size
            # image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float)
            if hasattr(instances, 'gt_masks'):
                gt_masks = instances.gt_masks
                gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                instances.gt_masks = gt_masks
            
            instance_ids = [(obj_id + 1) * 1000 + i for i, obj_id in enumerate(instances.gt_classes)]
            instances.instance_ids = torch.tensor(instance_ids, dtype=torch.int64)
            dataset_dict["instances"] = instances
            dataset_dict['instances_all'] = [instances]
            dataset_dict["all_classes"] = self.class_names
            
        dataset_dict['original_all_classes'] = copy.copy(self.class_names)
        dataset_dict['valid_class_ids'] = np.arange(len(self.class_names))
        dataset_dict['decoder_3d'] = False
        dataset_dict['file_names'] = [dataset_dict['file_name']]
        dataset_dict['valids'] = None
        dataset_dict["do_camera_drop"] = False
        dataset_dict["max_frames"] = 1
        dataset_dict["use_ghost"] = False
        dataset_dict['images'] = [dataset_dict['image']]
        dataset_dict['multiplier'] = 1000
        dataset_dict['dataset_name'] = self.dataset_name
        dataset_dict['num_classes'] = self.num_classes

        return dataset_dict
