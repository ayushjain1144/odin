import logging
import numpy as np
from natsort import natsorted
import operator
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json, register_coco_instances
from odin.global_vars import NAME_MAP20, AI2THOR_NAME_MAP, S3DIS_NAME_MAP, \
    SCANNET200_NAME_MAP, ALFRED_NAME_MAP, MATTERPORT_NAME_MAP

logger = logging.getLogger(__name__)

__all__ = ["load_scannet_json", "register_scannet_instances"]


NAME_MAP = { 1: 'cabinet', 2: 'bed', 3: 'chair', 4: 'sofa', 5: 'table', 6: 'door', 7: 'window',
             8: 'bookshelf', 9: 'picture', 10: 'counter', 11: 'desk', 12: 'curtain', 13: 'refridgerator',
             14: 'shower curtain', 15: 'toilet', 16: 'sink', 17:' bathtub',  18: 'otherfurniture'}

SCANNET_CATEGORIES = [
    {'id': key, 'name': item, 'supercategory': 'nyu40' } for key, item in NAME_MAP.items() 
]

SCANNET_CATEGORIES_20 = [
    {'id': key, 'name': item, 'supercategory': 'nyu40' } for key, item in NAME_MAP20.items() 
]

import ipdb
st = ipdb.set_trace


def _get_dataset_instances_meta(dataset='ai2thor'):
    if dataset == 'ai2thor':
        name_map = AI2THOR_NAME_MAP
    elif dataset == 'alfred':
        name_map = ALFRED_NAME_MAP
    elif dataset == 'replica':
        name_map = NAME_MAP20
    elif dataset == "s3dis":
        name_map = S3DIS_NAME_MAP
    elif dataset == 'scannet200':
        name_map = SCANNET200_NAME_MAP
    elif dataset == 'matterport':
        name_map = MATTERPORT_NAME_MAP
    elif dataset == 'scannet':
        name_map = NAME_MAP20
    else:
        assert False, 'dataset not supported: {}'.format(dataset)
    
    dataset_categories = [
        {'id': key, 'name': item, 'supercategory': 'nyu40' } for key, item in name_map.items()
    ]
    
    thing_ids = [k["id"] for k in dataset_categories]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in dataset_categories]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }
    return ret


def _get_scannet_instances_meta():
    # DELETE
    thing_ids = [k["id"] for k in SCANNET_CATEGORIES]
    assert len(thing_ids) == 18, len(thing_ids)
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in SCANNET_CATEGORIES]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }
    return ret


def _get_scannet_instances20_meta():
    # DELETE
    thing_ids = [k["id"] for k in SCANNET_CATEGORIES_20]
    assert len(thing_ids) == 20, len(thing_ids)
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in SCANNET_CATEGORIES_20]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }
    return ret


def get_scenes(image_dataset):
    '''
    takes in a video dataset and retuns dict of scene to images mapping
    returns : dict{scene_id: str, image_dataset_ids: []}
    '''
    scenes = {}
    for idx, img in enumerate(image_dataset):
        scene_id = img['file_name'].split('/')[-3]
        if scene_id in scenes:
            scenes[scene_id].append(idx)
        else:
            scenes[scene_id] = [idx]
    return scenes

def make_video_from_frames(imgs):
    # sort images by filename
    imgs = natsorted(imgs, key=operator.itemgetter(*['file_name']))
    file_names = []
    annotations = []
    for img in imgs:
        file_names.append(img['file_name'])
        annotations.append(img['annotations'])
    return file_names, annotations

def get_context_records(idx, img, scenes):
    scene_id = img['file_name'].split('/')[-3]
    dataset_ids = scenes[scene_id]
    return dataset_ids
    
def aggregate_images_by_sceneid(image_dataset):
    '''
    Takes in all scannet images, and add its multiview images as contexts.
    '''
    scenes = get_scenes(image_dataset)
    
    # add depth poses for reprojection
    for idx, img in enumerate(image_dataset):
        # add depth, pose, intrinsics
        img['depth_file'] = img['file_name'].replace('color', 'depth').replace('jpg', 'png')
        img['segment_file'] = img['file_name'].replace('color', 'segments').replace('jpg', 'png')
        img['valid_file'] = img['file_name'].replace('color', 'valids').replace('jpg', 'png')
        img['pose_file'] = f"{img['file_name'].split('color')[0]}pose/{img['file_name'].split('/')[-1].replace('jpg', 'txt').replace('png', 'txt')}"
        img['context_ids'] = get_context_records(idx, img, scenes)
    

    return image_dataset

def load_scannet_json(json_file, image_root, dataset_name=None):
    image_dataset = load_coco_json(
        json_file, image_root, dataset_name, extra_annotation_keys=['semantic_instance_id_scannet'])
    video_dataset = aggregate_images_by_sceneid(image_dataset)
    return video_dataset

def register_scannet_context_instances(name, metadata, json_file, image_root):
    DatasetCatalog.register(name, lambda: load_scannet_json(json_file, image_root, name))

    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="scannet", **metadata
    )


def load_scannet_json_single(json_file, image_root, dataset_name=None):
    if os.path.isdir(json_file):
        assert not os.path.isdir(image_root)
        "for some unknown reason, json_file is a directory, but image_root is not."
        tmp = json_file
        json_file = image_root
        image_root = tmp
    image_dataset = load_coco_json(
        json_file, image_root, dataset_name, extra_annotation_keys=['semantic_instance_id_scannet'])
    scenes = get_scenes(image_dataset)

    image_dataset = aggregate_images_by_sceneid(image_dataset)

    scene_dataset = []
    for scene_id, imgs in scenes.items():
        record = {}
        record["height"] = image_dataset[0]['height']
        record["width"] = image_dataset[0]["width"]
        record["file_names"] = []
        record["depth_file_names"] = []
        record["pose_file_names"] = []
        record["annotations"] = []
        record["segment_file_names"] = []
        record["valid_file_names"] = []
        record["image_ids"] = []

        for id in imgs:
            record["file_names"].append(image_dataset[id]['file_name'])
            record["depth_file_names"].append(image_dataset[id]['depth_file'])
            record["pose_file_names"].append(image_dataset[id]['pose_file'])
            record["annotations"].append(image_dataset[id]['annotations'])
            record["segment_file_names"].append(image_dataset[id]['segment_file'])
            record["valid_file_names"].append(image_dataset[id]['valid_file'])
            record["image_ids"].append(image_dataset[id]['image_id'])

        sorted_indices = natsorted(range(len(record["file_names"])), key=lambda i: record["file_names"][i])
        record["file_names"] = [record["file_names"][i] for i in sorted_indices]
        record["depth_file_names"] = [record["depth_file_names"][i] for i in sorted_indices]
        record["pose_file_names"] = [record["pose_file_names"][i] for i in sorted_indices]
        record["annotations"] = [record["annotations"][i] for i in sorted_indices]
        record["segment_file_names"] = [record["segment_file_names"][i] for i in sorted_indices]
        record["valid_file_names"] = [record["valid_file_names"][i] for i in sorted_indices]
        record["image_ids"] = [record["image_ids"][i] for i in sorted_indices]

        record["length"] = len(imgs)
        record["image_id"] = scene_id
        scene_dataset.append(record)
    return scene_dataset

def register_scannet_context_instances_single(name, metadata, json_file, image_root):
    DatasetCatalog.register(name, lambda: load_scannet_json_single(json_file, image_root, name))

    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="scannet", **metadata
    )


if __name__ == "__main__":
    """
    Test Scannet json dataloader
    """
    from detectron2.utils.logger import setup_logger
    from detectron2.utils.visualizer import Visualizer

    logger = setup_logger(name=__name__)
    meta = MetadataCatalog.get("scannet_train")

    json_file = "/projects/katefgroup/language_grounding/SEMSEG_100k/scannet_val_valid.coco.json"
    image_root = "/projects/katefgroup/language_grounding/SEMSEG_100k/frames_square"
    dicts = load_scannet_json(json_file, image_root, dataset_name="scannet_train")
    logger.info("Done loading {} samples.".format(len(dicts)))