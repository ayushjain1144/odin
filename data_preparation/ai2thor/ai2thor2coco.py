import os
import argparse
import numpy as np
import json

from PIL import Image
from pycococreatortools import pycococreatortools
from torchvision.transforms import Resize
import pycocotools.mask as mask_util
import pycocotools
from odin.global_vars import AI2THOR_NAME_MAP, AI2THOR_CLASS_ID_MULTIPLIER
from data_preparation.ai2thor.globals_dirs import DATA_DIR, SPLITS

import ipdb
st = ipdb.set_trace


INFO = {
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = [
    {'id': key, 'name': item, 'supercategory': 'nyu40' } for key, item in AI2THOR_NAME_MAP.items() 
]

def read_txt(path):
  """Read txt file into lines.
  """
  with open(path) as f:
    lines = f.readlines()
  lines = [x.strip() for x in lines]
  return lines


def polygons_to_bitmask(polygons, height: int, width: int) -> np.ndarray:
    """
    Args:
        polygons (list[ndarray]): each array has shape (Nx2,)
        height, width (int)

    Returns:
        ndarray: a bool mask of shape (height, width)
    """
    if len(polygons) == 0:
        # COCOAPI does not support empty polygons
        return np.zeros((height, width)).astype(bool)
    rles = mask_util.frPyObjects(polygons, height, width)
    rle = mask_util.merge(rles)
    return mask_util.decode(rle).astype(bool)


def convert_ai2thor_to_coco(path, phase):
    transform = Resize([512,512], Image.NEAREST)
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "depths": [],
        "poses": [],
        "valids": [],
        "segments": [],
        "annotations": []
    }

    # get list
    scene_ids = read_txt(SPLITS[phase])
    image_ids = []
    for scene_id in scene_ids:
        for image_id in os.listdir(os.path.join(path, scene_id, 'color')):
            image_ids.append(os.path.join(scene_id, image_id.split('.')[0]))
    print("images number in {}: {}".format(path, len(image_ids)))

    coco_image_id = 1
    coco_ann_id = 1
    for index in range(len(image_ids)):
        print("{}/{}".format(index, len(image_ids)), end='\r')

        scene_id = image_ids[index].split('/')[0]
        image_id = image_ids[index].split('/')[1]
        ann_path = os.path.join(path, scene_id, 'instance', image_id + '.png')
        ann_map = Image.open(ann_path)
        print("ann_map size: ", ann_map.size)
        ann_map = transform(ann_map)
        image_size = ann_map.size
        ann_map = np.array(ann_map)

        ann_ids = np.unique(ann_map)
        print(ann_path)
        for ann_id in ann_ids:
            label_id = int(ann_id / AI2THOR_CLASS_ID_MULTIPLIER)
            inst_id = int(ann_id % AI2THOR_CLASS_ID_MULTIPLIER)
            if label_id == 0:
                continue

            # is_crowd helps in making pycoco run RLE instead of polygons
            category_info = {'id': label_id, 'is_crowd': 1}
            binary_mask = (ann_map == ann_id).astype(np.uint8)
            mask_size = binary_mask.sum()

            if mask_size == 0:
                continue

            ann_info = pycococreatortools.create_annotation_info(
                coco_ann_id, coco_image_id, category_info, binary_mask,
                image_size, tolerance=0)
            ann_info['iscrowd'] = 0
            
            rle = ann_info['segmentation']
            rle =pycocotools.mask.frPyObjects(rle, rle['size'][0], rle['size'][1])
            rle['counts'] = rle['counts'].decode('ascii')
            ann_info['segmentation'] = rle
            
            ann_info['semantic_instance_id_scannet'] = label_id*AI2THOR_CLASS_ID_MULTIPLIER+inst_id

            if ann_info is not None:
                coco_output['annotations'].append(ann_info)
                coco_ann_id += 1

        image_filename = os.path.join(scene_id, 'color', image_id + '.jpg')
        image_info = pycococreatortools.create_image_info(coco_image_id, image_filename, image_size)
        coco_output['images'].append(image_info)

        depth_filename = os.path.join(scene_id, 'depth', image_id + '.png')
        depth_info = pycococreatortools.create_image_info(coco_image_id, depth_filename, image_size)
        coco_output['depths'].append(depth_info)

        pose_filename = os.path.join(scene_id, 'pose', image_id + '.txt')
        pose_info = pycococreatortools.create_image_info(coco_image_id, pose_filename, image_size)
        coco_output['poses'].append(pose_info)
        coco_image_id += 1

    parent_dir = os.path.dirname(path)
    json.dump(coco_output, open(f'{parent_dir}/ai2thor_{phase}_highres.coco.json','w'))


if __name__ == '__main__':
    phases = ['train', 'val', 'ten_scene', 'two_scene']
    for phase in phases:
        convert_ai2thor_to_coco(DATA_DIR, phase)