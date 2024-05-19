import os
import argparse
import numpy as np
import json
from tqdm import tqdm

from pycococreatortools import pycococreatortools
import pycocotools.mask as mask_util

from odin.global_vars import NAME_MAP, SCANNET200_NAME_MAP
from data_preparation.scannet.globals_dirs import DATA_DIR, SPLITS

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
    {'id': key, 'name': item, 'supercategory': 'nyu40' } for key, item in NAME_MAP.items() 
]

CATEGORIES_200 = [
    {'id': key, 'name': item, 'supercategory': 'nyu40' } for key, item in SCANNET200_NAME_MAP.items()
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


def convert_scannet_to_coco(path, phase, scannet200=False):
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES if not scannet200 else CATEGORIES_200,
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
    for scene_id in tqdm(scene_ids, desc="Processing scenes"):
        for image_id in os.listdir(os.path.join(path, scene_id, 'color')):
            image_ids.append(os.path.join(scene_id, image_id.split('.')[0]))
    print("images number in {}: {}".format(path, len(image_ids)))

    coco_image_id = 1
    for index in range(len(image_ids)):
        print("{}/{}".format(index, len(image_ids)), end='\r')

        scene_id = image_ids[index].split('/')[0]
        image_id = image_ids[index].split('/')[1]
        image_size = (640, 480)

        pose_filename = os.path.join(scene_id, 'pose', image_id + '.txt')

        # load pose and check if it is valid (no inf)
        pose = np.loadtxt(f'{path}/{pose_filename}')
        if np.any(np.isinf(pose)):
            print("invalid pose: {}".format(pose_filename))
            continue
        pose_info = pycococreatortools.create_image_info(coco_image_id, pose_filename, image_size)
        coco_output['poses'].append(pose_info)

        ext = 'png' if os.path.exists(os.path.join(path, scene_id, 'color', image_id + '.png')) else 'jpg'
        image_filename = os.path.join(scene_id, 'color', image_id + f'.{ext}')
        image_info = pycococreatortools.create_image_info(coco_image_id, image_filename, image_size)
        coco_output['images'].append(image_info)

        depth_filename = os.path.join(scene_id, 'depth', image_id + '.png')
        depth_info = pycococreatortools.create_image_info(coco_image_id, depth_filename, image_size)
        coco_output['depths'].append(depth_info)
        
        coco_image_id += 1

    parent_dir = os.path.dirname(path)
    dataset_name = 'scannet200' if scannet200 else 'scannet'
    json.dump(coco_output, open(f'{parent_dir}/{dataset_name}_highres_{phase}.coco.json','w'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scannet200', action='store_true')
    args = parser.parse_args()
    
    for phase in ['train', 'val', 'two_scene', 'ten_scene']:
        print("Processing phase: ", phase)
        convert_scannet_to_coco(DATA_DIR, phase, scannet200=args.scannet200)
