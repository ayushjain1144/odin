import torch
import yaml
import os
import multiprocessing as mp
import numpy as np
from glob import glob
from natsort import natsorted
from tqdm import tqdm
import zipfile
import tempfile
from plyfile import PlyData, PlyElement
from pathlib import Path
import json
import pandas as pd
import re

from odin.global_vars import MATTERPORT_NAME_MAP,  MATTERPORT_NAME_TO_ID

import ipdb
st = ipdb.set_trace


DATA_PATH = '/projects/katefgroup/language_grounding/SEMSEG_100k/matterport'
PC_PROCESSED = "/projects/katefgroup/language_grounding/mask3d_processed/matterport"


if not os.path.exists(PC_PROCESSED):
    os.makedirs(PC_PROCESSED)
    
    
def make_mapping():
    tsv_file = 'tools/matterport/category_mapping.tsv'

    category_mapping = pd.read_csv(tsv_file, sep='\t', header=0)

    label_name = []
    label_id = []

    num_classes = 160


    label_all = category_mapping['nyuClass'].tolist()
    eliminated_list = ['void', 'unknown']
    mapping = np.zeros(len(label_all)+1, dtype=int) # mapping from category id
    instance_count = category_mapping['count'].tolist()
    ins_count_list = []
    counter = 1
    flag_stop = False


    for i, x in enumerate(label_all):
        if not flag_stop and isinstance(x, str) and x not in label_name and x not in eliminated_list:
            label_name.append(x)
            label_id.append(counter)
            mapping[i+1] = counter
            counter += 1
            ins_count_list.append(instance_count[i])
            if counter == num_classes+1:
                flag_stop = True
        elif isinstance(x, str) and x in label_name:
            # find the index of the previously appeared object name
            mapping[i+1] = label_name.index(x)+1
    return mapping

        

def load_ply(filepath):
    with open(filepath, "rb") as f:
        plydata = PlyData.read(f)
    data = plydata.elements[0].data
    coords = np.array([data["x"], data["y"], data["z"]], dtype=np.float32).T
    feats = None
    labels = None
    if ({"red", "green", "blue"} - set(data.dtype.names)) == set():
        feats = np.array([data["red"], data["green"], data["blue"]], dtype=np.uint8).T
    if "label" in data.dtype.names:
        labels = np.array(data["label"], dtype=np.uint32)
    return coords, feats, labels


# Load the pointclouds from openscene

# fix them

# save them as npy

# make yaml files as lists with dictionary having entry 'raw_filepath'


# def process_pc(pc_path):
#     pc, color, labels = torch.load(pc_path)
#     labels = labels.reshape(-1, 1)
#     color = (color + 1) * 127.5
#     segments = np.ones_like(labels)
#     normals = np.ones_like(pc)
    
#     points = np.concatenate(
#         [pc, color, normals, segments, labels], axis=-1)
    
#     return points


def _read_json(path):
    try:
        with open(path) as f:
            file = json.load(f)
    except json.decoder.JSONDecodeError:
        with open(path) as f:
            # in some files I have wrong escapechars as "\o", while it should be "\\o"
            file = json.loads(f.read().replace(r"\o", r"\\o"))
    return file


def process_pc(filepath, mode, mapping):
    scene_id = filepath.split("/")[-1]
    print(scene_id)
    with tempfile.TemporaryDirectory() as tempdir:
        with zipfile.ZipFile(f"{filepath}/region_segmentations.zip") as f:
            f.extractall(path=tempdir)
        region_files = (Path(tempdir) / scene_id).glob(r"*/*.ply")
        filebase = []
        for region_file in region_files:
            fbase = {
                "filepath": "",
                "raw_filepath": str(filepath),
                "raw_filepath_in_archive": str(region_file),
                "file_len": -1,
            }
            # reading both files and checking that they are fitting
            coords, features, _ = load_ply(region_file)
            file_len = len(coords)
            fbase["file_len"] = file_len
            points = np.hstack((coords, features, np.ones_like(coords)))
            
            # getting instance info
            instance_info_filepath = str(region_file).replace(
                ".ply", ".semseg.json"
            )
            segment_indexes_filepath = str(region_file).replace(
                ".ply", ".vsegs.json"
            )
            fbase["raw_instance_filepath"] = instance_info_filepath
            fbase["raw_segmentation_filepath"] = segment_indexes_filepath
            instance_db = _read_json(instance_info_filepath)
            segments = _read_json(segment_indexes_filepath)
            segments = np.array(segments["segIndices"])
            points = np.hstack((points, segments.reshape(-1, 1)))
            
                        
            labels = np.full((points.shape[0], 2), -1)
            for instance in instance_db["segGroups"]:
                segments_occupied = np.array(instance["segments"])
                occupied_indices = np.isin(segments, segments_occupied)
                labels[occupied_indices, 1] = instance["id"] 
                scannet_label = MATTERPORT_NAME_TO_ID.get(instance['label'], -1)
                labels[occupied_indices, 0] = scannet_label
                
            points = np.hstack((points, labels))
            
            region_num = int(re.search(r"\d+", region_file.stem).group(0))
            processed_filepath = f"{PC_PROCESSED}/{mode}/{scene_id}_{region_num}.npy"
            np.save(processed_filepath, points.astype(np.float32))
            fbase["filepath"] = processed_filepath
            filebase.append(fbase)
    return filebase


def process_scene(filepath, mode, mapping):
    filebase = process_pc(filepath, mode, mapping)
    print(f"In process_scene: {len(filebase)}")
    return filebase

# train_database.yaml, validation_database.yaml, train_validation_database.yaml
train_list = []
val_list = []

splits = ['train', 'val']
# splits = ['train']

mapping = make_mapping()

split_files = f'm3d_splits'
scenes = {}
scenes['train'] = list(set([path.split('_')[0] for path in open(f'{split_files}/m3d_train.txt', 'r').read().split('\n') if path != '']))
scenes['val'] = list(set([path.split('_')[0] for path in open(f'{split_files}/m3d_val.txt', 'r').read().split('\n') if path != '']))    

for split in splits:
    if split == "train":
        save_list = train_list
    else:
        save_list = val_list
        
    if not os.path.exists(f'{PC_PROCESSED}/{split}'):
        os.makedirs(f'{PC_PROCESSED}/{split}')
    # scenes = natsorted(glob(os.path.join(DATA_PATH, "**")))
    # scenes = ['/projects/katefgroup/language_grounding/SEMSEG_100k/matterport_3d/train/17DRP5sb8fy_region5.pth']
    # for scene in tqdm(scenes[split]):
        
    with mp.Pool(processes=32) as pool:
        print(f"To process {len(scenes[split])} scenes")
        filebase = pool.starmap(process_scene, [(f'{DATA_PATH}/{scene}', split, mapping) for scene in scenes[split]])
        print(f"Total results: ", len(filebase))
        for file in filebase:
            print(f"Result: ", len(file))
            save_list.extend(file)
            
        # save_list.extend(filebase[0])
        # name = scene.split('/')[-1].split('.')[0]
        # save_path = f'{PC_PROCESSED}/{split_}/{name}.npy'
        # filebase = process_scene(f'{DATA_PATH}/{scene}', split, mapping)
        # save_list.extend(filebase)

print(f"Train: {len(train_list)}")
print(f"Val: {len(val_list)}")

with open(f'{PC_PROCESSED}/train_database.yaml', 'w') as outfile:
    yaml.dump(train_list, outfile, default_flow_style=False)

with open(f'{PC_PROCESSED}/validation_database.yaml', 'w') as outfile:
    yaml.dump(val_list, outfile, default_flow_style=False)
    
with open(f'{PC_PROCESSED}/train_validation_database.yaml', 'w') as outfile:
    yaml.dump(train_list + val_list, outfile, default_flow_style=False)
    
print(len(train_list + val_list), "total")
    
        
    