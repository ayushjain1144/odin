import prior
import numpy as np
import random
from PIL import Image
import os
from tqdm import tqdm

from data_preparation.ai2thor.controller_custom import Controller
from ai2thor_utils import get_origin_T_camX
from odin.global_vars import AI2THOR_NAME_TO_ID
from odin.utils.inpaint_depth import inpaint_depth
from data_preparation.ai2thor.globals_dirs import DATA_DIR, SPLITS_PATH, SPLITS

import ipdb
st = ipdb.set_trace

dataset = prior.load_dataset("procthor-10k")

# set seed
random.seed(42)
np.random.seed(42)

NUM_TRAIN_SCENES = 1300
NUM_VAL_SCENES = 320
IMAGE_SIZE = 512

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

train_scenes = dataset['train']
val_scenes = dataset['val']

def dump_scene(scene, scene_id):
    controller = Controller(
                scene=scene,
                visibilityDistance=1.5,
                gridSize=0.5,
                width=IMAGE_SIZE,
                height=IMAGE_SIZE,
                fieldOfView=90,
                renderObjectImage=True,
                renderDepthImage=True,
                renderInstanceSegmentation=True,
                x_display=str(0),
                snapToGrid=False,
                rotateStepDegrees=90,
        )

    event = controller.step(action="GetReachablePositions")
    reachable_positions = event.metadata["actionReturn"]
    print("Reachable positions:", len(reachable_positions))

    unique_obj_ids = []

    if len(reachable_positions) > 120:
        reachable_positions = random.sample(reachable_positions, 120)

    for i, position in enumerate(reachable_positions):
        rotation = random.choice(range(360))
        event = controller.step(action="Teleport", position=position, rotation=rotation)

        rgb_image = Image.fromarray(event.frame)
        pose = get_origin_T_camX(event)
        semantic = event.instance_segmentation_frame
        color_to_object_id = event.color_to_object_id
        instance_masks = event.instance_masks
        depth = event.depth_frame

        obj_ids = np.unique(semantic.reshape(-1, semantic.shape[2]), axis=0)

        instance_image = np.zeros((semantic.shape[0], semantic.shape[1])).astype(np.int32)

        obj_metadata_IDs = []
        for obj_m in event.metadata['objects']: #objects:
            obj_metadata_IDs.append(obj_m['objectId'])

        if len(obj_metadata_IDs) > 100:
            print("Too many objects in scene, skipping")
            controller.stop()
            return None

        for obj_id in range(obj_ids.shape[0]):
            try:
                obj_color = tuple(obj_ids[obj_id])
                object_id = color_to_object_id[obj_color]
            except:
                continue

            if object_id not in obj_metadata_IDs:
                continue

            obj_meta_index = obj_metadata_IDs.index(object_id)
            obj_meta = event.metadata['objects'][obj_meta_index]

            obj_category_name = obj_meta['objectType']
            # obj_id_name = obj_meta['objectId']

            i_mask = instance_masks[object_id]
            if obj_category_name not in AI2THOR_NAME_TO_ID:
                continue

            if object_id not in unique_obj_ids:
                unique_obj_ids.append(object_id)

            class_id = AI2THOR_NAME_TO_ID[obj_category_name]
            inst_id = obj_metadata_IDs.index(object_id) + 1
            instance_image[i_mask] = class_id * 200 + inst_id

        # instance image
        im = Image.fromarray(instance_image.astype(np.int32))
        # if (np.array(im) % 200).max() > 100:
        #     st()
        if not os.path.exists(f"{DATA_DIR}/scene_{scene_id}/instance"):
            os.makedirs(f"{DATA_DIR}/scene_{scene_id}/instance")
        im.save(f"{DATA_DIR}/scene_{scene_id}/instance/{i}.png")
        
        # rgb image
        color = rgb_image
        if  not os.path.exists(f"{DATA_DIR}/scene_{scene_id}/color"):
            os.makedirs(f"{DATA_DIR}/scene_{scene_id}/color")
        color.save(f"{DATA_DIR}/scene_{scene_id}/color/{i}.jpg")

        # depth image
        depth = Image.fromarray((depth * 1000).astype(np.uint16))
        if not os.path.exists(f"{DATA_DIR}/scene_{scene_id}/depth"):
            os.makedirs(f"{DATA_DIR}/scene_{scene_id}/depth")
        depth.save(f"{DATA_DIR}/scene_{scene_id}/depth/{i}.png")
        
        # inpaint depth image
        depth = inpaint_depth(np.array(depth).astype(np.float32)).astype(np.uint16)
        depth = Image.fromarray(depth)
        if not os.path.exists(f"{DATA_DIR}/scene_{scene_id}/depth_inpainted"):
            os.makedirs(f"{DATA_DIR}/scene_{scene_id}/depth_inpainted")
        depth.save(f"{DATA_DIR}/scene_{scene_id}/depth_inpainted/{i}.png")

        # pose in txt 
        if not os.path.exists(f"{DATA_DIR}/scene_{scene_id}/pose"):
            os.makedirs(f"{DATA_DIR}/scene_{scene_id}/pose")
        np.savetxt(f"{DATA_DIR}/scene_{scene_id}/pose/{i}.txt", pose)

    print("Total instances in scene:", len(unique_obj_ids))

    controller.stop()
    return True
        


# write data for all scenes
idx = 0
valid_train_indices = []
for i, scene in tqdm(enumerate(train_scenes)):
    print("Processing scene", idx)
    status = dump_scene(scene, idx)
    if status:
        idx += 1
        valid_train_indices.append(i)
    
    if idx == NUM_TRAIN_SCENES:
        break

# write out train/val splits
if not os.path.exists(f"{SPLITS_PATH}"):
    os.makedirs(f"{SPLITS_PATH}")
    
with open(f"{SPLITS['train']}", "w") as f:
    for i in range(idx):
        f.write(f"scene_{i}" + "\n")

train_idx = idx
print("Train scenes:", train_idx)

# write ten random training scenes
with open(f"{SPLITS['ten_scene']}", "w") as f:
    ten_scenes = random.sample(np.arange(train_idx).tolist(), 10)
    for i in ten_scenes:
        f.write(f"scene_{i}" + "\n")
        
# write two random training scenes
with open(f"{SPLITS['two_scene']}", "w") as f:
    two_scenes = random.sample(np.arange(train_idx).tolist(), 2)
    for i in two_scenes:
        f.write(f"scene_{i}" + "\n")
    

valid_val_indices = []
for i, scene in tqdm(enumerate(val_scenes)):
    print("Processing scene", idx)
    status = dump_scene(scene, idx)
    if status:
        idx += 1
        valid_val_indices.append(i)

    if idx == NUM_TRAIN_SCENES + NUM_VAL_SCENES:
        break

with open(f"{SPLITS['val']}", "w") as f:
    for i in range(train_idx, idx):
        f.write(f"scene_{i}" + "\n")

print("Val scenes:", idx - train_idx)





