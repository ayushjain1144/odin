import multiprocessing as mp
import os

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

load_depth = "depth"
save_depth = "depth_inpainted"

# Change this to the path of your data folder
INPUT_FOLDER = "/projects/katefgroup/language_grounding/SEMSEG_100k/s3dis_frames_fixed"


def inpaint_depth(depth):
    """
    inpaints depth using opencv
    Input: torch tensor with depthvalues: H, W
    Output: torch tensor with depthvalues: H, W
    """
    depth_inpaint = cv2.inpaint(
        depth, (depth == 0).astype(np.uint8), 5, cv2.INPAINT_TELEA
    )
    depth[depth == 0] = depth_inpaint[depth == 0]
    return depth


def process_depth(scene, depth_image):
    # load depth image
    depth = np.array(
        Image.open(os.path.join(INPUT_FOLDER, scene, load_depth, depth_image))
    )
    # inpaint depth image
    depth = inpaint_depth(depth.astype(np.float32)).astype(np.int32)
    # save depth image
    if not os.path.exists(os.path.join(INPUT_FOLDER, scene, save_depth)):
        os.makedirs(os.path.join(INPUT_FOLDER, scene, save_depth))
    Image.fromarray(depth).save(
        os.path.join(INPUT_FOLDER, scene, save_depth, depth_image)
    )


def process_scene(scene):
    print(scene)
    for depth_image in os.listdir(os.path.join(INPUT_FOLDER, scene, load_depth)):
        process_depth(scene, depth_image)


if __name__ == "__main__":
    NUM_PROCESSES = 16  # use all available CPU cores

    with mp.Pool(processes=32) as pool:
        # map the scenes to the pool of workers
        pool.map(process_scene, os.listdir(INPUT_FOLDER))
