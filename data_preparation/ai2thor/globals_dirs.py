
# SAVING DIRECTORY - CHANGE THIS
DATA_DIR = '/projects/katefgroup/language_grounding/SEMSEG_100k_reproduce/ai2thor_frames_512'
SPLITS_PATH = 'splits/scannet'

SPLITS = {
    'train':   f'{SPLITS_PATH}/scannet_train_512.txt',
    'val':     f'{SPLITS_PATH}/scannet_val_512.txt',
    'two_scene': f'{SPLITS_PATH}/two_scene.txt',
    'ten_scene': f'{SPLITS_PATH}/ten_scene.txt'
}
