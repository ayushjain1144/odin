
# SAVING DIRECTORY - CHANGE THIS
DATA_DIR = '/projects/katefgroup/odin/scannet'
SPLITS_PATH = 'splits/scannet_splits'

SPLITS = {
    'train':   f'{SPLITS_PATH}/scannetv2_train.txt',
    'val':     f'{SPLITS_PATH}/scannetv2_val.txt',
    'trainval': f'{SPLITS_PATH}/scannetv2_trainval.txt',
    'test':    f'{SPLITS_PATH}/scannetv2_test.txt',
    'two_scene': f'{SPLITS_PATH}/two_scene.txt',
    'ten_scene': f'{SPLITS_PATH}/ten_scene.txt'
}
