# SAVING DIRECTORY - CHANGE THIS
DATA_DIR = '/fsx-cortex/ayushjain1144/odin_data/SEMSEG_100k/s3dis_frames_fixed'
SPLITS_PATH = 'splits/s3dis_splits'

SPLITS = {
    'train':   f'{SPLITS_PATH}/s3dis_train.txt',
    'val':     f'{SPLITS_PATH}/s3dis_val.txt',
    'two_scene': f'{SPLITS_PATH}/two_scene.txt',
    'ten_scene': f'{SPLITS_PATH}/ten_scene.txt'
}
