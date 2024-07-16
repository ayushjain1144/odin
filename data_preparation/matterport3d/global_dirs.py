# SAVING DIRECTORY - CHANGE THIS
DATA_DIR = '/fsx-cortex/ayushjain1144/odin_data/SEMSEG_100k/matterport_frames'
SPLITS_PATH = 'splits/m3d_splits'
PC_DATA_DIR = "/fsx-cortex/ayushjain1144/odin_data/SEMSEG_100k/matterport3d_meshes/matterport"
PC_PROCESSED_PATH = "/fsx-cortex/ayushjain1144/odin_data/mask3d_processed/matterport"

SPLITS = {
    'train':   f'{SPLITS_PATH}/m3d_train.txt',
    'val':     f'{SPLITS_PATH}/m3d_val.txt',
    'two_scene': f'{SPLITS_PATH}/two_scene.txt',
    'ten_scene': f'{SPLITS_PATH}/ten_scene.txt'
}
