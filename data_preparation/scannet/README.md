## ScanNet and ScanNet200 Data Preparation

Look at `data_preparation/dataset/globals_dirs.py` and change the folder paths where you would like to store the data.


## Downloading ScanNet Mesh/PointCloud Data
- Download ScanNet v2 data from [HERE](https://github.com/ScanNet/ScanNet). Let DATA_ROOT be the path to folder that contains the downloaded annotations. Under DATA_ROOT there should be a folder scans. Under scans there should be folders with names like scene0001_01. We need `_vh_clean_2.ply`, `_vh_clean_2.0.010000.segs.json`, `_vh_clean_2.labels.ply`, `_vh_clean.aggregation.json`


## Processing the Mesh/PointCloud Data
 
For ScanNet, execute

```bash
python data_preparation/scannet/scannet_preprocessing.py preprocess --data_dir PATH_TO_RAW_SCANS --save_dir SAVE_DATA
```

Add `--scannet200 True` for ScanNet200

Make sure to change `SCANNET_DATA_DIR` in `odin/config.py` to the `SAVE_DATA/train_validation_database.yaml'

Similarly, change  `SCANNET200_DATA_DIR` in `odin/config.py` to the `SAVE_DATA/train_validation_database.yaml'



## Pre-processed RGB-D image Data
We provide preprocessed RGB-D data (~80G) for all scenes. You can downloading it using gdown in the data directory.

```bash
gdown --id 1Xq84J9Gl9CVns_4Q0gDBxcPoA7hSf-WY
```

## Process RGB-D Images on your own (Optional)
You can skip this if you just want to use our preprocessed RGB-D data

- First download the .sens files as well by using `--type .sens` argument with the scannet download script. 
- Execute the following script (make sure to change the data directory paths in the script)

```bash
 python data_preparation/scannet/preprocess_sens.sh 
```

## Generate jsons in COCO Format

For ScanNet, execute:
```bash
python data_preparation/scannet/scannet2coco.py
```

For ScanNet200, add `--scannet200` to the above command


## Instructions for setting up test set (Coming Soon)
