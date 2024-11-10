# Instructions

Look at `globals_dirs.py` and change the folder paths where you would like to store the data.

## Downloading Matterport3D Mesh/PointCloud Data
- Download Matterport3D data from [HERE](https://niessner.github.io/Matterport/). 

## Processing the Mesh/PointCloud Data
Run 

```bash
python data_preparation/matterport3d/process_matterport_3d.py preprocess --data_dir PATH_TO_RAW_DATA --save_dir SAVE_DATA
```

## Pre-processed RGB-D image Data
You can download the preprocessed RGB-D data (~80G) for all scenes by

```bash
gdown --id 1mWU8jxrAlxsci7ste07qUo6S895kDt-f
```

## Process RGB-D Images on your own (Optional)
TBA: But maybe take a look at `process_matterport.py`


## Generate jsons in COCO Format
Run 

```bash
python data_preparation/matterport3d/m3d2coco.py
```
