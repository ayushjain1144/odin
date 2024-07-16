# Instructions

Look at `globals_dirs.py` and change the folder paths where you would like to store the data.

## Downloading S3DIS Mesh/PointCloud Data
- Download S3DIS data [HERE](http://buildingparser.stanford.edu/dataset.html#Download). 

S3DIS data has some bugs, please refer [this](https://github.com/JonasSchult/Mask3D/issues/8#issuecomment-1279535948) to fix them. 

## Processing the Mesh/PointCloud Data

Run

```bash
python data_preparation/s3dis/s3dis_preprocessing.py preprocess --data_dir PATH_TO_RAW_DATA --save_dir SAVE_DATA
```

## Pre-processed RGB-D image Data
You can download the preprocessed RGB-D data (~80G) for all scenes by

```bash
gdown --id 1mQlc1th8XHW59bOdOjINfRfzP5V-EKNW
```

## Process RGB-D Images on your own (Optional)
TBA

## Generate jsons in COCO Format
Run 

```bash
python data_preparation/s3dis/s3dis2coco.py
```
