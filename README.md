[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/odin-a-single-model-for-2d-and-3d-perception/3d-instance-segmentation-on-scannet200)](https://paperswithcode.com/sota/3d-instance-segmentation-on-scannet200?p=odin-a-single-model-for-2d-and-3d-perception)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/odin-a-single-model-for-2d-and-3d-perception/3d-semantic-segmentation-on-scannet200)](https://paperswithcode.com/sota/3d-semantic-segmentation-on-scannet200?p=odin-a-single-model-for-2d-and-3d-perception)

# ODIN: A Single Model for 2D and 3D Segmentation (CVPR 2024 Highlight)


Authors: [Ayush Jain](https://ayushjain1144.github.io/), [Pushkal Katara](https://pushkalkatara.github.io/), [Nikolaos Gkanatsios](https://github.com/nickgkan), [Adam W. Harley](https://adamharley.com/), [Gabriel Sarch](https://gabesarch.me/), [Kriti Aggarwal](https://scholar.google.com/citations?hl=en&user=iB-h89EAAAAJ), [Vishrav Chaudhary](https://scholar.google.com/citations?hl=en&user=CXlp-fcAAAAJ), [Katerina Fragkiadaki](https://www.cs.cmu.edu/~katef/).

Official implementation of ["ODIN: A Single Model for 2D and 3D Segmentation"](https://odin-seg.github.io/), CVPR 2024.

<div align="center">
  <img src="https://odin-seg.github.io/data/teaser_v6-1.png" width="100%" height="100%"/>
</div><br/>


## Installation
Make sure you are using a newer version of GCC>=9.2.0

```bash
export TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6"
conda create -n odin python=3.10
conda activate odin
pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
pip install -r requirements.txt
sh init.sh
```

## Data Preparation

Please refer to README in data_preparation folder for individual datasets. For eg. ScanNet data preparation [README](data_preparation/scannet/README.md)

## Usage

We provide training scripts for various datasets in `scripts` folder. Please refer to these scripts for training ODIN. 

- Modify `DETECTRON2_DATASETS` to the path where you store the Posed RGB-D data. 
- Modify MODEL.WEIGHTS to load pre-trained weights. We use weights from Mask2Former. You can download the [Mask2Former-ResNet](https://huggingface.co/katefgroup/odin/resolve/main/m2f_coco.pkl) and [Mask2Former-Swin Weights](https://huggingface.co/katefgroup/odin/resolve/main/m2f_coco_swin.pkl). MODEL.WEIGHTS can also accept link to the checkpoint as well, so you can directly supply these links as the argument value.
- ODIN Pre-trained weights are provided below in the Model-Zoo. Simply point to these weights using the MODEL.WEIGHTS to run inference. You would also need to add `--eval-only` flag for running evaluation. 
- `SOLVER.IMS_PER_BATCH` controls the batch size. This is effective batch size i.e. if you are running on 2 GPUs and the batch size is set to 6, you are using bs=3 per GPU. 
- `SOLVER.TEST_IMS_PER_BATCH` controls the (effective) test batch size. Since, there are variable number of images in a scene, we use bs=1 per GPU at test time. `MAX_FRAME_NUM=-1` means that it loads all images in a scene for inference, which is our usual strategy. In some datasets, the images can simply be too large, thus there we actually set a maximum limit on images. 
- `INPUT.SAMPLING_FRAME_NUM` controls the number of images we sample at test time -- for eg. in ScanNet, we train on 25 image chunks at training time. 
- `CHECKPOINT_PERIOD` is the number of iterations after which a checkpoint is saved. `EVAL_PERIOD` specifies the number of steps after which the eval is run. 
- `OUTPUT_DIR` stores the checkpoints and the tensorboard logs. `--resume` resumes the training from the last checkpoint stored in `OUTPUT_DIR`. If no checkpoint is present, it loads the weights from `MODEL.WEIGHTS`



## Model Zoo

ScanNet Instance Segmentation
| Dataset | mAP | mAP@25 | Config | Checkpoint
|:-:|:-:|:-:|:-:|:-:|
| ScanNet val (ResNet50)  | 47.8 | 83.6 | [config](scripts/scannet/scannet_resnet.sh) | [checkpoint](https://huggingface.co/katefgroup/odin/resolve/main/scannet_resnet_47.8_73.3_32k_1.5k.pth) 
| ScanNet val (Swin-B)  | 50.0 | 83.6 | [config](scripts/scannet/scannet_swin.sh) | [checkpoint](https://huggingface.co/katefgroup/odin/resolve/main/scannet_swin_50.0_64k_6k.pth) 
| ScanNet test (Swin-B) | 47.7 | 86.2 | [config](scripts/scannet/scannet_swin_benchmark.sh) | [checkpoint](https://huggingface.co/katefgroup/odin/resolve/main/scannet_swin_50.0_64k_6k.pth) 

ScanNet Semantic Segmentation
| Dataset | mIoU | Config | Checkpoint
|:-:|:-:|:-:|:-:|
| ScanNet val (ResNet50)  | 73.3 | [config](scripts/scannet/scannet_resnet.sh) | [checkpoint](https://huggingface.co/katefgroup/odin/resolve/main/scannet_resnet_47.8_73.3_32k_1.5k.pth) 
| ScanNet val (Swin-B)  | 77.8 | [config](scripts/scannet/scannet_swin.sh) | [checkpoint](https://huggingface.co/katefgroup/odin/resolve/main/scannet_swin_semantic_77.8_64k_2k.pth) 
| ScanNet test (Swin-B) | 74.4 | [config](scripts/scannet/scannet_benchmark.sh) | [checkpoint](https://huggingface.co/katefgroup/odin/resolve/main/scannet_swin_semantic_77.8_64k_2k.pth)

Joint 2D-3D on ScanNet and COCO
| Model | mAP (ScanNet) | mAP25 (ScanNet) | mAP (COCO) | Config | Checkpoint
|:-:|:-:|:-:|:-:|:-:|:-:|
| ODIN | 49.1 | 83.1 | 41.2 | [config](scripts/joint_coco_scannet/scannet_coco_resnet.sh) | [checkpoint](https://huggingface.co/katefgroup/odin/resolve/main/scannet_coco_joint_49.1_41.2_88k_19k.pth)

ScanNet200 Instance Segmentation
| Dataset | mAP | mAP@25 | Config | Checkpoint
|:-:|:-:|:-:|:-:|:-:|
| ScanNet200 val (ResNet50)  | 25.6 | 36.9 | [config](scripts/scannet200/scannet200_resnet.sh) | [checkpoint](https://huggingface.co/katefgroup/odin/resolve/main/scannet200_resnet_25.6_35.8_52k_5k.pth) 
| ScanNet200 val (Swin-B)  | 31.5 | 45.3 | [config](scripts/scannet200/scannet200_swin.sh) | [checkpoint](https://huggingface.co/katefgroup/odin/resolve/main/scannet200_swin_31.5_76k_5.5k.pth) 
| ScanNet200 test (Swin-B) | 27.2 | 39.4 | [config](scripts/scannet200/scannet200_swin_benchmark.sh) | [checkpoint](https://huggingface.co/katefgroup/odin/resolve/main/scannet200_swin_31.5_76k_5.5k.pth) 


ScanNet200 Semantic Segmentation
| Dataset | mIoU | Config | Checkpoint
|:-:|:-:|:-:|:-:|
| ScanNet200 val (ResNet50)  | 35.8 | [config](scripts/scannet200/scannet200_resnet.sh) | [checkpoint](https://huggingface.co/katefgroup/odin/resolve/main/scannet200_resnet_25.6_35.8_52k_5k.pth) 
| ScanNet200 val (Swin-B)  | 40.5 | [config](scripts/scannet/scannet_val.sh) | [checkpoint](https://huggingface.co/katefgroup/odin/resolve/main/scannet200_swin_40.5_76k_3k.pth) 
| ScanNet test (Swin-B) | 36.8 | [config](scripts/scannet/scannet_benchmark.sh) | [checkpoint](https://huggingface.co/katefgroup/odin/resolve/main/scannet200_swin_40.5_76k_3k.pth)

AI2THOR Semantic and Instance Segmentation
| Dataset | mAP | mAP@25 | mIoU | Config | Checkpoint
|:-:|:-:|:-|:-:|:-:|:-:|
| AI2THOR val (ResNet)  | 63.8 | 80.2 | 71.5 | [config](scripts/ai2thor/ai2thor_resnet.sh) | [checkpoint](https://huggingface.co/katefgroup/odin/resolve/main/ai2thor_resnet_63.8_71.4_1.5k.pth)
| AI2RHOR val (Swin)  | 64.3 | 78.6 | 71.4 | [config](scripts/ai2thor/ai2thor_swin.sh) | [checkpoint](https://huggingface.co/katefgroup/odin/resolve/main/ai2thor_swin_64.3_71.4_112k_4.5k.pth)

Matterport3D Instance Segmentation
| Dataset | mAP | mAP@25 | Config | Checkpoint
|:-:|:-:|:-|:-:|:-:|
| Matterport3D val (ResNet)  | 11.5 | 27.6 | [config](scripts/matterport3d/matterport_resnet.sh) | [checkpoint](https://huggingface.co/katefgroup/odin/resolve/main/matterport_resnet_11.5_22.4_84k_13k.pth)
| Matterport val (Swin)  | 14.5 | 36.8 | [config](scripts/matterport3d/matterport_swin.sh) | [checkpoint](https://huggingface.co/katefgroup/odin/resolve/main/matterport_swin_14.5_68k_16k.pth)

Matterport3D Semantic Segmentation
| Dataset | mIoU | mAcc | Config | Checkpoint 
|:-:|:-:|:-:|:-:|:-:|
| Matterport3D val (ResNet)  | 22.4 | 28.5 | [config](scripts/matterport3d/matterport_resenet.sh) | [checkpoint](https://huggingface.co/katefgroup/odin/resolve/main/matterport_resnet_11.5_22.4_84k_13k.pth) 
| Matterport3D val (Swin) | 28.6 | 38.2 | [config](scripts/matterport3d/matterport_swin.sh) | [checkpoint](https://huggingface.co/katefgroup/odin/resolve/main/matterport_swin_28.6_68k_20k.pth)

S3DIS Instance Segmentation
| Dataset | mAP | mAP@25 | Config | Checkpoint
|:-:|:-:|:-:|:-:|:-:|
| S3DIS Area5 (ResNet50-Scratch) | 36.3 | 61.2 | [config](scripts/s3dis/s3dis_resnet_scratch.sh) | [checkpoint](https://huggingface.co/katefgroup/odin/resolve/main/s3dis_resnet_36.3_59.7_24k_0.5k.pth)
| S3DIS Area5 (ResNet50-Fine-Tuned) | 44.7 | 67.5 | [config](scripts/s3dis/s3dis_resnet_scratch.sh) | [checkpoint](https://huggingface.co/katefgroup/odin/resolve/main/s3dis_resnet_finetuned_44.7_66.7_6.5k_1.5k.pth)
| S3DIS Area5 (Swin-B) | 43.0 | 70.0 | [config](scripts/s3dis/s3dis_swin_finetuned.sh) | [checkpoint](https://huggingface.co/katefgroup/odin/resolve/main/s3dis_swin_finetuned_43.0_68.6_4.5k_3k.pth)

S3DIS Semantic Segmentation
| Dataset | mIoU | Config | Checkpoint
|:-:|:-:|:-:|:-:|
| S3DIS (ResNet50)  | 59.7 | [config](scripts/s3dis/s3dis_resnet_finetuned.sh) | [checkpoint](https://huggingface.co/katefgroup/odin/resolve/main/s3dis_resnet_36.3_59.7_24k_0.5k.pth)
| Swin-B | 68.6 | [config](scripts/s3dis_swin_finetuned.sh) | [checkpoint](https://huggingface.co/katefgroup/odin/resolve/main/s3dis_swin_finetuned_43.0_68.6_4.5k_3k.pth)


## Training Logs:

Please find training logs for all models [here](https://huggingface.co/katefgroup/odin_logs/tensorboard)


## <a name="CitingODIN"></a>Citing ODIN
If you find ODIN useful in your research, please consider citing:


```BibTeX
@misc{jain2024odin,
      title={ODIN: A Single Model for 2D and 3D Perception}, 
      author={Ayush Jain and Pushkal Katara and Nikolaos Gkanatsios and Adam W. Harley and Gabriel Sarch and Kriti Aggarwal and Vishrav Chaudhary and Katerina Fragkiadaki},
      year={2024},
      eprint={2401.02416},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## License

Shield: [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The majority of ODIN is licensed under a [MIT License](LICENSE).



## Acknowledgement

Parts of this code were based on the codebase of [Mask2Former](https://github.com/facebookresearch/Mask2Former) and [Mask3D](https://github.com/JonasSchult/Mask3D).
