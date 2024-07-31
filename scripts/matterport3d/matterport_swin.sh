set -e

export DETECTRON2_DATASETS="/projects/katefgroup/language_grounding/SEMSEG_100k"
MATTERPORT_DATA_DIR="/path/to/train_validation_database.yaml"

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0,1 python train_odin.py  --dist-url='tcp://127.0.0.1:1982' --num-gpus 2 --config-file configs/scannet_context/swin_3d.yaml \
OUTPUT_DIR /projects/katefgroup/language_grounding/bdetr2/arxiv_reproduce/matterport_swinb SOLVER.IMS_PER_BATCH 4 \
SOLVER.CHECKPOINT_PERIOD 4000 TEST.EVAL_PERIOD 4000 \
INPUT.FRAME_LEFT 7 INPUT.FRAME_RIGHT 7 INPUT.SAMPLING_FRAME_NUM 15 \
MODEL.WEIGHTS '/projects/katefgroup/language_grounding/odin_arxiv/m2f_coco_swin.pkl' \
SOLVER.BASE_LR 1e-4 \
INPUT.IMAGE_SIZE 512 \
MODEL.CROSS_VIEW_CONTEXTUALIZE True \
INPUT.CAMERA_DROP True \
INPUT.STRONG_AUGS True \
INPUT.AUGMENT_3D True \
INPUT.VOXELIZE True \
INPUT.SAMPLE_CHUNK_AUG True \
MODEL.MASK_FORMER.TRAIN_NUM_POINTS 50000 \
MODEL.CROSS_VIEW_BACKBONE True \
DATASETS.TRAIN "('matterport_train_single',)" \
DATASETS.TEST "('matterport_val_single','matterport_train_eval_single',)" \
MODEL.PIXEL_DECODER_PANET True \
MODEL.SEM_SEG_HEAD.NUM_CLASSES 160 \
MODEL.MASK_FORMER.TEST.SEMANTIC_ON True \
SKIP_CLASSES None \
USE_GHOST_POINTS True \
MODEL.FREEZE_BACKBONE False \
SOLVER.TEST_IMS_PER_BATCH 2 \
SAMPLING_STRATEGY "consecutive" \
USE_SEGMENTS True \
SOLVER.MAX_ITER 100000 \
DATALOADER.NUM_WORKERS 8 \
DATALOADER.TEST_NUM_WORKERS 2 \
MAX_FRAME_NUM 170 \
MODEL.MASK_FORMER.DICE_WEIGHT 6.0 \
MODEL.MASK_FORMER.MASK_WEIGHT 15.0 \
USE_WANDB True \
USE_MLP_POSITIONAL_ENCODING True \
INPUT.MIN_SIZE_TEST 512 \
INPUT.MAX_SIZE_TEST 512 \
MATTERPORT_DATA_DIR $MATTERPORT_DATA_DIR


# reduce lr at 68k iterations to 1e-5, get the best checkpoint at 16k iterations for instance segmentation \
# and 20k iterations for semantic segmentation

# USE MATTERPORT_ALL_CLASSES_TO_21 True flag for evaluating on the 21 benchmark classes
