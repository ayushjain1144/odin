set -e

export DETECTRON2_DATASETS="/projects/katefgroup/language_grounding/SEMSEG_100k"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0,1 python train_odin.py  --dist-url='tcp://127.0.0.1:9827' --num-gpus 2  --resume --config-file configs/scannet_context/3d.yaml \
OUTPUT_DIR /projects/katefgroup/language_grounding/bdetr2/arxiv_reproduce/s3dis_resnet_finetuned SOLVER.IMS_PER_BATCH 4 \
SOLVER.CHECKPOINT_PERIOD 500 TEST.EVAL_PERIOD 500 \
INPUT.FRAME_LEFT 12 INPUT.FRAME_RIGHT 12 INPUT.SAMPLING_FRAME_NUM 25 \
MODEL.WEIGHTS '/projects/katefgroup/language_grounding/odin_arxiv/scannet_resnet_47.8_73.3_32k_1.5k.pth' \
SOLVER.BASE_LR 1e-4 \
INPUT.IMAGE_SIZE 256 \
MODEL.CROSS_VIEW_CONTEXTUALIZE True \
INPUT.CAMERA_DROP True \
INPUT.STRONG_AUGS True \
INPUT.AUGMENT_3D True \
INPUT.VOXELIZE True \
INPUT.SAMPLE_CHUNK_AUG True \
MODEL.MASK_FORMER.TRAIN_NUM_POINTS 50000 \
MODEL.CROSS_VIEW_BACKBONE True \
DATASETS.TRAIN "('s3dis_train_single',)" \
DATASETS.TEST "('s3dis_val_single','s3dis_train_eval_single',)" \
MODEL.PIXEL_DECODER_PANET True \
MODEL.SEM_SEG_HEAD.NUM_CLASSES 13 \
MODEL.MASK_FORMER.TEST.SEMANTIC_ON True \
SKIP_CLASSES None \
USE_GHOST_POINTS True \
MODEL.FREEZE_BACKBONE False \
SOLVER.TEST_IMS_PER_BATCH 2 \
SAMPLING_STRATEGY "consecutive" \
USE_SEGMENTS False \
SOLVER.MAX_ITER 100000 \
DATALOADER.NUM_WORKERS 8 \
DATALOADER.TEST_NUM_WORKERS 2 \
MAX_FRAME_NUM 450 \
MODEL.MASK_FORMER.DICE_WEIGHT 6.0 \
MODEL.MASK_FORMER.MASK_WEIGHT 15.0 \
USE_WANDB True \
USE_MLP_POSITIONAL_ENCODING True \
INPUT.MIN_SIZE_TEST 256 \
INPUT.MAX_SIZE_TEST 256

# MODEL.WEIGHTS '/projects/katefgroup/language_grounding/odin_arxiv/scannet_resnet_47.8_73.3_32k_1.5k.pth' \
# reduce lr at 6.5k iterations to 1e-5, get the best checkpoint at 1.5k iterations
