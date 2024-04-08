set -e

export DETECTRON2_DATASETS="/projects/katefgroup/language_grounding/SEMSEG_100k"
export DETECTRON2_DATASETS_2D="/projects/katefgroup/datasets"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0,1 python train_odin.py  --dist-url='tcp://127.0.0.1:7292' --num-gpus 2 --resume  --config-file configs/scannet_context/3d.yaml \
OUTPUT_DIR /projects/katefgroup/language_grounding/bdetr2/arxiv_reproduce/scannet_coco_joint SOLVER.IMS_PER_BATCH 6 \
SOLVER.CHECKPOINT_PERIOD 4000 TEST.EVAL_PERIOD 4000 \
INPUT.FRAME_LEFT 12 INPUT.FRAME_RIGHT 12 INPUT.SAMPLING_FRAME_NUM 25 \
INPUT.FRAME_LEFT_2D 0 INPUT.FRAME_RIGHT_2D 0 INPUT.SAMPLING_FRAME_NUM_2D 1 \
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
MODEL.MASK_FORMER.TRAIN_NUM_POINTS_2D 12544 \
MODEL.CROSS_VIEW_BACKBONE True \
DATASETS.TRAIN "('scannet_context_instance_train_20cls_single_highres_100k','coco_2017_train',)" \
DATASETS.TEST "('scannet_context_instance_val_20cls_single_highres_100k','scannet_context_instance_train_eval_20cls_single_highres_100k','coco_2017_val')" \
MODEL.PIXEL_DECODER_PANET True \
MODEL.SEM_SEG_HEAD.NUM_CLASSES 20 \
MODEL.MASK_FORMER.TEST.SEMANTIC_ON True \
SKIP_CLASSES "[19, 20]" \
SKIP_CLASSES_2D None \
USE_GHOST_POINTS True \
MODEL.FREEZE_BACKBONE False \
SOLVER.TEST_IMS_PER_BATCH 2 \
SAMPLING_STRATEGY "consecutive" \
USE_SEGMENTS True \
SOLVER.MAX_ITER 200000 \
DATALOADER.NUM_WORKERS 8 \
DATALOADER.TEST_NUM_WORKERS 2 \
MAX_FRAME_NUM -1 \
MODEL.MASK_FORMER.DICE_WEIGHT 6.0 \
MODEL.MASK_FORMER.MASK_WEIGHT 15.0 \
USE_WANDB True \
USE_MLP_POSITIONAL_ENCODING True \
MODEL.OPEN_VOCAB True \
NON_PARAM_SOFTMAX True \
MODEL.MAX_SEQ_LEN 256 \
MULTI_TASK_TRAINING True \
TRAIN_3D True \
TRAIN_2D True \
DATASETS.TRAIN_3D "('scannet_context_instance_train_20cls_single_highres_100k',)" \
DATASETS.TRAIN_2D "('coco_2017_train',)" \
DATASETS.TEST_2D_ONLY "('coco_2017_val',)" \
DATASETS.TEST_3D_ONLY "('scannet_context_instance_val_20cls_single_highres_100k','scannet_context_instance_train_eval_20cls_single_highres_100k',)" \
MODEL.DECODER_2D True \
MODEL.DECODER_3D True \
TEST.EVAL_3D True \
TEST.EVAL_2D True  \
SOLVER.IMS_PER_BATCH_3D 6 \
SOLVER.IMS_PER_BATCH_2D 16 \
INPUT.IMAGE_SIZE_2D 1024 \
INPUT.MIN_SIZE_TEST_2D 800 \
INPUT.MAX_SIZE_TEST_2D 1333 \
AUGMENT_WITH_3D_SCALE True \
PROB "[0.3, 0.7]"

# MODEL.WEIGHTS '/projects/katefgroup/language_grounding/bdetr2/arxiv_reproduce/scannet_coco_joint_train_ckpt/model_0087999.pth'
# reduce lr at 88k iterations to 1e-5 and get the best cckpt at 19k iterations
