set -e

export DETECTRON2_DATASETS="/projects/katefgroup/embodied_llm/data/teach_helper_traj_data/"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0,1 python train_odin.py --resume  --dist-url='tcp://127.0.0.1:9811' --num-gpus 2 --resume --config-file configs/scannet_context/3d.yaml \
OUTPUT_DIR /projects/katefgroup/language_grounding/bdetr2/arxiv_reproduce/alfred_resnet SOLVER.IMS_PER_BATCH 4 \
SOLVER.CHECKPOINT_PERIOD 4000 TEST.EVAL_PERIOD 4000 \
INPUT.FRAME_LEFT 7 INPUT.FRAME_RIGHT 7 INPUT.SAMPLING_FRAME_NUM 15 \
MODEL.WEIGHTS '/projects/katefgroup/language_grounding/odin_arxiv/ai2thor_resnet_63.8_71.4_1.5k.pth' \
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
DATASETS.TRAIN "('alfred_train_single',)" \
DATASETS.TEST "('alfred_valid_seen_single','alfred_valid_unseen_single')" \
MODEL.PIXEL_DECODER_PANET True \
MODEL.SEM_SEG_HEAD.NUM_CLASSES 123 \
SKIP_CLASSES "[116, 117]" \
MODEL.MASK_FORMER.TEST.SEMANTIC_ON True \
MODEL.FREEZE_BACKBONE False \
SOLVER.TEST_IMS_PER_BATCH 2 \
SAMPLING_STRATEGY "consecutive" \
SOLVER.MAX_ITER 200000 \
DATALOADER.NUM_WORKERS 8 \
DATALOADER.TEST_NUM_WORKERS 2 \
MAX_FRAME_NUM 50 \
INPUT.INPAINT_DEPTH False \
IGNORE_DEPTH_MAX 15.0 \
MODEL.SUPERVISE_SPARSE True \
TEST.EVAL_SPARSE True \
MODEL.MASK_FORMER.DICE_WEIGHT 6.0 \
MODEL.MASK_FORMER.MASK_WEIGHT 15.0 \
USE_WANDB True \
USE_MLP_POSITIONAL_ENCODING True \
INPUT.MIN_SIZE_TEST 512 \
INPUT.MAX_SIZE_TEST 512 \
HIGH_RES_SUBSAMPLE True \
HIGH_RES_INPUT True \
MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 150

# MODEL.WEIGHTS '/projects/katefgroup/language_grounding/odin_arxiv/ai2thor_resnet_63.8_71.4_1.5k.pth' \
# reduce lr at 96k iterations to 1e-5 to get the best checkpoint at 17k iterations
