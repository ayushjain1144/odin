# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN


def add_maskformer2_config(cfg):
    """
    Add config for MASK_FORMER.
    """
    # NOTE: configs from original maskformer
    # data config
    # select the dataset mapper
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic"
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    # mask_former model config
    cfg.MODEL.MASK_FORMER = CN()

    # loss
    cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION = True
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 20.0

    # transformer config
    cfg.MODEL.MASK_FORMER.NHEADS = 8
    cfg.MODEL.MASK_FORMER.DROPOUT = 0.1
    cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MASK_FORMER.ENC_LAYERS = 0
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 6
    cfg.MODEL.MASK_FORMER.PRE_NORM = False

    cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100

    cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "res5"
    cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ = False

    # mask_former inference config
    cfg.MODEL.MASK_FORMER.TEST = CN()
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32

    # pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    # adding transformer in pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    # pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "BasePixelDecoder"

    # swin transformer backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False

    # NOTE: maskformer2 extra configs
    # transformer module
    cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME = "MultiScaleMaskedTransformerDecoder"

    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0

    # MSDeformAttn encoder configs
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS = 8

    # point loss configs
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS = 112 * 112
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO = 3.0
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO = 0.75

    cfg.MODEL.CROSS_VIEW_BACKBONE = False
    cfg.MODEL.DECODER_2D = True
    cfg.MODEL.DECODER_3D = False
    cfg.MODEL.FREEZE_BACKBONE = False
    cfg.MODEL.SEM_SEG_HEAD.NO_SKIP_CONN = False
    cfg.INPUT.FRAME_LEFT = 0
    cfg.INPUT.FRAME_RIGHT = 0
    cfg.INPUT.SAMPLING_FRAME_NUM = 1
    cfg.MODEL.MASK_FORMER.NON_PARAM_QUERY = False
    cfg.WEIGHT_TYING = False
    cfg.MODEL.CROSS_VIEW_CONTEXTUALIZE = False
    cfg.MODEL.SUPERVISE_SPARSE = False
    cfg.TEST.EVAL_SPARSE = False
    cfg.MODEL.OPEN_VOCAB = False
    cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS_2D = cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS
  

def add_maskformer2_video_config(cfg):
    # video data
    # DataLoader
    cfg.INPUT.FRAME_RIGHT = 2
    cfg.INPUT.FRAME_LEFT = 2
    cfg.INPUT.SAMPLING_FRAME_NUM = cfg.INPUT.FRAME_RIGHT + cfg.INPUT.FRAME_LEFT + 1
    cfg.INPUT.AUGMENTATIONS = [] # "brightness", "contrast", "saturation", "rotation"
    cfg.MODEL.DECODER_3D = False
    cfg.TEST.EVAL_3D = False
    cfg.MODEL.CROSS_VIEW_CONTEXTUALIZE = False
    cfg.INPUT.INPAINT_DEPTH = False
    cfg.INPUT.STRONG_AUGS = False
    cfg.INPUT.CAMERA_DROP = False
    cfg.INPUT.MIN_SIZE_TEST_SAMPLING = "choice"
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333
    cfg.MODEL.SUPERVISE_SPARSE = False
    cfg.TEST.EVAL_SPARSE = False
    cfg.MODEL.SEM_SEG_HEAD.NO_SKIP_CONN = False
    cfg.MODEL.KNN = 8
    cfg.INPUT.AUGMENT_3D = False
    cfg.MODEL.FREEZE_BACKBONE = False
    cfg.INPUT.SAMPLE_CHUNK_AUG = False
    cfg.INPUT.VOXELIZE = False
    cfg.INPUT.VOXEL_SIZE = [0.02, 0.04, 0.08, 0.16]
    cfg.DATALOADER.TEST_NUM_WORKERS = 4
    cfg.MODEL.CROSS_VIEW_BACKBONE = False
    cfg.INPUT.ORIGINAL_EVAL = False
    cfg.INPUT.UNIFORM_SAMPLE = False
    cfg.SOLVER.TEST_IMS_PER_BATCH = cfg.SOLVER.IMS_PER_BATCH
    cfg.INPUT.CHUNK_AUG_MAX = 5
    cfg.MODEL.CROSS_VIEW_NUM_LAYERS = [2, 2, 6, 2]
    cfg.USE_GHOST_POINTS = False # featurizes the ghost points and do dot product with them
    cfg.MODEL.DECODER_PANET = False
    cfg.SCANNET_DATA_DIR = "/projects/katefgroup/language_grounding/mask3d_processed/scannet/train_validation_database.yaml"
    cfg.S3DIS_DATA_DIR = "/projects/katefgroup/language_grounding/SEMSEG_100k/s3dis/train_validation_database.yaml"
    cfg.SKIP_CLASSES = None
    cfg.VISUALIZE = False
    cfg.FEATURE_VIS = False
    cfg.VISUALIZE_LOG_DIR = '/projects/katefgroup/language_grounding/visualizations/default'
    cfg.DO_TRILINEAR_INTERPOLATION = True
    cfg.INTERP_NEIGHBORS = 8
    cfg.MODEL.INTERPOLATION_METHOD = "nearest"
    cfg.MODEL.PIXEL_DECODER_PANET = False
    cfg.TEST.EVAL_2D = False
    cfg.DECODER_NUM_LAYERS = 1
    cfg.SAMPLING_STRATEGY = "consecutive"
    cfg.MODEL.DECODER_ONLY_PANET = False
    cfg.MAX_FRAME_NUM = -1
    cfg.USE_SEGMENTS = False
    cfg.SAMPLED_CROSS_ATTENTION = True
    cfg.SAMPLE_SIZES = [800, 3200, 12800]
    cfg.KNN_THRESH = 1e-3
    cfg.MODEL.DECODER_2D = False
    cfg.MODEL.CROSS_VIEW_NUM_LAYERS_ENCODER_TIED = 3
    cfg.DO_FLIPPING = False
    cfg.INPUT.COLOR_AUG = True
    cfg.IGNORE_DEPTH_MAX = -1.0
    cfg.MULTI_TASK_TRAINING = False
    cfg.DATASETS.TRAIN_3D = []
    cfg.DATASETS.TRAIN_2D = []
    cfg.TRAIN_3D = False
    cfg.TRAIN_2D = False
    cfg.FIND_UNUSED_PARAMETERS = False
    cfg.HIGH_RES_SUBSAMPLE = False
    cfg.HIGH_RES_INPUT = False
    cfg.DATASETS.TEST_3D_ONLY = []
    cfg.DATASETS.TEST_2D_ONLY = []
    cfg.EVALUATE_SUBSET = None
    cfg.EVAL_PER_IMAGE = False
    cfg.DO_ELASTIC_DISTORTION = False
    cfg.DEPTH_PREFIX = 'depth_inpainted'
    cfg.TRAIN_SUBSET_3D = None
    cfg.TRAIN_SUBSET_2D = None
    cfg.SOLVER.IMS_PER_BATCH_2D = cfg.SOLVER.IMS_PER_BATCH
    cfg.SOLVER.IMS_PER_BATCH_3D = cfg.SOLVER.IMS_PER_BATCH
    cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS_2D = cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS
    cfg.INPUT.FRAME_LEFT_2D = cfg.INPUT.FRAME_LEFT
    cfg.INPUT.FRAME_RIGHT_2D = cfg.INPUT.FRAME_RIGHT
    cfg.INPUT.SAMPLING_FRAME_NUM_2D = cfg.INPUT.SAMPLING_FRAME_NUM
    cfg.TEST.SUBSAMPLE_DATA = None
    cfg.DATASETS.TEST_SUBSAMPLED = []
    cfg.SKIP_CLASSES_2D = cfg.SKIP_CLASSES
    cfg.VISUALIZE_PRED = False
    cfg.INPUT.MIN_SIZE_TEST_2D = cfg.INPUT.MIN_SIZE_TEST
    cfg.INPUT.MAX_SIZE_TEST_2D = cfg.INPUT.MAX_SIZE_TEST
    cfg.INPUT.IMAGE_SIZE_2D = cfg.INPUT.IMAGE_SIZE
    cfg.MATTERPORT_DATA_DIR = "/projects/katefgroup/language_grounding/mask3d_processed/matterport/train_validation_database.yaml"
    cfg.SCANNET200_DATA_DIR = "/projects/katefgroup/language_grounding/mask3d_processed/scannet200/train_validation_database.yaml"
    cfg.AUGMENT_WITH_3D_SCALE = False
    cfg.REPEAT_S3DIS = False
    cfg.BALANCE_3D_DATASETS = False
    cfg.MODEL.NO_DECODER_PANET = False
    cfg.EXPORT_BENCHMARK_DATA = False
    cfg.MATTERPORT_ALL_CLASSES_TO_21 = False
    cfg.DO_ROT_SCALE = True
    cfg.USE_WANDB = False
    cfg.WANDB_NAME = None
    cfg.VISUALIZE_3D = False
    cfg.USE_MLP_POSITIONAL_ENCODING = False
    cfg.PROB = None
    cfg.EXPORT_BENCHMARK_PATH = None
    
    # Open Vocab configs
    cfg.MODEL.OPEN_VOCAB = False
    cfg.MODEL.LANG_FREEZE_BACKBONE = True
    cfg.MODEL.MAX_SEQ_LEN = 256
    cfg.NON_PARAM_SOFTMAX = False
    cfg.DISABLE_SHUFFLE = True
    cfg.RANDOM_SELECT_CLASSES = False
    cfg.TEXT_ENCODER_TYPE = "roberta"
