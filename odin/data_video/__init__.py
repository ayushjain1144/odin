# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/sukjunhwang/IFC

from .dataset_mapper_scannet import ScannetDatasetMapper
from .build import *

from .datasets import *
from .scannet_3d_eval import Scannet3DEvaluator
from .scannet_3d_eval_semantic import ScannetSemantic3DEvaluator
from .coco_evaluation import COCOEvaluatorMemoryEfficient