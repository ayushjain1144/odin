# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MaskFormer Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
import warnings
warnings.filterwarnings('ignore')

import copy
import itertools
import logging
import os
import gc
import weakref
import time

from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from torch.nn.parallel import DistributedDataParallel
from detectron2.config import get_cfg
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
    AMPTrainer,
    SimpleTrainer
)
from detectron2.evaluation import (
    DatasetEvaluator,
    COCOEvaluator,
    inference_on_dataset,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

# MaskFormer
from odin.data_video.dataset_mapper_coco import COCOInstanceNewBaselineDatasetMapper

from odin import (
    ScannetDatasetMapper,
    Scannet3DEvaluator,
    ScannetSemantic3DEvaluator,
    COCOEvaluatorMemoryEfficient,
    add_maskformer2_video_config,
    add_maskformer2_config,
    build_detection_train_loader,
    build_detection_test_loader,
    get_detection_dataset_dicts,
    build_detection_train_loader_multi_task,
)
from odin.data_video.build import merge_datasets
from odin.global_vars import SCANNET_LIKE_DATASET
from torchinfo import summary

torch.multiprocessing.set_sharing_strategy('file_system')

import ipdb
st = ipdb.set_trace


class OneCycleLr_D2(torch.optim.lr_scheduler.OneCycleLR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def state_dict(self):
        return {"base_lrs": self.base_lrs, "last_epoch": self.last_epoch}
        

def create_ddp_model(model, *, fp16_compression=False, find_unused_parameters=False, **kwargs):
    """
    Create a DistributedDataParallel model if there are >1 processes.

    Args:
        model: a torch.nn.Module
        fp16_compression: add fp16 compression hooks to the ddp object.
            See more at https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook
        kwargs: other arguments of :module:`torch.nn.parallel.DistributedDataParallel`.
    """  # noqa
    if comm.get_world_size() == 1:
        return model
    if "device_ids" not in kwargs:
        kwargs["device_ids"] = [comm.get_local_rank()]
    ddp = DistributedDataParallel(model, **kwargs, find_unused_parameters=find_unused_parameters)
    if fp16_compression:
        from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks

        ddp.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)
    return ddp


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        super(DefaultTrainer, self).__init__()
        # super().__init__()
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        model = create_ddp_model(model, broadcast_buffers=False, find_unused_parameters=cfg.MULTI_TASK_TRAINING or cfg.FIND_UNUSED_PARAMETERS)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    @classmethod
    def build_evaluator(
        cls, cfg, dataset_name,
        output_folder=None, use_2d_evaluators_only=False,
        use_3d_evaluators_only=False,
    ):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            os.makedirs(output_folder, exist_ok=True)
        evaluators = []
        if cfg.TEST.EVAL_3D and cfg.MODEL.DECODER_3D and not use_2d_evaluators_only:
            if cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
                evaluators.append(
                        ScannetSemantic3DEvaluator(
                            dataset_name, 
                            output_dir=output_folder, 
                            eval_sparse=cfg.TEST.EVAL_SPARSE,
                            cfg=cfg
                        ))
            if cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
                evaluators.append(
                        Scannet3DEvaluator(
                            dataset_name,
                            output_dir=output_folder,
                            eval_sparse=cfg.TEST.EVAL_SPARSE,
                            cfg=cfg
                        ))
        if (cfg.TEST.EVAL_2D or cfg.EVAL_PER_IMAGE) and not use_3d_evaluators_only:
            if cfg.INPUT.ORIGINAL_EVAL:
                print("Using original COCO Eval, potentially is RAM hungry")
                evaluators.append(COCOEvaluator(dataset_name, output_dir=output_folder, use_fast_impl=False))
            else:
                evaluators.append(COCOEvaluatorMemoryEfficient(
                    dataset_name, output_dir=output_folder, use_fast_impl=False,
                    per_image_eval=cfg.EVAL_PER_IMAGE, evaluate_subset=cfg.EVALUATE_SUBSET,))
        return evaluators

    @classmethod
    def build_train_loader(cls, cfg):
        if cfg.MULTI_TASK_TRAINING:
            if cfg.TRAIN_3D:
                if len(cfg.DATASETS.TRAIN_3D) > 1:
                    dataset_dicts = [get_detection_dataset_dicts(
                        cfg.DATASETS.TRAIN_3D[i],
                        proposal_files=None,
                    ) for i in range(len(cfg.DATASETS.TRAIN_3D))]
                    mappers = [
                        ScannetDatasetMapper(cfg, is_train=True, dataset_name=dataset_name, dataset_dict=dataset_dict) for dataset_name, dataset_dict in zip(cfg.DATASETS.TRAIN, dataset_dicts)
                    ]
                    dataset_dict_3d = merge_datasets(dataset_dicts, mappers, balance=cfg.BALANCE_3D_DATASETS)
                    mapper_3d = None
                else:
                    dataset_dict_3d = get_detection_dataset_dicts(
                            cfg.DATASETS.TRAIN_3D,
                            proposal_files=None,
                    )
                    mapper_3d = ScannetDatasetMapper(
                        cfg, is_train=True,
                        dataset_name=cfg.DATASETS.TRAIN_3D[0],
                        dataset_dict=dataset_dict_3d
                    )
            else:
                dataset_dict_3d = None
                mapper_3d = None
            
            if cfg.TRAIN_2D:
                    dataset_dict_2d = get_detection_dataset_dicts(
                        cfg.DATASETS.TRAIN_2D,
                        proposal_files=None,
                    )
                    if 'coco' in cfg.DATASETS.TRAIN_2D[0]:
                        mapper_2d = COCOInstanceNewBaselineDatasetMapper(cfg, True, dataset_name=cfg.DATASETS.TRAIN_2D[0])
                    else:
                        mapper_2d = ScannetDatasetMapper(
                            cfg, is_train=True,
                            dataset_name=cfg.DATASETS.TRAIN_2D[0],
                            dataset_dict=dataset_dict_2d,
                            force_decoder_2d=cfg.FORCE_DECODER_3D,
                            frame_left=0,
                            frame_right=0,
                            decoder_3d=False
                        )
            else:
                dataset_dict_2d = None
                mapper_2d = None
            
            return build_detection_train_loader_multi_task(
                    cfg, mapper_3d=mapper_3d, mapper_2d=mapper_2d,
                    dataset_3d=dataset_dict_3d, dataset_2d=dataset_dict_2d
            )
        else:
            dataset_name = cfg.DATASETS.TRAIN[0]
            scannet_like = False
            for scannet_like_dataset in SCANNET_LIKE_DATASET:
                if scannet_like_dataset in dataset_name:
                    scannet_like = True
                    break

            if scannet_like:
                dataset_dict = get_detection_dataset_dicts(
                    dataset_name,
                    proposal_files=None,
                )
                mapper = ScannetDatasetMapper(cfg, is_train=True, dataset_name=dataset_name, dataset_dict=dataset_dict)
                return build_detection_train_loader(cfg, mapper=mapper, dataset=dataset_dict)
            elif 'coco' in dataset_name:
                mapper = COCOInstanceNewBaselineDatasetMapper(cfg, True, dataset_name=dataset_name)
                return build_detection_train_loader(cfg, mapper=mapper)
            else:
                raise NotImplementedError

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        scannet_like = False
        for scannet_like_dataset in SCANNET_LIKE_DATASET:
            if scannet_like_dataset in dataset_name:
                scannet_like = True
                break
        if scannet_like:
            dataset_dict = get_detection_dataset_dicts(
                [dataset_name],
                proposal_files=[
                    cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(dataset_name)]
                ]
                if cfg.MODEL.LOAD_PROPOSALS
                else None,
                subsample_data=cfg.TEST.SUBSAMPLE_DATA if dataset_name in cfg.DATASETS.TEST_SUBSAMPLED else None,
            )
            mapper = ScannetDatasetMapper(
                cfg, is_train=False, dataset_name=dataset_name, dataset_dict=dataset_dict,
                decoder_3d=False if dataset_name in cfg.DATASETS.TEST_2D_ONLY else cfg.MODEL.DECODER_3D,
            )
            return build_detection_test_loader(cfg, mapper=mapper, dataset=dataset_dict)
        elif 'coco' in dataset_name:
            dataset_dict = get_detection_dataset_dicts(
                [dataset_name],
                proposal_files=[
                    cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(dataset_name)]
                ]
                if cfg.MODEL.LOAD_PROPOSALS
                else None,
                subsample_data=cfg.TEST.SUBSAMPLE_DATA if dataset_name in cfg.DATASETS.TEST_SUBSAMPLED else None,
            )
            mapper = COCOInstanceNewBaselineDatasetMapper(cfg, is_train=False, dataset_name=dataset_name)
            return build_detection_test_loader(cfg, mapper=mapper, dataset=dataset_dict)
        else:
            raise NotImplementedError

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        if cfg.SOLVER.LR_SCHEDULER_NAME == "onecyclelr":
            return OneCycleLr_D2(
                optimizer,
                max_lr=cfg.SOLVER.BASE_LR,
                total_steps=cfg.SOLVER.MAX_ITER,
            )
        else:
            return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()

        print(summary(model))

        panet_resnet_layers = ['cross_view_attn', 'res_to_trans', 'trans_to_res']
        panet_swin_layers = ['cross_view_attn', 'cross_layer_norm', 'res_to_trans', 'trans_to_res']

        if cfg.MODEL.BACKBONE.NAME == "build_resnet_backbone":
            backbone_panet_layers = panet_resnet_layers
        elif cfg.MODEL.BACKBONE.NAME == "D2SwinTransformer":
            backbone_panet_layers = panet_swin_layers
        else:
            raise NotImplementedError


        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name :
                    # panet layers are initialize from scratch so use default lr
                    panet_found = False
                    for panet_name in backbone_panet_layers:
                        if panet_name in module_name:
                            hyperparams["lr"] = hyperparams["lr"]
                            panet_found = True
                            break

                    if not panet_found:
                        hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.
        Returns:
            dict: a dict of result metrics
        """
        from torch.cuda.amp import autocast
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(
                        cfg, dataset_name, use_2d_evaluators_only=dataset_name in cfg.DATASETS.TEST_2D_ONLY if cfg.MULTI_TASK_TRAINING else False,
                        use_3d_evaluators_only=dataset_name in cfg.DATASETS.TEST_3D_ONLY if cfg.MULTI_TASK_TRAINING else False,)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            with autocast():
                results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i

            
        gc.collect()
        torch.cuda.empty_cache()
        
        if not cfg.MULTI_TASK_TRAINING:
            #format for writer
            if len(results) == 1:
                results_structured = list(results.values())[0]
            elif len(results) == 2:
                # find a better way than hard-coding here
                results_val = results[cfg.DATASETS.TEST[0]].copy()
                suffix = '_full' if 'single' in cfg.DATASETS.TEST[0] else ''
                suffix += f'_{dataset_name.split("_")[0]}'
                results_val = {f'val{suffix}'+k: v for k, v in results_val.items()}
                # st()
                try:
                    if cfg.EVAL_PER_IMAGE:
                        results_val[f'val_{dataset_name.split("_")[0]}segm'] = results_val[f'val{suffix}segm']
                        del results_val[f'val{suffix}segm']
                except:
                    print("Error in Logging")
                    print(results_val.keys(), print(f'val{suffix}segm'))
                results_train = results[cfg.DATASETS.TEST[1]].copy()
                results_train = {f'train{suffix}'+k: v for k, v in results_train.items()}
                try:
                    if cfg.EVAL_PER_IMAGE:
                        results_train[f'train_{dataset_name.split("_")[0]}segm'] = results_train[f'train{suffix}segm']
                        del results_train[f'train{suffix}segm']
                except:
                    print(results_train.keys(), print(f'train{suffix}segm'))
                results_structured = {}
                results_structured.update(results_train)
                results_structured.update(results_val)

            else:
                for dataset_name in cfg.DATASETS.TEST:
                    results_structured = {}
                    suffix = 'train_full' if 'train_eval' in dataset_name else 'val_full'
                    results_val = results[dataset_name].copy()
                    results_val = {f'{suffix}_{dataset_name.split("_")[0]}'+k: v for k, v in results_val.items()}
                    results_structured.update(results_val)

        else:
            results_structured = {}
            for dataset_name in cfg.DATASETS.TEST_3D_ONLY:
                if dataset_name in results:
                    suffix = 'train_full' if 'train_eval' in dataset_name else 'val_full'
                    suffix += f'_{dataset_name.split("_")[0]}'
                    results_val = results[dataset_name].copy()
                    results_val = {f'{suffix}'+k: v for k, v in results_val.items()}
                    results_structured.update(results_val)
                
            for dataset_name in cfg.DATASETS.TEST_2D_ONLY:
                if dataset_name in results:
                    suffix = 'train' if 'train_eval' in dataset_name else 'val'
                    suffix += f'_{dataset_name.split("_")[0]}'
                    results_val = results[dataset_name].copy()
                    results_val = {f'{suffix}'+k: v for k, v in results_val.items()}
                    results_structured.update(results_val)
        return results_structured

    def run_step(self):
        """
        Implement the AMP training logic.
        """
        self._trainer.iter = self.iter
        
        assert self._trainer.model.training, "[AMPTrainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[AMPTrainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast
        
        assert self.cfg.SOLVER.AMP.ENABLED

        start = time.perf_counter()
        data = next(self._trainer._data_loader_iter)
        data_time = time.perf_counter() - start

        with autocast(dtype=self._trainer.precision):
            loss_dict = self._trainer.model(data)
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                loss_custom = None
                if 'loss_3d' in loss_dict or 'loss_2d' in loss_dict:
                    loss_name = 'loss_3d' if 'loss_3d' in loss_dict else 'loss_2d'
                    loss_custom = loss_dict[loss_name]
                    loss_dict.pop('loss_3d', None)
                    loss_dict.pop('loss_2d', None)
                losses = sum(loss_dict.values())
                
                if loss_custom is not None:
                    loss_dict[loss_name] = loss_custom

        self._trainer.optimizer.zero_grad()
        self._trainer.grad_scaler.scale(losses).backward()
        
        self._trainer.after_backward()

        self._trainer._write_metrics(loss_dict, data_time)

        self._trainer.grad_scaler.step(self.optimizer)
        self._trainer.grad_scaler.update()  
        
def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_maskformer2_video_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="odin")
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            raise NotImplementedError
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    
    # this is needed to prevent memory leak in conv2d layers
    # see: https://github.com/pytorch/pytorch/issues/98688#issuecomment-1869290827
    os.environ['TORCH_CUDNN_V8_API_DISABLED'] = '1' 
    
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
