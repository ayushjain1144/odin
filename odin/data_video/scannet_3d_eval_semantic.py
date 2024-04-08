import copy
import itertools
import logging
import numpy as np
import torch

import detectron2.utils.comm as comm

from ..utils.util_3d import convert_3d_to_2d_dict_format_semantic
from prettytable import PrettyTable

from odin.global_vars import (
    AI2THOR_CLASS_ID_MULTIPLIER, NAME_MAP20, AI2THOR_CLASS_NAMES,
    CLASS_NAME_DICT, SCANNET_CLASS_LABELS_200, ALFRED_CLASS_ID_MULTIPLIER, 
    ALFRED_CLASS_NAMES, S3DIS_NAME_MAP, MATTERPORT_CLASS_LABELS
)
from .scannet_3d_eval import Scannet3DEvaluator
from odin.utils.misc import all_gather, is_dist_avail_and_initialized


import ipdb
st = ipdb.set_trace


def intersection_and_union_gpu(output, target, k, ignore_index=-1):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3, 4])
    assert output.shape == target.shape, "{} vs {}".format(output.shape, target.shape)
    output = output.reshape(-1)
    target = target.reshape(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection.float(), bins=k, min=0, max=k - 1)
    area_output = torch.histc(output.float(), bins=k, min=0, max=k - 1)
    area_target = torch.histc(target.float(), bins=k, min=0, max=k - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


class ScannetSemantic3DEvaluator(Scannet3DEvaluator):
    def __init__(
        self,
        dataset_name,
        eval_sparse=False,
        distributed=True,
        output_dir=None,
        cfg=None
    ):
        self._logger = logging.getLogger(__name__)
        self._distributed = distributed
        self._output_dir = output_dir
        self.eval_sparse = eval_sparse
        self.cfg = cfg

        if 'ai2thor' in dataset_name:
            self.multiplier = AI2THOR_CLASS_ID_MULTIPLIER
        elif 'alfred' in dataset_name:
            self.multiplier = ALFRED_CLASS_ID_MULTIPLIER
        else:
            self.multiplier = 1000

        if 'ai2thor' in dataset_name:
            self.CLASS_LABELS = AI2THOR_CLASS_NAMES
        elif 's3dis' in dataset_name:
            self.CLASS_LABELS =  list(S3DIS_NAME_MAP.values())
        elif 'alfred' in dataset_name:
            self.CLASS_LABELS = ALFRED_CLASS_NAMES
        elif 'scannet200' in dataset_name:
            self.CLASS_LABELS = SCANNET_CLASS_LABELS_200
        elif 'matterport' in dataset_name:
            self.CLASS_LABELS = MATTERPORT_CLASS_LABELS
        else:
            self.CLASS_LABELS =  list(NAME_MAP20.values())

        self._cpu_device = torch.device("cpu")
 
    def reset(self):
        self._predictions = []
        self._ground_truths = [] # ground-truth from dataloader
        self._batch_ap_scores = []

        self.intersection = []
        self.union = []
        self.target = []
    
    def parse_preds(self, output):
        pred = output['semantic_3d'].cpu()
        return pred

    def process(self, inputs, outputs):
        if outputs is None:
            return
        for input, output in zip(inputs, outputs):
            # output stores the processed gt in case of ghost point experiment
            if self.cfg.USE_GHOST_POINTS:
                gt_parse_dict = output['instances_3d']
            else:
                gt_parse_dict = input
            
            segment = self.parse_gt(gt_parse_dict)['labels']
            segment = segment // self.multiplier - 1
            pred = self.parse_preds(output)

            intersection, union, target = intersection_and_union_gpu(pred, segment, len(self.CLASS_LABELS))
            self.intersection.append(intersection.cpu().numpy())
            self.union.append(union.cpu().numpy())
            self.target.append(target.cpu().numpy())
    
    def evaluate(self):
        """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset
        """

        if is_dist_avail_and_initialized():
            intersection = all_gather(self.intersection)
            intersection = list(itertools.chain(*intersection))
            
            union = all_gather(self.union)
            union = list(itertools.chain(*union))

            target = all_gather(self.target)
            target = list(itertools.chain(*target))
            
            # clear the memory
            self.intersection = []
            self.union = []
            self.target = []
            

            if not comm.is_main_process():
                return {}
        else:
            # sum across batch, ref total from evaluator pointcept
            intersection = self.intersection
            union = self.union
            target = self.target
            
            # clear the memory
            self.intersection = []
            self.union = []
            self.target = []

        intersection = np.array(intersection).sum(axis=0)
        union = np.array(union).sum(axis=0)
        target = np.array(target).sum(axis=0)
        
        valid_classes = target != 0
        iou_class = intersection / (union + 1e-10)
        acc_class = intersection / (target + 1e-10)
        m_iou = np.nanmean(iou_class[valid_classes])
        m_acc = np.nanmean(acc_class[valid_classes])
        
        all_acc = sum(intersection[valid_classes]) / (sum(target[valid_classes]) + 1e-10)

        self._logger.info('Val result: mIoU/mAcc/allAcc {:.3f}/{:.3f}/{:.3f}.'.format(
            m_iou, m_acc, all_acc))
        
        data_dict = {
            'M_IOU': m_iou,
            'M_ACC': m_acc,
            'ALL_ACC': all_acc,
            'class_labels': self.CLASS_LABELS,
            'iou_class': iou_class,
            'acc_class': acc_class
        }
                
        if self.cfg.EVALUATE_SUBSET is not None:
            for name in self.cfg.EVALUATE_SUBSET:
                class_ids = CLASS_NAME_DICT[name]
                class_ids = np.array(class_ids) - 1
                class_iou = iou_class[class_ids]
                class_acc = acc_class[class_ids]
                class_valids = valid_classes[class_ids]
                class_miou = np.nanmean(class_iou[class_valids])
                class_macc = np.nanmean(class_acc[class_valids])
                class_allacc = sum(intersection[class_ids][class_valids]) / (sum(target[class_ids][class_valids]) + 1e-10)
                self._logger.info('Val result: mIoU/mAcc/allAcc for {}: {:.3f}/{:.3f}/{:.3f}.'.format(
                    name, class_miou, class_macc, class_allacc))
                
                data_dict[f'{name}_M_IOU'] = class_miou
                data_dict[f'{name}_M_ACC'] = class_macc
                data_dict[f'{name}_ALL_ACC'] = class_allacc

        table = PrettyTable()
        table.field_names = ["Class", "IOU", "Accuracy"]
        for i in range(len(self.CLASS_LABELS)):
            table.add_row(
                [self.CLASS_LABELS[i], "{:.3f}".format(iou_class[i] if valid_classes[i] else np.nan),
                "{:.3f}".format(acc_class[i] if valid_classes[i] else np.nan)])
        print(table)

        self._results = convert_3d_to_2d_dict_format_semantic(data_dict)

        return copy.deepcopy(self._results)
    
