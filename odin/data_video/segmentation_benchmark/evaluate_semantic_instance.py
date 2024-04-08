import math
import os, sys, argparse
import inspect
from copy import deepcopy
from uuid import uuid4

import torch

try:
    import numpy as np
except:
    print("Failed to import numpy package.")
    sys.exit(-1)

from scipy import stats
from prettytable import PrettyTable

from ..segmentation_benchmark import util
from ..segmentation_benchmark import util_3d
from odin.global_vars import AI2THOR_CLASS_ID_MULTIPLIER, \
    AI2THOR_INSTANCE_CLASS_NAMES, SCANNET_INSTANCE_CLASS_NAMES, \
    S3DIS_NAME_MAP, \
    CLASS_NAME_DICT, \
    SCANNET_INSTANCE_CLASS_NAMES_200, ALFRED_CLASS_NAMES, \
    ALFRED_CLASS_ID_MULTIPLIER, MATTERPORT_CLASS_LABELS 


import ipdb
st = ipdb.set_trace

opt = {}
opt['overlaps']             = np.append(np.arange(0.5,0.95,0.05), 0.25)
# minimum region size for evaluation [verts]
opt['min_region_sizes']     = np.array( [ 100 ] ) 
# distance thresholds [m]
opt['distance_threshes']    = np.array( [  float('inf') ] )
# distance confidences
opt['distance_confs']       = np.array( [ -float('inf') ] )


class Scannet_Evaluator():
    def __init__(self, dataset_name, evaluate_subset=None):
        self.dataset_name = dataset_name
        self.EVALUATE_SUBSET = evaluate_subset
        if 'ai2thor' in dataset_name:
            self.multiplier = AI2THOR_CLASS_ID_MULTIPLIER
        elif 'alfred' in dataset_name:
            self.multiplier = ALFRED_CLASS_ID_MULTIPLIER
        else:
            self.multiplier = 1000

        if 'ai2thor' in dataset_name:
            self.CLASS_LABELS = AI2THOR_INSTANCE_CLASS_NAMES
        elif 's3dis' in dataset_name:
            self.CLASS_LABELS = list(S3DIS_NAME_MAP.values())
        elif 'alfred' in dataset_name:
            self.CLASS_LABELS = ALFRED_CLASS_NAMES
        elif 'scannet200' in dataset_name:
                self.CLASS_LABELS = SCANNET_INSTANCE_CLASS_NAMES_200
        elif 'matterport' in dataset_name:
            self.CLASS_LABELS = MATTERPORT_CLASS_LABELS
        else:
            self.CLASS_LABELS = SCANNET_INSTANCE_CLASS_NAMES
        
        self.VALID_CLASS_IDS = np.arange(1, len(self.CLASS_LABELS) + 1)

        self.ID_TO_LABEL = {}
        self.LABEL_TO_ID = {}
        for i in range(len(self.VALID_CLASS_IDS)):
            self.LABEL_TO_ID[self.CLASS_LABELS[i]] = self.VALID_CLASS_IDS[i]
            self.ID_TO_LABEL[self.VALID_CLASS_IDS[i]] = self.CLASS_LABELS[i]

    def evaluate_matches(self, matches):
        overlaps = opt['overlaps']
        min_region_sizes = [ opt['min_region_sizes'][0] ]
        dist_threshes = [ opt['distance_threshes'][0] ]
        dist_confs = [ opt['distance_confs'][0] ]

        # results: class x overlap
        # ap = np.zeros( (len(dist_threshes) , len(CLASS_LABELS) , len(overlaps)) , float )
        has_gt_full = np.zeros((len(dist_threshes), len(overlaps), len(self.CLASS_LABELS)), bool)
        has_pred_full = np.zeros((len(dist_threshes), len(overlaps), len(self.CLASS_LABELS)), bool)
        y_score_full = np.empty((len(dist_threshes), len(overlaps), len(self.CLASS_LABELS)), object)
        y_true_full = np.empty((len(dist_threshes), len(overlaps), len(self.CLASS_LABELS)), object)
        hard_false_negatives_full = np.zeros((len(dist_threshes), len(overlaps), len(self.CLASS_LABELS)), int)

        for di, (min_region_size, distance_thresh, distance_conf) in enumerate(zip(min_region_sizes, dist_threshes, dist_confs)):
            for oi, overlap_th in enumerate(overlaps):
                pred_visited = {}
                for m in matches:
                    for p in matches[m]['pred']:
                        for label_name in self.CLASS_LABELS:
                            for p in matches[m]['pred'][label_name]:
                                if 'uuid' in p:
                                    pred_visited[p['uuid']] = False
                for li, label_name in enumerate(self.CLASS_LABELS):
                    y_true = np.empty(0)
                    y_score = np.empty(0)
                    hard_false_negatives = 0
                    has_gt = False
                    has_pred = False
                    for m in matches:
                        pred_instances = matches[m]['pred'][label_name]
                        gt_instances = matches[m]['gt'][label_name]
                        # filter groups in ground truth
                        # NOTE: this can have problems
                        # 
                        gt_instances = [ gt for gt in gt_instances if gt['instance_id']>=self.multiplier and gt['vert_count']>=min_region_size and gt['med_dist']<=distance_thresh and gt['dist_conf']>=distance_conf ]
                        if gt_instances:
                            has_gt = True
                        if pred_instances:
                            has_pred = True

                        cur_true  = np.ones ( len(gt_instances) )
                        cur_score = np.ones ( len(gt_instances) ) * (-float("inf"))
                        cur_match = np.zeros( len(gt_instances) , dtype=bool )
                        # collect matches
                        for (gti,gt) in enumerate(gt_instances):
                            found_match = False
                            num_pred = len(gt['matched_pred'])
                            for pred in gt['matched_pred']:
                                # greedy assignments
                                if pred_visited[pred['uuid']]:
                                    continue
                                overlap = float(pred['intersection']) / (gt['vert_count']+pred['vert_count']-pred['intersection'])
                                if overlap > overlap_th:
                                    confidence = pred['confidence']
                                    # if already have a prediction for this gt,
                                    # the prediction with the lower score is automatically a false positive
                                    if cur_match[gti]:
                                        max_score = max( cur_score[gti] , confidence )
                                        min_score = min( cur_score[gti] , confidence )
                                        cur_score[gti] = max_score
                                        # append false positive
                                        cur_true  = np.append(cur_true,0)
                                        cur_score = np.append(cur_score,min_score)
                                        cur_match = np.append(cur_match,True)
                                    # otherwise set score
                                    else:
                                        found_match = True
                                        cur_match[gti] = True
                                        cur_score[gti] = confidence
                                        pred_visited[pred['uuid']] = True
                            if not found_match:
                                hard_false_negatives += 1
                        # remove non-matched ground truth instances
                        cur_true  = cur_true [ cur_match==True ]
                        cur_score = cur_score[ cur_match==True ]

                        # collect non-matched predictions as false positive
                        for pred in pred_instances:
                            found_gt = False
                            for gt in pred['matched_gt']:
                                overlap = float(gt['intersection']) / (gt['vert_count']+pred['vert_count']-gt['intersection'])
                                if overlap > overlap_th:
                                    found_gt = True
                                    break
                            if not found_gt:
                                num_ignore = pred['void_intersection']
                                for gt in pred['matched_gt']:
                                    # group?
                                    if gt['instance_id'] < self.multiplier:
                                        num_ignore += gt['intersection']
                                    # small ground truth instances
                                    if gt['vert_count'] < min_region_size or gt['med_dist']>distance_thresh or gt['dist_conf']<distance_conf:
                                        num_ignore += gt['intersection']
                                proportion_ignore = float(num_ignore)/pred['vert_count']
                                # if not ignored append false positive
                                if proportion_ignore <= overlap_th:
                                    cur_true = np.append(cur_true,0)
                                    confidence = pred["confidence"]
                                    cur_score = np.append(cur_score,confidence)

                        # append to overall results
                        y_true  = np.append(y_true,cur_true)
                        y_score = np.append(y_score,cur_score)

                
                    has_gt_full[di, oi, li] = has_gt
                    has_pred_full[di, oi, li] = has_pred
                    y_true_full[di, oi, li] = y_true
                    y_score_full[di, oi, li] = y_score
                    hard_false_negatives_full[di, oi, li] = hard_false_negatives
        return has_gt_full, has_pred_full, y_true_full, y_score_full, hard_false_negatives_full

    def compute_ap(self, has_gt_full, has_pred_full, y_score_full, y_true_full, hard_false_negatives_full):
        # compute average precision
        overlaps = opt['overlaps']
        min_region_sizes = [ opt['min_region_sizes'][0] ]
        dist_threshes = [ opt['distance_threshes'][0] ]
        dist_confs = [ opt['distance_confs'][0] ]

        ap = np.zeros( (len(dist_threshes) , len(self.CLASS_LABELS) , len(overlaps)) , float )

        for di, (min_region_size, distance_thresh, distance_conf) in enumerate(zip(min_region_sizes, dist_threshes, dist_confs)):
            for oi, overlap_th in enumerate(overlaps):
                for li, label_name in enumerate(self.CLASS_LABELS):
                    has_gt = has_gt_full[di, oi, li]
                    has_pred = has_pred_full[di, oi, li]
                    y_score = y_score_full[di, oi, li]
                    y_true = y_true_full[di, oi, li]
                    hard_false_negatives = hard_false_negatives_full[di, oi, li]

                    if has_gt and has_pred:
                        # sorting and cumsum
                        score_arg_sort      = np.argsort(y_score)
                        y_score_sorted      = y_score[score_arg_sort]
                        y_true_sorted       = y_true[score_arg_sort]
                        y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                        # unique thresholds
                        (thresholds,unique_indices) = np.unique( y_score_sorted , return_index=True )
                        num_prec_recall = len(unique_indices) + 1

                        # prepare precision recall
                        num_examples      = len(y_score_sorted)
                        # https://github.com/ScanNet/ScanNet/pull/26
                        # all predictions are non-matched but also all of them are ignored and not counted as FP
                        # y_true_sorted_cumsum is empty
                        # num_true_examples = y_true_sorted_cumsum[-1]
                        num_true_examples = y_true_sorted_cumsum[-1] if len(y_true_sorted_cumsum) > 0 else 0
                        precision         = np.zeros(num_prec_recall)
                        recall            = np.zeros(num_prec_recall)

                        # deal with the first point
                        y_true_sorted_cumsum = np.append( y_true_sorted_cumsum , 0 )
                        # deal with remaining
                        # st()
                        for idx_res,idx_scores in enumerate(unique_indices):
                            cumsum = y_true_sorted_cumsum[idx_scores-1]
                            tp = num_true_examples - cumsum
                            fp = num_examples      - idx_scores - tp
                            fn = cumsum + hard_false_negatives
                            p  = float(tp)/(tp+fp)
                            r  = float(tp)/(tp+fn)
                            precision[idx_res] = p
                            recall   [idx_res] = r

                        # first point in curve is artificial
                        precision[-1] = 1.
                        recall   [-1] = 0.

                        # compute average of precision-recall curve
                        recall_for_conv = np.copy(recall)
                        recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                        recall_for_conv = np.append(recall_for_conv, 0.)

                        stepWidths = np.convolve(recall_for_conv,[-0.5,0,0.5],'valid')
                        # integrate is now simply a dot product
                        ap_current = np.dot(precision, stepWidths)

                    elif has_gt:
                        ap_current = 0.0
                    else:
                        ap_current = float('nan')
                    ap[di,li,oi] = ap_current
        return ap

    def compute_averages(self, aps):
        d_inf = 0
        o50   = np.where(np.isclose(opt['overlaps'],0.5))
        o25   = np.where(np.isclose(opt['overlaps'],0.25))
        oAllBut25  = np.where(np.logical_not(np.isclose(opt['overlaps'],0.25)))
        avg_dict = {}
        #avg_dict['all_ap']     = np.nanmean(aps[ d_inf,:,:  ])
        avg_dict['all_ap']     = np.nanmean(aps[ d_inf,:,oAllBut25])
        avg_dict['all_ap_50%'] = np.nanmean(aps[ d_inf,:,o50])
        avg_dict['all_ap_25%'] = np.nanmean(aps[ d_inf,:,o25])
        
        avg_dict["classes"]  = {}
        for (li,label_name) in enumerate(self.CLASS_LABELS):
            avg_dict["classes"][label_name]             = {}
            avg_dict["classes"][label_name]["ap"]       = np.average(aps[ d_inf,li,oAllBut25])
            avg_dict["classes"][label_name]["ap50%"]    = np.average(aps[ d_inf,li,o50])
            avg_dict["classes"][label_name]["ap25%"]    = np.average(aps[ d_inf,li,o25])
        
        if self.EVALUATE_SUBSET is not None:
            for name  in self.EVALUATE_SUBSET:
                class_ids = CLASS_NAME_DICT[name]
                # remove last two class ids because they are bg
                remove_ids = None
                if 'scannet200' in self.dataset_name and 'head' in name:
                    remove_ids = [199, 200]
                elif 'ai2thor' in self.dataset_name:
                    remove_ids = [116, 117]
                    
                if remove_ids is not None:
                    for id in remove_ids:
                        if id in class_ids:
                            class_ids.remove(id)

                class_ids  = np.array(class_ids) - 1
                avg_dict[f'{name}_ap'] = np.nanmean(aps[ d_inf,:,oAllBut25][..., class_ids])
                avg_dict[f'{name}_ap_50%'] = np.nanmean(aps[ d_inf,:,o50][..., class_ids])
                avg_dict[f'{name}_ap_25%'] = np.nanmean(aps[ d_inf,:,o25][..., class_ids])
        return avg_dict

    def make_pred_info(self, pred: dict):
        # pred = {'pred_scores' = 100, 'pred_classes' = 100 'pred_masks' = Nx100}
        pred_info = {}
        assert(pred['pred_classes'].shape[0] == pred['pred_scores'].shape[0] == pred['pred_masks'].shape[1])
        for i in range(len(pred['pred_classes'])):
            info = {}
            info["label_id"] = pred['pred_classes'][i]
            info["conf"] = pred['pred_scores'][i]
            info["mask"] = pred['pred_masks'][:,i]
            pred_info[uuid4()] = info # we later need to identify these objects
        return pred_info

    def assign_instances_for_scan(self, pred: dict, gt: dict):
        pred_info = self.make_pred_info(pred)
        gt_ids = gt['labels']

        # get gt instances
        # associate
        gt_instances = util_3d.get_instances(gt_ids, self.VALID_CLASS_IDS, self.CLASS_LABELS, self.ID_TO_LABEL, self.multiplier)
        gt2pred = deepcopy(gt_instances)
        for label in gt2pred:
            for gt in gt2pred[label]:
                gt['matched_pred'] = []
        pred2gt = {}
        for label in self.CLASS_LABELS:
            pred2gt[label] = []
        num_pred_instances = 0
        # mask of void labels in the groundtruth
        bool_void = np.logical_not(np.in1d(gt_ids//self.multiplier, self.VALID_CLASS_IDS))
        # go thru all prediction masks
        for uuid in pred_info:
            label_id = int(pred_info[uuid]['label_id'])
            conf = pred_info[uuid]['conf']
            if not label_id in self.ID_TO_LABEL:
                continue
            label_name = self.ID_TO_LABEL[label_id]
            # read the mask
            pred_mask = pred_info[uuid]['mask']
            assert(len(pred_mask) == len(gt_ids)), f"pred_mask: {len(pred_mask)}, gt_ids: {len(gt_ids)}"
            # convert to binary
            pred_mask = np.not_equal(pred_mask, 0)
            num = np.count_nonzero(pred_mask)
            if num < opt['min_region_sizes'][0]:
                continue  # skip if empty

            pred_instance = {}
            pred_instance['uuid'] = uuid
            pred_instance['pred_id'] = num_pred_instances
            pred_instance['label_id'] = label_id
            pred_instance['vert_count'] = num
            pred_instance['confidence'] = conf
            pred_instance['void_intersection'] = np.count_nonzero(np.logical_and(bool_void, pred_mask))

            # matched gt instances
            matched_gt = []
            # go thru all gt instances with matching label
            for (gt_num, gt_inst) in enumerate(gt2pred[label_name]):
                intersection = np.count_nonzero(np.logical_and(gt_ids == gt_inst['instance_id'], pred_mask))
                if intersection > 0:
                    gt_copy = gt_inst.copy()
                    pred_copy = pred_instance.copy()
                    gt_copy['intersection']   = intersection
                    pred_copy['intersection'] = intersection
                    matched_gt.append(gt_copy)
                    gt2pred[label_name][gt_num]['matched_pred'].append(pred_copy)

            pred_instance['matched_gt'] = matched_gt
            num_pred_instances += 1
            pred2gt[label_name].append(pred_instance)

        return gt2pred, pred2gt


    def print_results(self, avgs, logger):
        sep     = ""
        col1    = ":"
        lineLen = 64

        logger.info("")
        logger.info("#"*lineLen)
        line  = ""
        line += "{:<15}".format("what"      ) + sep + col1
        line += "{:>15}".format("AP"        ) + sep
        line += "{:>15}".format("AP_50%"    ) + sep
        line += "{:>15}".format("AP_25%"    ) + sep
        logger.info(line)
        logger.info("#"*lineLen)

        for (li,label_name) in enumerate(self.CLASS_LABELS):
            ap_avg  = avgs["classes"][label_name]["ap"]
            ap_50o  = avgs["classes"][label_name]["ap50%"]
            ap_25o  = avgs["classes"][label_name]["ap25%"]
            line  = "{:<15}".format(label_name) + sep + col1
            line += sep + "{:>15.3f}".format(ap_avg ) + sep
            line += sep + "{:>15.3f}".format(ap_50o ) + sep
            line += sep + "{:>15.3f}".format(ap_25o ) + sep
            logger.info(line)

        all_ap_avg  = avgs["all_ap"]
        all_ap_50o  = avgs["all_ap_50%"]
        all_ap_25o  = avgs["all_ap_25%"]

        logger.info("-"*lineLen)
        line  = "{:<15}".format("average") + sep + col1
        line += "{:>15.3f}".format(all_ap_avg)  + sep
        line += "{:>15.3f}".format(all_ap_50o)  + sep
        line += "{:>15.3f}".format(all_ap_25o)  + sep
        
        logger.info(line)
        
        if self.EVALUATE_SUBSET:
            for name in self.EVALUATE_SUBSET:
                logger.info("-"*lineLen)
                line = ""
                line  += "{:<15}".format(f"average {name}") + sep + col1
                subset_ap_avg  = avgs[f"{name}_ap"]
                subset_ap_50o  = avgs[f"{name}_ap_50%"]
                subset_ap_25o  = avgs[f"{name}_ap_25%"]
                line += "{:>15.3f}".format(subset_ap_avg)  + sep
                line += "{:>15.3f}".format(subset_ap_50o)  + sep
                line += "{:>15.3f}".format(subset_ap_25o)  + sep
                logger.info(line)
        logger.info("")

    def evaluate(self, preds: dict, gts: dict):
        matches = {}
        for i,(k,v) in enumerate(preds.items()):
            matches_key = i
            # assign gt to predictions
            gt2pred, pred2gt = self.assign_instances_for_scan(v, gts[i])
            matches[matches_key] = {}
            matches[matches_key]['gt'] = gt2pred
            matches[matches_key]['pred'] = pred2gt
        ap_scores = self.evaluate_matches(matches)
        return ap_scores
