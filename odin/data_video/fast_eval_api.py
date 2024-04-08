# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
import numpy as np
import time
from pycocotools.cocoeval import COCOeval

from detectron2 import _C

logger = logging.getLogger(__name__)


import ipdb
st = ipdb.set_trace


class COCOeval_opt(COCOeval):
    """
    This is a slightly modified version of the original COCO API, where the functions evaluateImg()
    and accumulate() are implemented in C++ to speedup evaluation
    """
    def evaluate(self):
        """
        Run per image evaluation on given images and store results in self.evalImgs_cpp, a
        datastructure that isn't readable from Python but is used by a c++ implementation of
        accumulate().  Unlike the original COCO PythonAPI, we don't populate the datastructure
        self.evalImgs because this datastructure is a computational bottleneck.
        :return: None
        """
        tic = time.time()

        p = self.params
        # add backward compatibility if useSegm is specified in params
        if p.useSegm is not None:
            p.iouType = "segm" if p.useSegm == 1 else "bbox"
        # logger.info("Evaluate annotation type *{}*".format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params = p

        self._prepare()  # bottleneck

        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        if p.iouType == "segm" or p.iouType == "bbox":
            computeIoU = self.computeIoU
        elif p.iouType == "keypoints":
            computeIoU = self.computeOks
        self.ious = {
            (imgId, catId): computeIoU(imgId, catId) for imgId in p.imgIds for catId in catIds
        }  # bottleneck

        maxDet = p.maxDets[-1]

        # <<<< Beginning of code differences with original COCO API
        def convert_instances_to_cpp(instances, is_det=False):
            # Convert annotations for a list of instances in an image to a format that's fast
            # to access in C++
            instances_cpp = []
            for instance in instances:
                instance_cpp = _C.InstanceAnnotation(
                    int(instance["id"]),
                    instance["score"] if is_det else instance.get("score", 0.0),
                    instance["area"],
                    bool(instance.get("iscrowd", 0)),
                    bool(instance.get("ignore", 0)),
                )
                instances_cpp.append(instance_cpp)
            return instances_cpp

        # Convert GT annotations, detections, and IOUs to a format that's fast to access in C++
        ground_truth_instances = [
            [convert_instances_to_cpp(self._gts[imgId, catId]) for catId in p.catIds]
            for imgId in p.imgIds
        ]
        detected_instances = [
            [convert_instances_to_cpp(self._dts[imgId, catId], is_det=True) for catId in p.catIds]
            for imgId in p.imgIds
        ]
        ious = [[self.ious[imgId, catId] for catId in catIds] for imgId in p.imgIds]

        if not p.useCats:
            # For each image, flatten per-category lists into a single list
            ground_truth_instances = [[[o for c in i for o in c]] for i in ground_truth_instances]
            detected_instances = [[[o for c in i for o in c]] for i in detected_instances]

        # Call C++ implementation of self.evaluateImgs()
        _evalImgs_cpp = _C.COCOevalEvaluateImages(
            p.areaRng, maxDet, p.iouThrs, ious, ground_truth_instances, detected_instances
        )
        self._evalImgs = None

        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        # logger.info("COCOeval_opt.evaluate() finished in {:0.2f} seconds.".format(toc - tic))
        # >>>> End of code differences with original COCO API
        return _evalImgs_cpp

def accumulate(_evalImgs_cpp, _paramsEval):
    """
    Accumulate per image evaluation results and store the result in self.eval.  Does not
    support changing parameter settings from those used by self.evaluate()
    """
    logger.info("Accumulating evaluation results...")
    tic = time.time()

    eval = _C.COCOevalAccumulate(_paramsEval, _evalImgs_cpp)

    # recall is num_iou_thresholds X num_categories X num_area_ranges X num_max_detections
    eval["recall"] = np.array(eval["recall"]).reshape(
        eval["counts"][:1] + eval["counts"][2:]
    )

    # precision and scores are num_iou_thresholds X num_recall_thresholds X num_categories X
    # num_area_ranges X num_max_detections
    eval["precision"] = np.array(eval["precision"]).reshape(eval["counts"])
    eval["scores"] = np.array(eval["scores"]).reshape(eval["counts"])
    toc = time.time()
    logger.info("COCOeval_opt.accumulate() finished in {:0.2f} seconds.".format(toc - tic))
    return eval

def summarize(params, eval):
    '''
    Compute and display summary metrics for evaluation results.
    Note this functin can *only* be applied on the default parameter setting
    '''
    def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
        p = params
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap==1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:,:,:,aind,mind]
        else:
            # dimension of recall: [TxKxAxM]
            s = eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:,:,aind,mind]
        if len(s[s>-1])==0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s>-1])
        print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
        return mean_s
    def _summarizeDets():
        stats = np.zeros((12,))
        stats[0] = _summarize(1)
        stats[1] = _summarize(1, iouThr=.5, maxDets=params.maxDets[2])
        stats[2] = _summarize(1, iouThr=.75, maxDets=params.maxDets[2])
        stats[3] = _summarize(1, areaRng='small', maxDets=params.maxDets[2])
        stats[4] = _summarize(1, areaRng='medium', maxDets=params.maxDets[2])
        stats[5] = _summarize(1, areaRng='large', maxDets=params.maxDets[2])
        stats[6] = _summarize(0, maxDets=params.maxDets[0])
        stats[7] = _summarize(0, maxDets=params.maxDets[1])
        stats[8] = _summarize(0, maxDets=params.maxDets[2])
        stats[9] = _summarize(0, areaRng='small', maxDets=params.maxDets[2])
        stats[10] = _summarize(0, areaRng='medium', maxDets=params.maxDets[2])
        stats[11] = _summarize(0, areaRng='large', maxDets=params.maxDets[2])
        return stats
    def _summarizeKps():
        stats = np.zeros((10,))
        stats[0] = _summarize(1, maxDets=20)
        stats[1] = _summarize(1, maxDets=20, iouThr=.5)
        stats[2] = _summarize(1, maxDets=20, iouThr=.75)
        stats[3] = _summarize(1, maxDets=20, areaRng='medium')
        stats[4] = _summarize(1, maxDets=20, areaRng='large')
        stats[5] = _summarize(0, maxDets=20)
        stats[6] = _summarize(0, maxDets=20, iouThr=.5)
        stats[7] = _summarize(0, maxDets=20, iouThr=.75)
        stats[8] = _summarize(0, maxDets=20, areaRng='medium')
        stats[9] = _summarize(0, maxDets=20, areaRng='large')
        return stats

    iouType = params.iouType
    if iouType == 'segm' or iouType == 'bbox':
        summarize = _summarizeDets
    elif iouType == 'keypoints':
        summarize = _summarizeKps
    stats = summarize()
    return stats