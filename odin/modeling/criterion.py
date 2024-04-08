# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from odin.utils.misc import is_dist_avail_and_initialized

import ipdb
st = ipdb.set_trace

def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


def get_uncertain_point_coords(
    coords, uncertainty, important_sample_ratio, num_points):
    """
    Args:
        coords (Tensor): A tensor of shape (R, 3) that contains the coordinates of the
            predicted masks in all images.
        uncertainty (Tensor): A tensor of shape (R, 1) that contains uncertainty scores
            for the predicted masks in all images.
        important_sample_ratio (float): The ratio of points to sample from the most uncertain
            locations.
        num_points (int): The total number of points to sample.
    Returns:
        coords (Tensor): A tensor of shape (num_points) that contains the indices of the
            points to sample from the most uncertain locations.
    """
    num_uncertain_points = int(important_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(uncertainty[:, 0, :], k=num_uncertain_points, dim=1)[1]
    if num_random_points > 0:
        idx = torch.cat(
            [
                idx,
                torch.randint(
                    high=coords.shape[1],
                    size=(coords.shape[0], num_random_points),
                    device=coords.device,
                ),
            ],
            dim=1,
        )
    return idx




class ODINSetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio,
                 supervise_sparse=False, cfg=None):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.cfg = cfg


        if self.cfg.MODEL.OPEN_VOCAB and not self.cfg.NON_PARAM_SOFTMAX:
            empty_weight = torch.ones(self.cfg.MODEL.MAX_SEQ_LEN)
        else:
            empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.supervise_sparse = supervise_sparse
        
    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        num_classes = src_logits.shape[-1]
        empty_weight = torch.ones(num_classes, device=src_logits.device)
        empty_weight[-1] = self.eos_coef
        target_classes = torch.full(
            src_logits.shape[:2], num_classes - 1, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, empty_weight)
        losses = {"loss_ce": loss_ce}
        return losses

    def loss_masks_3d(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        Note: Looping over batch is a necessary evil -- output_masks are generated by some operation
        using scatter_mean which can add padding of 0, but for masks 0 logit is 0.5 after sigmoid and 
        would cause undue loss. So we need to loop over batch and only compute loss on non-padded
        """
        # # looped version
        loss_masks = []
        loss_dice = []
        for batch_id, (map_id, target_id) in enumerate(indices):
            src_masks = outputs["pred_masks"][batch_id]
            src_masks = src_masks[map_id]

            if self.cfg.USE_SEGMENTS:
                target_masks = targets[batch_id]["segment_mask"][target_id]
            elif self.cfg.INPUT.VOXELIZE:
                target_masks = targets[batch_id]["voxel_masks"][target_id]
            else:
                target_masks = targets[batch_id]["masks"][target_id]

            target_masks = target_masks.to(src_masks)
            
            if not self.cfg.INPUT.VOXELIZE:
                target_masks = target_masks.flatten(1)
                src_masks = src_masks.flatten(1)

            # handle padding
            src_masks = src_masks[:, :target_masks.shape[1]]
            if self.supervise_sparse and not self.cfg.USE_SEGMENTS:
                valids = targets[batch_id]["valids"].flatten()
                src_masks = src_masks[:, valids]
                target_masks = target_masks[:, valids]

            if src_masks.shape[1] > self.cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS:
                # randomly subsample points
                idx = torch.randperm(src_masks.shape[1], device=src_masks.device)[:self.cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS]
                target_masks = target_masks[:, idx]
                src_masks = src_masks[:, idx]

            loss_masks.append(sigmoid_ce_loss_jit(src_masks, target_masks, 1))
            loss_dice.append(dice_loss_jit(src_masks, target_masks, 1))
        
        losses = {
            "loss_mask": torch.sum(torch.stack(loss_masks)) / num_masks,
            "loss_dice": torch.sum(torch.stack(loss_dice)) / num_masks,
        }

        return losses


    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        # Modified to handle video
        target_masks = torch.cat([t['masks'][i] for t, (_, i) in zip(targets, indices)]).to(src_masks)

        if self.supervise_sparse:
            target_valids = torch.cat([targets[batch_id]["valids"][None].repeat(len(indices[batch_id][1]), 1, 1, 1) for batch_id in range(len(targets))]).to(src_masks)
        else:
            target_valids = None


        # No need to upsample predictions as we are using normalized coordinates :)
        # NT x 1 x H x W
        N, T, H, W = target_masks.shape
        if N == 0:
            return {"loss_mask": 0.0  * src_masks.sum(), "loss_dice": 0.0 * src_masks.sum()}
        
        if self.num_points == -1:
            target_masks = F.interpolate(
                    target_masks,
                    size=src_masks.shape[-2:],
                    mode="nearest",
                ).flatten(0, 1)
            src_masks = src_masks.flatten(0, 1)
            
            point_labels = target_masks
            point_logits = src_masks
        else:
            src_masks = src_masks.flatten(0, 1)[:, None]
            target_masks = target_masks.flatten(0, 1)[:, None]

            with torch.no_grad():
                # sample point_coords
                point_coords = get_uncertain_point_coords_with_randomness(
                    src_masks,
                    lambda logits: calculate_uncertainty(logits),
                    self.num_points,
                    self.oversample_ratio,
                    self.importance_sample_ratio,
                )
                # get gt labels
                point_labels = point_sample(
                    target_masks,
                    point_coords,
                    align_corners=False,
                ).squeeze(1)

            point_logits = point_sample(
                src_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

            if target_valids is not None:
                target_valids = target_valids.flatten(0, 1)[:, None]
                point_valids = point_sample(
                    target_valids,
                    point_coords,
                    align_corners=False,
                ).squeeze(1).bool()
                
                with torch.no_grad():
                    point_labels[~point_valids] = 0
                    point_logits[~point_valids] = point_logits.min()

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks, decoder_3d=False):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks_3d if decoder_3d else self.loss_masks,
        }

        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets, decoder_3d=False):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets, decoder_3d)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=outputs['pred_masks'].device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()
        
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks, decoder_3d))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                if 'text_attn_mask' in outputs:
                    aux_outputs['text_attn_mask'] = outputs['text_attn_mask']
                indices = self.matcher(aux_outputs, targets, decoder_3d)

                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks, decoder_3d)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
