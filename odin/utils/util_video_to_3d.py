import torch
import itertools


import ipdb
st = ipdb.set_trace

def convert_video_instances_to_3d(
    instances_all, num_frames, h_pad, w_pad, device, convert_point_semantic_instance=False, skip_classes=None, 
    multiplier=1000, evaluate_subset=False
    ):
    all_instances = list(itertools.chain.from_iterable([instances.instance_ids for instances in instances_all]))
    unique_instances = torch.unique(torch.tensor(all_instances))

    if skip_classes is not None:
        unique_instances = unique_instances[~torch.tensor(skip_classes).unsqueeze(1).eq(unique_instances // multiplier + 1).any(0)]

    inst_to_count = {inst.item(): id for id, inst in enumerate(unique_instances)}
    num_instances = len(unique_instances)
    mask_shape = [num_instances, num_frames, h_pad, w_pad]
    if device is None:
        gt_masks_per_video = torch.zeros(mask_shape, dtype=torch.bool)
        gt_classes_per_video = torch.zeros(num_instances, dtype=torch.int64)
    else:
        gt_masks_per_video = torch.zeros(mask_shape, dtype=torch.bool, device=device)
        gt_classes_per_video = torch.zeros(num_instances, dtype=torch.int64, device=device)

    if convert_point_semantic_instance:
        if device is None:
            point_semantic_instance_label_per_video = torch.zeros(
                (num_frames, h_pad, w_pad), dtype=torch.int64
            )
        else:
            point_semantic_instance_label_per_video = torch.zeros(
                (num_frames, h_pad, w_pad), dtype=torch.int64, device=device
            )
    
    for f_i, targets_per_frame in enumerate(instances_all):
        if device is not None:
            targets_per_frame = targets_per_frame.to(device)
        gt_cls = targets_per_frame.gt_classes
        gt_instance_ids = targets_per_frame.instance_ids
        h, w = targets_per_frame.image_size
        for idx, instance_id in enumerate(gt_instance_ids):
            if instance_id.item() not in inst_to_count:
                continue
            inst_idx = inst_to_count[instance_id.item()]
            gt_masks_per_video[inst_idx, f_i, :h, :w] = targets_per_frame.gt_masks[idx]
            gt_classes_per_video[inst_idx] = gt_cls[idx]
            if convert_point_semantic_instance:
                new_instance_id = (gt_cls[idx]+1) * multiplier + instance_id % multiplier
                point_semantic_instance_label_per_video[f_i, :h, :w][targets_per_frame.gt_masks[idx]] = new_instance_id
    
    target_dict = {
        "labels": gt_classes_per_video,
        "masks": gt_masks_per_video
    }

    if convert_point_semantic_instance:
        target_dict["point_semantic_instance_label"] = point_semantic_instance_label_per_video

    return target_dict