import numpy as np
import torchvision.transforms as transforms
from collections import OrderedDict
from imageio import imread
import math 
from PIL import Image
from copy import deepcopy
import os
from odin.global_vars import SCANNET_COLOR_MAP_20

import matplotlib.pyplot as plt

import ipdb
st = ipdb.set_trace

import pyviz3d.visualizer as vis

import wandb


def plot_only_3d(xdata, ydata, zdata, color=None, b_min=2, b_max=8, view=(45, 45)):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, dpi=200)
    ax.view_init(view[0], view[1])

    ax.set_xlim(b_min, b_max)
    ax.set_ylim(b_min, b_max)
    ax.set_zlim(b_min, b_max)

    ax.scatter3D(xdata, ydata, zdata, c=color, cmap='rgb', s=0.1)


def plot_masks(masks, xdata, ydata, zdata, color):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_axis_off()

    color_masks = np.zeros(xdata.shape + (3,), dtype=np.float32)
    if masks is not None:
        for i, mask in enumerate(masks):
            mask = mask.nonzero()[0]
            color_masks[mask] = np.array(SCANNET_COLOR_MAP_20[(i+1) % 40]) / 255.0

    color_new = color_masks * 0.5 + color * 0.5

    ax.scatter3D(xdata, ydata, zdata, c=color_new, cmap='rgb')
    ax.set_xlim(xdata.min(), xdata.max())
    ax.set_ylim(ydata.min(), ydata.max())
    ax.set_zlim(zdata.min(), zdata.max())

    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    fig.clf()
    plt.close(fig)
    return image_from_plot


def plot_3d(xdata, ydata, zdata, color=None, b_min=2, b_max=8, view=(45, 45), 
            masks=None, legend=None, valids=None, labels=None,
            gt_masks=None, gt_labels=None):

    if valids is not None:
        xdata = xdata[valids.reshape(-1)]
        ydata = ydata[valids.reshape(-1)]
        zdata = zdata[valids.reshape(-1)]
        color = color[valids.reshape(-1), :]
        if masks is not None:
            masks = masks[:, valids.reshape(-1)]
        if gt_masks is not None:
            gt_masks = gt_masks[:, valids.reshape(-1)]

    
    pred_fig = plot_masks(masks, xdata, ydata, zdata, color)
    gt_fig = plot_masks(gt_masks, xdata, ydata, zdata, color)

    # concatenate two images
    fig = np.concatenate((pred_fig, gt_fig), axis=1)

    log_dict = {"pred_3d": wandb.Image(fig, caption="pred_3d")}
    plt.close()
    return log_dict

def get_color_pc_from_mask(_mask, label, pcd, instance=False,
    color_map=SCANNET_COLOR_MAP_20):
    point_select = np.zeros(_mask.shape[1], dtype=bool)
    if _mask is not None:
        for i, __mask in enumerate(_mask):
            __mask = __mask.nonzero()[0]
            point_select[__mask] = True
    
    masks_pcs = pcd[point_select, :]

    color_masks = np.zeros((_mask.shape[1], 3), dtype=np.float32)
    if _mask is not None:
        for i, __mask in enumerate(_mask):
            __mask = __mask.nonzero()[0]
            if instance:
                color_masks[__mask] = np.array(color_map[(i+1) % (len(color_map)-1)])
            else:
                color_masks[__mask] = np.array(color_map[(label[i].item()+1) % (len(color_map)-1)])
    masks_colors = color_masks[point_select, :]
    return masks_pcs, masks_colors


def plot_3d_new(xdata, ydata, zdata, color=None, b_min=2, b_max=8, view=(45, 45), 
            masks=None, legend=None, valids=None, labels=None,
            gt_masks=None, gt_labels=None, point_size=25, scene_name=None, data_dir=None, 
            mask_classes=[18, 19], color_map=SCANNET_COLOR_MAP_20):

    if valids is not None:
        xdata = xdata[valids.reshape(-1)]
        ydata = ydata[valids.reshape(-1)]
        zdata = zdata[valids.reshape(-1)]
        color = color[valids.reshape(-1), :]
        if masks is not None:
            masks = masks[:, valids.reshape(-1)]
        if gt_masks is not None:
            gt_masks = gt_masks[:, valids.reshape(-1)]

    if gt_labels is not None:
        mask_out = np.zeros_like(gt_labels)
        for mask_class in mask_classes:
            mask_out = np.logical_or(mask_out, gt_labels == mask_class-1)
        gt_labels = gt_labels[~mask_out]
        gt_masks = gt_masks[~mask_out]

    pcd = np.vstack([xdata, ydata, zdata]).transpose(1, 0)

    # plot gt
    v = vis.Visualizer()
    v.add_points("RGB", pcd,
                    colors=color*255,
                    alpha=0.8,
                    visible=False,
                    point_size=point_size)
    
    if gt_masks is not None:
        masks_pcs, masks_colors = get_color_pc_from_mask(
            gt_masks, gt_labels, pcd, color_map=color_map)
        v.add_points("Semantics (GT)", masks_pcs,
                        colors=masks_colors,
                        alpha=0.8,
                        visible=False,
                        point_size=point_size)

        masks_pcs, masks_colors = get_color_pc_from_mask(
            gt_masks, gt_labels, pcd, instance=True, color_map=color_map)
        v.add_points("Instances (GT)", masks_pcs,
                        colors=masks_colors,
                        alpha=0.8,
                        visible=False,
                        point_size=point_size)

    masks_pcs, masks_colors = get_color_pc_from_mask(
        masks, labels - 1, pcd, color_map=color_map)
    v.add_points("Semantics (Ours)", masks_pcs,
                colors=masks_colors,
                visible=False,
                alpha=0.8,
                point_size=point_size)
    
    masks_pcs, masks_colors = get_color_pc_from_mask(
        masks, labels - 1, pcd, instance=True, color_map=color_map)
    v.add_points("Instances (Ours)", masks_pcs,
                    colors=masks_colors,
                    visible=False,
                    alpha=0.8,
                    point_size=point_size)
    
    if data_dir is None:
        data_dir = '/projects/katefgroup/language_grounding/bdetr2_visualizations'

    if os.path.exists(data_dir) == False:
        os.makedirs(data_dir)
        
    v.save(f"{data_dir}/{scene_name}")

def plot_3d_old(xdata, ydata, zdata, color=None, b_min=2, b_max=8, view=(45, 45), 
            masks=None, legend=None, valids=None, labels=None,
            gt_masks=None, gt_labels=None):
    # overlay masks and labels if provided
    if valids is not None:
        xdata = xdata[valids.reshape(-1)]
        ydata = ydata[valids.reshape(-1)]
        zdata = zdata[valids.reshape(-1)]
        color = color[valids.reshape(-1), :]
        if masks is not None:
            masks = masks[:, valids.reshape(-1)]
        if gt_masks is not None:
            gt_masks = gt_masks[:, valids.reshape(-1)]

    # plot rgb
    fig1, ax = plt.subplots(subplot_kw={"projection": "3d"}, dpi=200)
    ax.view_init(view[0], view[1])

    ax.set_xlim(b_min, b_max)
    ax.set_ylim(b_min, b_max)
    ax.set_zlim(b_min, b_max)

    ax.scatter3D(xdata, ydata, zdata, c=color, cmap='rgb', s=0.1)

    # plot semantic masks
    fig2, ax = plt.subplots(subplot_kw={"projection": "3d"}, dpi=200)
    ax.view_init(view[0], view[1])

    ax.set_xlim(b_min, b_max)
    ax.set_ylim(b_min, b_max)
    ax.set_zlim(b_min, b_max)

    # 0 label: empty : black
    color_masks = np.zeros(xdata.shape + (3,), dtype=np.float32)
    if masks is not None:
        for i, mask in enumerate(masks):
            label = labels[i]
            mask = mask.nonzero()[0]
            color_masks[mask] = np.array(SCANNET_COLOR_MAP_20[label.item()+1]) / 255.0

    ax.scatter3D(xdata, ydata, zdata, c=color_masks, cmap='rgb', s=0.1)

    # plot gt semantic masks stacked if provided:
    if gt_labels is not None and gt_masks is not None:
        fig2_gt, ax = plt.subplots(subplot_kw={"projection": "3d"}, dpi=200)
        ax.view_init(view[0], view[1])

        ax.set_xlim(b_min, b_max)
        ax.set_ylim(b_min, b_max)
        ax.set_zlim(b_min, b_max)
        
        # 0 label: empty : black
        color_masks = np.zeros(xdata.shape + (3,), dtype=np.float32)
        if gt_masks is not None:
            for i, mask in enumerate(gt_masks):
                label = gt_labels[i]
                mask = mask.nonzero()[0]
                color_masks[mask] = np.array(SCANNET_COLOR_MAP_20[label.item()+1]) / 255.0

        ax.scatter3D(xdata, ydata, zdata, c=color_masks, cmap='rgb', s=0.1)        

    # plot instance masks
    fig3, ax = plt.subplots(subplot_kw={"projection": "3d"}, dpi=200)
    ax.view_init(view[0], view[1])

    ax.set_xlim(b_min, b_max)
    ax.set_ylim(b_min, b_max)
    ax.set_zlim(b_min, b_max)

    # 0 label: empty : black
    color_masks = np.zeros(xdata.shape + (3,), dtype=np.float32)
    if masks is not None:
        for i, mask in enumerate(masks):
            mask = mask.nonzero()[0]
            color_masks[mask] = np.array(SCANNET_COLOR_MAP_20[(i+1) % 40])

    ax.scatter3D(xdata, ydata, zdata, c=color_masks, cmap='rgb', s=0.1)

    # plot gt instance masks stacked if provided:
    if gt_labels is not None and gt_masks is not None:
        fig3_gt, ax = plt.subplots(subplot_kw={"projection": "3d"}, dpi=200)
        ax.view_init(view[0], view[1])

        ax.set_xlim(b_min, b_max)
        ax.set_ylim(b_min, b_max)
        ax.set_zlim(b_min, b_max)

        # 0 label: empty : black
        color_masks = np.zeros(xdata.shape + (3,), dtype=np.float32)
        if gt_masks is not None:
            for i, mask in enumerate(gt_masks):
                label = gt_labels[i]
                mask = mask.nonzero()[0]
                color_masks[mask] = np.array(SCANNET_COLOR_MAP_20[(i+1) % 40])

        ax.scatter3D(xdata, ydata, zdata, c=color_masks, cmap='rgb', s=0.1)   

    def map_fig_to_img(fig):
        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return image_from_plot

    if gt_labels is not None and gt_masks is not None:
        fig1 = np.concatenate([map_fig_to_img(fig1), map_fig_to_img(fig1)], axis=0)
        fig2 = np.concatenate([map_fig_to_img(fig2), map_fig_to_img(fig2_gt)], axis=0)
        fig3 = np.concatenate([map_fig_to_img(fig3), map_fig_to_img(fig3_gt)], axis=0)

    log_dict =  {
        "pred: top, gt: bottom ; rgb/semantics/instance": [wandb.Image(fig1), wandb.Image(fig2), wandb.Image(fig3)]
    }

    plt.close()

    return log_dict

def resize_crop_image(image, new_image_dims):
    image_dims = [image.shape[1], image.shape[0]]
    if image_dims == new_image_dims:
        return image
    resize_width = int(math.floor(new_image_dims[1] * float(image_dims[0]) / float(image_dims[1])))
    image = transforms.Resize([new_image_dims[1], resize_width], interpolation=Image.NEAREST)(Image.fromarray(image))
    image = transforms.CenterCrop([new_image_dims[1], new_image_dims[0]])(image)
    image = np.array(image)
    
    return image

def load_pose(filename):
    lines = open(filename).read().splitlines()
    assert len(lines) == 4
    lines = [[x[0],x[1],x[2],x[3]] for x in (x.split(" ") for x in lines)]

    return np.asarray(lines).astype(np.float32)

def load_depth(file, image_dims):
    depth_image = imread(file)
    # preprocess
    depth_image = resize_crop_image(depth_image, image_dims)
    depth_image = depth_image.astype(np.float32) / 1000.0

    return depth_image


def convert_3d_to_2d_dict_format(format_B):
    # for now both bbox and segm to same as format_B is giving
    # only segm, fix this later once supported.
    # bbox = OrderedDict()
    segm = OrderedDict()
    segm_per_class = OrderedDict()
    
    # Extract all_ap, all_ap_50% and all_ap_25%
    # bbox['AP'] = format_B['all_ap']
    # bbox['AP50'] = format_B['all_ap_50%']
    # bbox['AP75'] = format_B['all_ap_25%']
    segm = deepcopy(format_B)
    
    segm['AP'] = format_B['all_ap']
    segm['AP50'] = format_B['all_ap_50%']
    segm['AP75'] = format_B['all_ap_25%']
    del segm['all_ap']
    del segm['all_ap_50%']
    del segm['all_ap_25%']
    
    # Extract class-wise APs
    for cls_name, cls_data in format_B['classes'].items():
        # bbox[f'AP-{cls_name}'] = np.nan if np.isnan(cls_data['ap']) else cls_data['ap']
        segm_per_class[f'AP-{cls_name}'] = np.nan if np.isnan(cls_data['ap']) else cls_data['ap']
        
    return OrderedDict([('3d_segm', segm), ('3d_segm_per_class', segm_per_class)])


def convert_3d_to_2d_dict_format_semantic(data_dict):
    # for now both bbox and segm to same as format_B is giving
    # only segm, fix this later once supported.
    segm_per_class = OrderedDict()
    segm = OrderedDict()
    
    segm = deepcopy(data_dict)
    # delete the keys that are not needed
    del segm['class_labels']
    del segm['iou_class']
    del segm['acc_class']
    
    # Extract all_ap, all_ap_50% and all_ap_25%
    class_labels = data_dict['class_labels']
    iou_class = data_dict['iou_class']
    acc_class = data_dict['acc_class']
    
    # Extract class-wise APs
    for i, class_name in enumerate(class_labels):
        segm_per_class[f'IOU-{class_name}'] = iou_class[i]
        segm_per_class[f'ACC-{class_name}'] = acc_class[i]

    return OrderedDict([('3d_semantic_per_class', segm_per_class), ('3d_semantic', segm)])

