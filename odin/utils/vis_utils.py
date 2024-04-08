import os
import wandb
import torch
import numpy as np
from torch.nn import functional as F
from einops import rearrange
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import pyviz3d.visualizer as vis
from odin.global_vars import SCANNET_COLOR_MAP_20, \
    SCANNET_COLOR_MAP_200, GENERAL_COLOR_MAP_200

from .feature_vis import embedding_to_3d_color
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import Instances

import ipdb
st = ipdb.set_trace


def visualize_embeddings(embeddings, words, name='embeddings'):
    tsne = TSNE(n_components=2, random_state=0, perplexity=len(words)-1)
    two_d_embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 8))
    for i, word in enumerate(words):
        x, y = two_d_embeddings[i, :]
        plt.scatter(x, y)
        plt.annotate(word, (x, y), xytext=(5, 2), textcoords="offset points", ha="right", va="bottom")
    plt.savefig(f"{name}.png")


def visualize_features(features):
    """visualize PCA of features
    
    Keyword arguments:
    features -- features to visualize (B X V X H X W X C)
    
    Return: None
    """
    b, v, h, w, c = features.shape
    color = embedding_to_3d_color(
        rearrange(features, 'b v h w c -> b c (v h w)'),
        project_type='pca'
    )
    color = rearrange(color, 'b c (v h w) -> b v h w c', v=v, h=h, w=w).numpy()
    
    if wandb.run is None:
        wandb.init(project='odin')
        
    wandb.log({"features": [wandb.Image(color[i, j]) for i in range(b) for j in range(v)]})
    
    
def visualize_pca_from_feature_folders(feature_folders):
    """visualize PCA of features
    
    Keyword arguments:
    feature_folders -- list of feature folders
    
    Return: None
    """
    if wandb.run is None:
        wandb.init(project='odin')
        
    feature_dict = {}
    for feature_folder in feature_folders:
        # iterate over all files in the folder
        for file in sorted(os.listdir(feature_folder)):
            if file.endswith(".npy"):
                feature_path = os.path.join(feature_folder, file)
                feature = torch.from_numpy(np.load(feature_path))
                if feature.shape[1] != 1:
                    print(f"Skipping {feature_path} as it has shape {feature.shape}")
                    continue
                if feature_folder in feature_dict:
                    feature_dict[feature_folder].append(feature)
                else:
                    feature_dict[feature_folder] = [feature]
                    
    # compute principal components from all features
    all_features = torch.cat([torch.cat(feature_dict[feature_folder], 1) for feature_folder in feature_dict], dim=0)
    print(all_features.mean(), all_features.sum())
    
    # st()
    pca = embedding_to_3d_color(rearrange(all_features, 'b v h w c -> c (b v h w)')[None],
    )[0].permute(1, 0).contiguous()
    
    output = rearrange(pca, '(b v h w) c -> b v h w c', b=all_features.shape[0], v=all_features.shape[1], h=all_features.shape[2], w=all_features.shape[3])

    output = output.numpy()
    for j in range(output.shape[1]):
        wandb.log({"principal_component": [wandb.Image(output[i, j]) for i in range(output.shape[0])]})
        

def visualize_2d_masks(images, masks, labels, coco_metadata, captions=None, field_name=None):
    """sumary_line
    
    Keyword arguments:
    images: B, V, 3, H, W
    masks: list of length B, each elemet N X V X H X W
    labels: list of length B, each element N
    """
    if wandb.run is None:
        wandb.init(project='odin')
    
    B, V, _, H, W = images.shape
    for i in range(B):
        images_ = []
        for j in range(V):
            im = images[i, j].permute(1, 2, 0).numpy()
            v = Visualizer(im, coco_metadata)
            predictions = Instances((H, W))
            predictions.pred_masks = masks[i][:, j].cpu().numpy()
            predictions.pred_classes = labels[i].cpu().numpy()
            instance_result = v.draw_instance_predictions(predictions).get_image()
            # images_.append(instance_result)
        
            # image = np.concatenate(images_, axis=1)
            captions_ = captions[i] if captions is not None else None
            wandb.log({field_name: wandb.Image(instance_result, caption=captions_)})
    

def convert_instance_to_semantic(masks, labels):
    """sumary_line
    
    Keyword arguments:
        masks: N, V, H, W
        labels: N
    returns:
        V, H, W
    """
    N, V, H, W = masks.shape
    semantic = torch.zeros(V, H, W, dtype=torch.long, device=masks.device)
    for i in range(N):
        semantic[masks[i] > 0] = labels[i]
    return semantic
              
    

def visualize_2d_masks_semantic(images, masks, thing_classes, captions=None, field_name='sem_pred', gt_masks=None):
    # uses native wandb logging instead of detectron2 visualizer
    """sumary_line
    
    Keyword arguments:
        images: B, 3, H, W
        masks: B, H, W
        coco_metadata: metadata for the dataset
        gt_masks: B, H, W
    """
    
    if wandb.run is None:
        wandb.init(project='odin')
        
    B, H, W = masks.shape
    class_labels = {i + 1: thing_classes[i] for i in range(len(thing_classes))}
    
    for i in range(B):
        wandb.log({
            f"{field_name}": wandb.Image(
                images[i].permute(1, 2, 0).numpy(),
                masks={
                    'predictions': {"mask_data": masks[i].numpy(), "class_labels": class_labels},
                    'ground_truth': {"mask_data": gt_masks[i].numpy(), "class_labels": class_labels}
                }
            ),
        })
    


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


def plot_3d_offline(
    pc, pc_color, masks, labels, valids=None,
    gt_masks=None, gt_labels=None, scene_name=None,
    data_dir=None, mask_classes=None, dataset_name='scannet'):
    """
    Input:
        pc: N, 3
        pc_color: N, 3 (range: [0, 1])
        masks: M, N
        labels: M
        valids: N, 
        gt_masks: M_, N
        gt_labels: M_
        scene_name: str
        data_dir: str
        mask_classes: list of classes to exclude from the visualization
        dataset_name: 
    """
    if 'ai2thor' in dataset_name:
        color_map = GENERAL_COLOR_MAP_200
    elif 'scannet200' in dataset_name:
        color_map = SCANNET_COLOR_MAP_200
    else:
        color_map = SCANNET_COLOR_MAP_20

    if valids is not None:
        pc = pc[valids]
        pc_color = pc_color[valids]
        if masks is not None:
            masks = masks[:, valids]
        if gt_masks is not None:
            gt_masks = gt_masks[:, valids]
        
    if mask_classes is not None:
        mask_classes = set(mask_classes)
        if masks is not None:
            masks = masks[labels != mask_classes]
            labels = labels[labels != mask_classes]
        if gt_masks is not None:
            gt_masks = gt_masks[gt_labels != mask_classes]
            gt_labels = gt_labels[gt_labels != mask_classes]

    v = vis.Visualizer()
    point_size = 25
    v.add_points("RGB", pc,
                    colors=pc_color*255,
                    alpha=0.8,
                    visible=False,
                    point_size=point_size)
    
    if gt_masks is not None:
        masks_pcs, masks_colors = get_color_pc_from_mask(
            gt_masks, gt_labels, pc, color_map=color_map)
        v.add_points("Semantics (GT)", masks_pcs,
                        colors=masks_colors,
                        alpha=0.8,
                        visible=False,
                        point_size=point_size)

        masks_pcs, masks_colors = get_color_pc_from_mask(
            gt_masks, gt_labels, pc, instance=True, color_map=color_map)
        v.add_points("Instances (GT)", masks_pcs,
                        colors=masks_colors,
                        alpha=0.8,
                        visible=False,
                        point_size=point_size)

    masks_pcs, masks_colors = get_color_pc_from_mask(
        masks, labels - 1, pc, color_map=color_map)
    v.add_points("Semantics (Ours)", masks_pcs,
                colors=masks_colors,
                visible=False,
                alpha=0.8,
                point_size=point_size)
    
    masks_pcs, masks_colors = get_color_pc_from_mask(
        masks, labels - 1, pc, instance=True, color_map=color_map)
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
            
        
    


if __name__ == "__main__":
    feature_folders = [
        # "/projects/katefgroup/language_grounding/feature_vis/2d",
        # "/projects/katefgroup/language_grounding/feature_vis/no_tri_no_vox"
        
        # '/projects/katefgroup/language_grounding/feature_vis/2d_single_view',
        '/projects/katefgroup/language_grounding/feature_vis/3d_single_view_float64',
        '/projects/katefgroup/language_grounding/feature_vis/3d_single_view_float64_rerun',
        # 3d_single_view_float64_rerun
    ]
    visualize_pca_from_feature_folders(feature_folders)