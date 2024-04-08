import os
import ipdb
st = ipdb.set_trace


from .scannet_context import (
    register_scannet_context_instances,
    _get_scannet_instances_meta,
    register_scannet_context_instances_single,
    _get_scannet_instances20_meta, 
    _get_dataset_instances_meta
)

from detectron2.data.datasets.builtin import register_all_coco, _PREDEFINED_SPLITS_COCO
from detectron2.data import DatasetCatalog, MetadataCatalog


_PREDEFINED_SPLITS_CONTEXT20_SCANNET_SINGLE_100K = {
    "scannet_context_instance_train_20cls_single_highres_100k": (
        "frames_square_highres",
        "scannet_highres_train.coco.json"
    ),
    "scannet_context_instance_val_20cls_single_highres_100k": (
        "frames_square_highres",
         "scannet_highres_val.coco.json"
    ),
    "scannet_context_instance_train_eval_20cls_single_highres_100k": (
        "frames_square_highres",
        "scannet_highres_ten_scene.coco.json", 
    ),
    "scannet_context_instance_test_20cls_single_highres_100k": (
        "frames_square_highres",
         "scannet_highres_test.coco.json"
    ),
    "scannet_context_instance_debug_20cls_single_highres_100k": (
        "frames_square_highres",
        "scannet_highres_two_scene.coco.json",  
    ),

}

_PREDEFINED_SPLITS_CONTEXT20_SCANNET200_SINGLE_100K = {
    "scannet200_context_instance_train_200cls_single_highres_100k": (
        "frames_square_highres",
        "scannet200_highres_train.coco.json"
    ),
    "scannet200_context_instance_val_200cls_single_highres_100k": (
        "frames_square_highres",
         "scannet200_highres_val.coco.json"
    ),
    "scannet200_context_instance_trainval_200cls_single_highres_100k": (
        "frames_square_highres",
        "scannet200_highres_trainval.coco.json",
    ),
    "scannet200_context_instance_test_200cls_single_highres_100k": (
        "frames_square_highres",
        "scannet200_highres_test.coco.json",
    ),
    "scannet200_context_instance_train_eval_200cls_single_highres_100k": (
        "frames_square_highres",
        "scannet200_highres_ten_scene.coco.json", 
    ),
    "scannet200_context_instance_debug_200cls_single_highres_100k": {
        "frames_square_highres",
        "scannet200_highres_two_scene.coco.json", 
    },
}

_PREDEFINED_SPLITS_AI2THOR = {
    "ai2thor_highres_train_single": (
        "ai2thor_frames_512",
        "ai2thor_train_highres.coco.json"
    ),
    "ai2thor_highres_val_single": (
        "ai2thor_frames_512",
        "ai2thor_val_highres.coco.json"
    ),
    "ai2thor_highres_val50_single": (
        "ai2thor_frames_512",
        "ai2thor_val50_highres.coco.json"
    ),
    "ai2thor_highres_train_eval_single": (
        "ai2thor_frames_512",
        "ai2thor_ten_scene_highres.coco.json"
    ),
     "ai2thor_highres_debug_single": (
        "ai2thor_frames_512",
        "ai2thor_two_scene_highres.coco.json"
    ),
     
}

_PREDEFINED_SPLITS_ALFRED = {
    "alfred_train_single": (
        "teach_data/m2f3d_data",
        "ai2thor_alfred_train_highres.coco.json"
    ),
    "alfred_valid_seen_single": (
        "teach_data_valid_seen/m2f3d_data",
         "ai2thor_alfred_valid_seen_highres.coco.json"
    ),
    "alfred_valid_unseen_single": (
        "teach_data_valid_unseen/m2f3d_data",
        "ai2thor_alfred_valid_unseen_highres.coco.json"
    ),
     "alfred_debug_single": (
        "teach_data/m2f3d_data",
        "ai2thor_alfred_two_scene_highres.coco.json"
    ),  
}


_PREDEFINED_SPLITS_S3DIS = {
    "s3dis_train_single": (
        "s3dis_frames_fixed",
        "s3dis_train.coco.json"
    ),
    "s3dis_val_single": (
        "s3dis_frames_fixed",
         "s3dis_val.coco.json"
    ),
    "s3dis_train_eval_single": (
        "s3dis_frames_fixed",
        "s3dis_ten_scene.coco.json"
    ),
     "s3dis_debug_single": (
        "s3dis_frames_fixed",
        "s3dis_two_scene.coco.json"
    ),
}


_PREDEFINED_SPLITS_MATTERPORT = {
    "matterport_train_single": (
        "matterport_frames",
        "m3d_train.coco.json"
    ),
    "matterport_val_single": (
        "matterport_frames",
         "m3d_val.coco.json"
    ),
    "matterport_train_eval_single": (
        "matterport_frames",
        "m3d_ten_scene.coco.json"
    ),
     "matterport_debug_single": (
        "matterport_frames",
        "m3d_two_scene.coco.json"
    ),
}


_PREDEFINED_SPLITS_AI2THOR_JOINT = {
    "ai2thor_highres_train": (
        "ai2thor_frames_512",
        "ai2thor_train_highres.coco.json"
    ),
    "ai2thor_highres_val": (
        "ai2thor_frames_512",
        "ai2thor_val_highres.coco.json"
    ),
    "ai2thor_highres_train_eval": (
        "ai2thor_frames_512",
        "ai2thor_ten_scene_highres.coco.json"
    ),
     "ai2thor_highres_debug": (
        "ai2thor_frames_512",
        "ai2thor_two_scene_highres.coco.json"
    ),
}



def register_all_dataset(root, dataset_name="ai2thor"):
    if dataset_name == "ai2thor":
        split_dict = _PREDEFINED_SPLITS_AI2THOR_JOINT
    else:
        raise NotImplementedError("dataset_name {} not supported".format(dataset_name))
    for key, (image_root, json_file) in split_dict.items():
        register_scannet_context_instances(
            key,
            _get_dataset_instances_meta(dataset=dataset_name),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

def register_all_scannet_context20_scannet_single(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_CONTEXT20_SCANNET_SINGLE_100K.items():
        register_scannet_context_instances_single(
            key,
            _get_scannet_instances20_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )



def register_all_dataset_single(root, dataset_name="ai2thor"):
    if dataset_name == "ai2thor":
        split_dict = _PREDEFINED_SPLITS_AI2THOR
    elif dataset_name == "alfred":
        split_dict = _PREDEFINED_SPLITS_ALFRED
    elif dataset_name == "s3dis":
        split_dict = _PREDEFINED_SPLITS_S3DIS
    elif dataset_name == "matterport":
        split_dict = _PREDEFINED_SPLITS_MATTERPORT
    elif dataset_name == 'scannet200':
        split_dict = _PREDEFINED_SPLITS_CONTEXT20_SCANNET200_SINGLE_100K
    else:
        raise NotImplementedError("dataset_name {} not supported".format(dataset_name))
    for key, (image_root, json_file) in split_dict.items():
        register_scannet_context_instances_single(
            key,
            _get_dataset_instances_meta(dataset=dataset_name),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

if __name__.endswith(".builtin"):
    # Detectron1 registers some datasets on its own, we remove them here
    DatasetCatalog.clear()
    MetadataCatalog.clear()

    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    register_all_scannet_context20_scannet_single(_root)

    # ai2thor register
    register_all_dataset_single(_root, dataset_name="ai2thor")
    register_all_dataset(_root, dataset_name='ai2thor')
    
    # alfred register
    register_all_dataset_single(_root, dataset_name="alfred")

    # s3dis register
    register_all_dataset_single(_root, dataset_name="s3dis")

    # scannet200 register
    register_all_dataset_single(_root, dataset_name="scannet200")
    
    # matterport register
    register_all_dataset_single(_root, dataset_name="matterport")

    _root_2d = os.getenv("DETECTRON2_DATASETS_2D", "datasets")
    register_all_coco(_root_2d)

