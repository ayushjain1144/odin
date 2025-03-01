# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from typing import Dict
import torch

from torch import nn
from torch_scatter import scatter_mean

from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from odin.modeling.transformer_decoder.odin_transformer_decoder import build_transformer_decoder
from odin.modeling.pixel_decoder.msdeformattn import build_pixel_decoder


import ipdb
st = ipdb.set_trace

@SEM_SEG_HEADS_REGISTRY.register()
class ODINHead(nn.Module):

    _version = 2

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "sem_seg_head" in k and not k.startswith(prefix + "predictor"):
                    newk = k.replace(prefix, prefix + "pixel_decoder.")
                    # logger.debug(f"{k} ==> {newk}")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        num_classes: int,
        pixel_decoder: nn.Module,
        loss_weight: float = 1.0,
        ignore_value: int = -1,
        # extra parameters
        transformer_in_feature: str,
        decoder_3d=False,
        cross_view=False,
        hidden_dim=None,
        cfg=None,
        transformer_predictor_in_channels=None,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
            transformer_predictor: the transformer decoder that makes prediction
            transformer_in_feature: input feature name to the transformer_predictor
        """
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]

        self.ignore_value = ignore_value
        self.common_stride = 4
        self.loss_weight = loss_weight
        self.hidden_dim = hidden_dim
        self.cfg = cfg

        self.pixel_decoder = pixel_decoder
        
        self.transformer_in_feature = transformer_in_feature

        self.num_classes = num_classes
        self.decoder_3d = decoder_3d
        self.cross_view = cross_view

        self.predictor = build_transformer_decoder(
                cfg,
                transformer_predictor_in_channels,
                mask_classification=True,
            )
        
    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        # figure out in_channels to transformer predictor
        if cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "transformer_encoder":
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        elif cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "pixel_embedding":
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        elif cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "multi_scale_pixel_decoder":  # for maskformer2
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        else:
            transformer_predictor_in_channels = input_shape[cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE].channels
        return {
            "input_shape": {
                k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
            },
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "pixel_decoder": build_pixel_decoder(cfg, input_shape),
            "loss_weight": cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT,
            "transformer_in_feature": cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE,
            "decoder_3d": cfg.MODEL.DECODER_3D,
            "cross_view": cfg.MODEL.CROSS_VIEW_CONTEXTUALIZE,
            "hidden_dim": cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
            "cfg": cfg,
            "transformer_predictor_in_channels": transformer_predictor_in_channels,
        }

    def forward(
        self, features, shape=None, mask=None,
        multiview_data=None, scannet_pc=None, scannet_p2v=None,
        segments=None, decoder_3d=False,
        captions=None, positive_map_od=None, num_classes=None,
        scene_names=None
    ):
        """
        Args:
            - features: dict,
                keys are ['res2', 'res3', 'res4', 'res5']
                    if self.cfg.MODEL.UPSAMPLE_FMAP 'res1' will also be there
                values are [(B*num_views, F_i, H_i, W_i)]
            - shape: the shape of original image, e.g. (B, 3, 256, 320)
            - mask: padding mask, usually None
            - multiview_data: {
                'multi_scale_xyz': [(B, num_views, H_i, W_i, 3)],  # xyz
                'multi_scale_p2v': [(B, num_views * H_i * W_i)]  # point2voxel
            }  # order is from 'res5' to 'res2'
            - scannet_pc: original scannet points
            - segments: tensor (B, num_points), scannet segemnts

        Returns:
            - predictions: {
                'pred_logits': tensor (B, num_queries, num_classes),
                'pred_masks': tensor (B, num_queries, num_views, H_m, W_m),
                    m is the index of the largest feature map that we score
                'pred_scannet_masks':,
                'aux_outputs': [{'pred_logits', 'pred_masks'}]
                    intermediate outputs, shapes same to corr above fields
                    length is equal to the number of layers
            }
        """
        multi_scale_xyz = None  # [features to attend to, diff scales]
        mask_features_xyz = None  # features to score
        mask_features_p2v = None  # point2voxel of mask_features_xyz

        # Select feature scales to attend to
        if decoder_3d:
            multi_scale_xyz = multiview_data['multi_scale_xyz']
            mask_features_xyz = multi_scale_xyz[3]  # xyz of res2
            multi_scale_xyz = multi_scale_xyz[:3]  # xyz of [res5:3]
            mask_features_p2v = multiview_data['multi_scale_p2v'][3] if self.cfg.INPUT.VOXELIZE else None  # p2v of res2


        if decoder_3d and self.cfg.USE_GHOST_POINTS:
            scannet_pc = scatter_mean(
                scannet_pc, scannet_p2v, dim=1
            )
            scannet_p2v = torch.arange(scannet_pc.shape[1], device=scannet_pc.device).unsqueeze(0).repeat(scannet_pc.shape[0], 1)

        # Decoder-part (upsampler) of ResUNet
        mask_features, _, multi_scale_features = self.pixel_decoder.forward_features(
            features, shape, multi_scale_xyz, multiview_data=multiview_data,
            mask_features_xyz=mask_features_xyz, 
            mask_features_p2v=mask_features_p2v, 
            scannet_pc=scannet_pc, scannet_p2v=scannet_p2v,
            decoder_3d=decoder_3d
        )

        # mask_features (B*num_views, F_m, H_m, W_m), m is largest f_map (res2)
        # multi_scale_features: feats of small scales [res5, res4, res3]
        if decoder_3d:
            if self.cfg.USE_GHOST_POINTS and not self.cfg.DO_FEATURE_INTERPOLATION_LATER:
                mask_features_xyz = scannet_pc
            elif self.cfg.INPUT.VOXELIZE:
                mask_features_xyz = scatter_mean(
                    mask_features_xyz.flatten(1, 3), mask_features_p2v, dim=1
                )

        # Feed to Transformer decoder
        if shape is None:
            shape = [multi_scale_features[0].shape[0], 1]
            
        predictions = self.predictor(
            x=multi_scale_features,
            mask_features=mask_features,
            shape = shape[:2],
            x_xyz=multi_scale_xyz,
            mask=mask,
            mask_features_xyz=mask_features_xyz, 
            multiview_data=multiview_data, 
            segments=segments,
            scannet_p2v=scannet_p2v,
            decoder_3d=decoder_3d,
            captions=captions,
            positive_map_od=positive_map_od,
            num_classes=num_classes,
            scannet_pc=scannet_pc
        )
        return predictions
