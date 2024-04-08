# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
from typing import Callable, Dict, List, Optional, Union

import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import normal_
from torch.cuda.amp import autocast

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from odin.modeling.transformer_decoder.position_encoding import PositionEmbeddingSine
from odin.modeling.meta_arch.self_cross_attention_layers import _get_clones, _get_activation_fn
from .ops.modules import MSDeformAttn
from odin.modeling.meta_arch.cross_view_attention import CrossViewPAnet
from odin.modeling.backproject.backproject import interpolate_feats_3d


import ipdb
st = ipdb.set_trace


def build_pixel_decoder(cfg, input_shape):
    """
    Build a pixel decoder from `cfg.MODEL.MASK_FORMER.PIXEL_DECODER_NAME`.
    """
    name = cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME
    model = SEM_SEG_HEADS_REGISTRY.get(name)(cfg, input_shape)
    forward_features = getattr(model, "forward_features", None)
    if not callable(forward_features):
        raise ValueError(
            "Only SEM_SEG_HEADS with forward_features method can be used as pixel decoder. "
            f"Please implement forward_features for {name} to only return mask features."
        )
    return model


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

# MSDeformAttn Transformer encoder in deformable detr
class MSDeformAttnTransformerEncoderOnly(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu",
                 num_feature_levels=4, enc_n_points=4,
        ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_encoder_layers

        encoder_layer = MSDeformAttnTransformerEncoderLayer(d_model, dim_feedforward,
                                                            dropout, activation,
                                                            num_feature_levels, nhead, enc_n_points)
        self.encoder = MSDeformAttnTransformerEncoder(encoder_layer, num_encoder_layers)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    
    def prepare_inputs(self, srcs, pos_embeds):
        masks = [torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool) for x in srcs]
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        return src_flatten, mask_flatten, lvl_pos_embed_flatten, spatial_shapes, level_start_index, valid_ratios


    def forward(self, srcs, pos_embeds):
        
        src_flatten, mask_flatten, lvl_pos_embed_flatten, spatial_shapes, level_start_index, valid_ratios = self.prepare_inputs(
            srcs, pos_embeds)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        return memory, spatial_shapes, level_start_index


class MSDeformAttnTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class MSDeformAttnTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output


@SEM_SEG_HEADS_REGISTRY.register()
class MSDeformAttnPixelDecoder(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        transformer_dropout: float,
        transformer_nheads: int,
        transformer_dim_feedforward: int,
        transformer_enc_layers: int,
        conv_dim: int,
        mask_dim: int,
        norm: Optional[Union[str, Callable]] = None,
        # deformable transformer encoder args
        transformer_in_features: List[str],
        common_stride: int,
        cfg,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            transformer_dropout: dropout probability in transformer
            transformer_nheads: number of heads in transformer
            transformer_dim_feedforward: dimension of feedforward network
            transformer_enc_layers: number of transformer encoder layers
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__()
        self.cfg = cfg
        self.decoder_3d = cfg.MODEL.DECODER_3D
        self.conv_dim = conv_dim
        # self.num_frames = cfg.INPUT.SAMPLING_FRAME_NUM
        transformer_input_shape = {
            k: v for k, v in input_shape.items() if k in transformer_in_features
        }

        # this is the input shape of pixel decoder
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]  # starting from "res2" to "res5"
        self.feature_strides = [v.stride for k, v in input_shape]
        self.feature_channels = [v.channels for k, v in input_shape]
        
        # this is the input shape of transformer encoder (could use less features than pixel decoder
        transformer_input_shape = sorted(transformer_input_shape.items(), key=lambda x: x[1].stride)
        self.transformer_in_features = [k for k, v in transformer_input_shape]  # starting from "res2" to "res5"
        transformer_in_channels = [v.channels for k, v in transformer_input_shape]
        self.transformer_feature_strides = [v.stride for k, v in transformer_input_shape]  # to decide extra FPN layers

        self.transformer_num_feature_levels = len(self.transformer_in_features)
        if self.transformer_num_feature_levels > 1:
            input_proj_list = []
            # from low resolution to high resolution (res5 -> res2)
            for in_channels in transformer_in_channels[::-1]:
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, conv_dim, kernel_size=1),
                    nn.GroupNorm(32, conv_dim),
                ))
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(transformer_in_channels[-1], conv_dim, kernel_size=1),
                    nn.GroupNorm(32, conv_dim),
                )])

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        if self.cfg.MODEL.FREEZE_BACKBONE:
            for param in self.input_proj.parameters():
                param.requires_grad = False

        self.transformer = MSDeformAttnTransformerEncoderOnly(
            d_model=conv_dim,
            dropout=transformer_dropout,
            nhead=transformer_nheads,
            dim_feedforward=transformer_dim_feedforward,
            num_encoder_layers=transformer_enc_layers,
            num_feature_levels=self.transformer_num_feature_levels,
        )
        if self.cfg.MODEL.FREEZE_BACKBONE:
            for param in self.transformer.parameters():
                param.requires_grad = False

        if (cfg.MODEL.DECODER_3D or self.cfg.PASS_2D_CROSS_VIEW) and self.cfg.MODEL.CROSS_VIEW_CONTEXTUALIZE and not self.cfg.MODEL.NO_DECODER_PANET:
            self.cross_view_attn = nn.ModuleList([
                CrossViewPAnet(
                    latent_dim=conv_dim, nsample=self.cfg.MODEL.KNN,
                    dropout=self.cfg.MODEL.MASK_FORMER.DROPOUT, 
                    num_layers=self.cfg.DECODER_NUM_LAYERS,
                    cfg=self.cfg,
                ) for _ in range(self.transformer_num_feature_levels)
            ])

        self.pe_layer = self.init_pe()

        self.mask_dim = mask_dim
        # use 1x1 conv instead
        self.mask_features = Conv2d(
            conv_dim,
            mask_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        weight_init.c2_xavier_fill(self.mask_features)
        
        self.maskformer_num_feature_levels = 3  # always use 3 scales
        self.common_stride = common_stride

        # extra fpn levels
        stride = min(self.transformer_feature_strides)
        self.num_fpn_levels = int(np.log2(stride) - np.log2(self.common_stride))

        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(self.feature_channels[:self.num_fpn_levels]):
            lateral_norm = get_norm(norm, conv_dim)
            lateral_conv = Conv2d(
                in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            self.add_module("adapter_{}".format(idx + 1), lateral_conv)
            weight_init.c2_xavier_fill(lateral_conv)
            lateral_convs.append(lateral_conv)

            output_norm = get_norm(norm, conv_dim)
            print("output_norm", output_norm)
            if (self.cfg.MODEL.DECODER_3D and self.cfg.DO_TRILINEAR_INTERPOLATION) or self.cfg.USE_CONV1D:
                output_conv = Conv2d(
                    conv_dim,
                    conv_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    norm=output_norm,
                    activation=F.relu,
                )
            else:
                output_conv = Conv2d(
                    conv_dim,
                    conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                    norm=output_norm,
                    activation=F.relu,
                )
            weight_init.c2_xavier_fill(output_conv)
            self.add_module("layer_{}".format(idx + 1), output_conv)
            output_convs.append(output_conv)
        
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

    def init_pe(self):
        N_steps = self.conv_dim // 2
        pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        return pe_layer

    def get_pos(self, x):
        return self.pe_layer(x)

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        ret = {}
        ret["input_shape"] = {
            k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
        }
        ret["conv_dim"] = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["norm"] = cfg.MODEL.SEM_SEG_HEAD.NORM
        ret["transformer_dropout"] = cfg.MODEL.MASK_FORMER.DROPOUT
        ret["transformer_nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        # ret["transformer_dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD
        ret["transformer_dim_feedforward"] = 1024  # use 1024 for deformable transformer encoder
        ret[
            "transformer_enc_layers"
        ] = cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS  # a separate config
        ret["transformer_in_features"] = cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES
        ret["common_stride"] = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE
        ret['cfg'] = cfg
        return ret

    @autocast(enabled=False)
    def forward_features(
        self, features, shape=None, x_xyz=None,
        multiview_data=None, mask_features_xyz=None,
        mask_features_p2v=None, scannet_pc=None, 
        scannet_p2v=None, decoder_3d=False,
    ):
        """
        Args:
            - features: dict,
                keys are ['res2', 'res3', 'res4', 'res5']
                    if self.cfg.MODEL.UPSAMPLE_FMAP 'res1' will also be there
                values are [(B*num_views, F_i, H_i, W_i)]
            - shape: the shape of original image, e.g. (B, 3, 256, 320)
            - x_xyz: [(B, num_views, H_i, W_i, 3)],  # xyz of res5:res3
            - poses: tensor (B, num_views, 4, 4)
            - intrinsics: tensor (B, num_views, 3, 3)
            - depths: tensor (B, num_views, H, W), same (H, W) as 'shape'
            - mask: padding mask, usually None
            - multiview_data: {
                'multi_scale_xyz': [(B, num_views, H_i, W_i, 3)],  # xyz
                'multi_scale_p2v': [(B, num_views * H_i * W_i)]  # point2voxel
            }  # order is from 'res5' to 'res2'
            - mask_features_xyz: tensor, (B, num_views, H_i, W_i, 3) res2 xyz

        Returns:
            - mask_features: (B*num_views, F_m, H_m, W_m),
                m is largest f_map (res2)
            - *
            - multi_scale_features: feats of small scales [res5, res4, res3]
        """
        srcs = []  # store feature maps from low to high resolution
        pos = []  # store respective pos embs

        # Project feature maps and compute pos embs
        for idx, f in enumerate(self.transformer_in_features[::-1]):
            x = features[f].float()  # deformable detr does not support half precision
            srcs.append(self.input_proj[idx](x))
            pos.append(self.get_pos(x))

        multi_scale_features = []

        # Multi-scale deform-attention for cross-scale interaction
        if not self.training and ((self.cfg.INPUT.MIN_SIZE_TEST > 500 and srcs[0].shape[0] > 100) or srcs[0].shape[0] > 150):
            max_bs = 100
            num_splits = int(np.ceil(srcs[0].shape[0] / max_bs))
            y, spatial_shapes, level_start_index = [], [], []
            for i in range(num_splits):
                srcs_ = [srcs[j][i * max_bs:(i + 1) * max_bs] for j in range(len(srcs))]
                pos_ = [pos[j][i * max_bs:(i + 1) * max_bs] for j in range(len(pos))]
                y_, spatial_shapes_, level_start_index_ = self.transformer(
                    srcs_, 
                    pos_,
                )
                y.append(y_)
            y = torch.cat(y, dim=0)
            spatial_shapes = spatial_shapes_
            level_start_index = level_start_index_
            bs = y.shape[0]

        else: 
            y, spatial_shapes, level_start_index = self.transformer(
                srcs, pos)
            bs = y.shape[0]
            
        
        # check for nans in y
        if torch.isnan(y).any():
            st()

        # Split again into multi-scale features (list)
        split_size_or_sections = [None] * self.transformer_num_feature_levels
        for i in range(self.transformer_num_feature_levels):
            if i < self.transformer_num_feature_levels - 1:
                split_size_or_sections[i] = level_start_index[i + 1] - level_start_index[i]
            else:
                split_size_or_sections[i] = y.shape[1] - level_start_index[i]
        y = torch.split(y, split_size_or_sections, dim=1)
        out = []
        for i, z in enumerate(y):
            out.append(z.transpose(1, 2).view(bs, -1, spatial_shapes[i][0], spatial_shapes[i][1]))

        # Cross-view contextualization
        if (decoder_3d or self.cfg.PASS_2D_CROSS_VIEW) and self.cfg.MODEL.CROSS_VIEW_CONTEXTUALIZE:
            out_new = []
            for i in range(self.transformer_num_feature_levels):
                mv_data = {}
                mv_data['multi_scale_p2v'] = [multiview_data['multi_scale_p2v'][i]]
                out_new.append(
                    self.cross_view_attn[i](
                        feature_list=[out[i]],
                        xyz_list=[x_xyz[i]],
                        shape=shape[:2],
                        multiview_data=mv_data,
                        voxelize=self.cfg.INPUT.VOXELIZE
                    )[0]
                )
            out = out_new
        
        # append `out` with extra FPN levels
        # Reverse feature maps into top-down order (from low to high resolution)
        if decoder_3d and self.cfg.USE_GHOST_POINTS:
            # interpolate features for skip connections
            assert self.num_fpn_levels == 1
            f = self.in_features[:self.num_fpn_levels][0]

            skip_feats = interpolate_feats_3d(
                features[f], mask_features_xyz, 
                mask_features_p2v, scannet_pc,
                scannet_p2v, shape[:2], num_neighbors=self.cfg.INTERP_NEIGHBORS,
                voxelize=self.cfg.INPUT.VOXELIZE,
                return_voxelized=True
            )[..., None]
            lateral_conv = self.lateral_convs[0]
            skip_feats = lateral_conv(skip_feats)


            cur_feats = interpolate_feats_3d(
                out[-1], multiview_data['multi_scale_xyz'][2],
                multiview_data['multi_scale_p2v'][2],
                scannet_pc, scannet_p2v,
                shape[:2], num_neighbors=self.cfg.INTERP_NEIGHBORS,
                voxelize=self.cfg.INPUT.VOXELIZE,
                return_voxelized=True
            )[..., None]
            y = cur_feats + skip_feats
            y = self.output_convs[0](y)
            out.append(y)
        else:
            for idx, f in enumerate(self.in_features[:self.num_fpn_levels][::-1]):
                x = features[f].float()
                lateral_conv = self.lateral_convs[idx]
                cur_fpn = lateral_conv(x)
                output_conv = self.output_convs[idx]

                # Following FPN implementation, we use nearest upsampling here
                if decoder_3d and self.cfg.DO_TRILINEAR_INTERPOLATION:
                    # print("doing trilinear interpolation")
                    shape_ = [shape[0], 1] if self.cfg.HIGH_RES_INPUT and self.training and self.cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS != -1 and self.cfg.HIGH_RES_SUBSAMPLE else shape[:2]
                    interp = interpolate_feats_3d(
                        out[-1], multiview_data['multi_scale_xyz'][len(out) - 1],
                        multiview_data['multi_scale_p2v'][len(out) - 1] if self.cfg.INPUT.VOXELIZE else None,
                        mask_features_xyz, mask_features_p2v,
                        shape_, num_neighbors=self.cfg.INTERP_NEIGHBORS,
                        voxelize=self.cfg.INPUT.VOXELIZE,
                    )
                else:
                    interp = F.interpolate(out[-1], size=cur_fpn.shape[-2:], mode="bilinear", align_corners=False)
                y = cur_fpn + interp
                y = output_conv(y)
                out.append(y)

        # Return the updated feature maps (res5,4,3) and projected res2
        num_cur_levels = 0
        for o in out:
            if num_cur_levels < self.maskformer_num_feature_levels:
                multi_scale_features.append(o)
                num_cur_levels += 1

        return self.mask_features(out[-1]), out[0], multi_scale_features
