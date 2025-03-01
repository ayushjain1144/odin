# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import logging
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_mean
import numpy as np
from torch.cuda.amp import autocast

from detectron2.config import configurable
from detectron2.layers import Conv2d
from detectron2.utils.registry import Registry

from odin.modeling.transformer_decoder.position_encoding import PositionEmbeddingSine3D, PositionEmbeddingLearned
from odin.modeling.meta_arch.self_cross_attention_layers import SelfAttentionLayer, CrossAttentionLayer, FFNLayer, MLP
from odin.modeling.meta_arch.language_encoder import LanguageEncoder
from odin.modeling.backproject.backproject import interpolate_feats_3d
from odin.data_video.sentence_utils import convert_grounding_to_od_logits_batched

import ipdb
st = ipdb.set_trace


TRANSFORMER_DECODER_REGISTRY = Registry("TRANSFORMER_MODULE")
TRANSFORMER_DECODER_REGISTRY.__doc__ = """
Registry for transformer module in MaskFormer.
"""

def build_transformer_decoder(cfg, in_channels, mask_classification=True):
    """
    Build a instance embedding branch from `cfg.MODEL.INS_EMBED_HEAD.NAME`.
    """
    name = cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME
    return TRANSFORMER_DECODER_REGISTRY.get(name)(
        cfg, in_channels, mask_classification)
    

@TRANSFORMER_DECODER_REGISTRY.register()
class ODINMultiScaleMaskedTransformerDecoder(nn.Module):

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
                if "static_query" in k:
                    newk = k.replace("static_query", "query_feat")
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
        in_channels,
        mask_classification=True,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
        # video related
        num_frames,
        decoder_3d: bool,
        cfg=None,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()
        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # self.num_frames = num_frames
        self.decoder_3d = decoder_3d
        self.hidden_dim = hidden_dim
        self.cfg = cfg
        
        self.pe_layer, self.pe_layer_2d = self.init_pe()
        
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_text_cross_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers_f_to_q = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=self.cfg.MODEL.MASK_FORMER.DROPOUT,
                    normalize_before=pre_norm,
                )
            )
            
            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=self.cfg.MODEL.MASK_FORMER.DROPOUT,
                    normalize_before=pre_norm,
                )
            )

            if self.cfg.MODEL.OPEN_VOCAB:
                self.transformer_text_cross_attention_layers.append(
                    CrossAttentionLayer(
                        d_model=hidden_dim,
                        nhead=nheads,
                        dropout=self.cfg.MODEL.MASK_FORMER.DROPOUT,
                        normalize_before=pre_norm,
                    )
                )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=self.cfg.MODEL.MASK_FORMER.DROPOUT,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries

        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        if self.mask_classification and not self.cfg.MODEL.OPEN_VOCAB:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)


        if self.cfg.MODEL.OPEN_VOCAB:
            self.lang_encoder = LanguageEncoder(cfg, d_model=hidden_dim)
            self.max_seq_len = cfg.MODEL.MAX_SEQ_LEN
            
            self.lang_pos_embed = nn.Embedding(self.max_seq_len, hidden_dim)

            self.class_embed = nn.Linear(hidden_dim, hidden_dim)


    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification
        
        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM

        ret["num_frames"] = cfg.INPUT.SAMPLING_FRAME_NUM
        ret["decoder_3d"] = cfg.MODEL.DECODER_3D
        ret["cfg"] = cfg

        return ret

    def init_pe(self):
        pe_layer_3d, pe_layer_2d = None, None
        N_steps = self.hidden_dim // 2
        if self.cfg.MODEL.DECODER_3D:
            pe_layer_3d = PositionEmbeddingLearned(
                    dim=3, num_pos_feats=self.hidden_dim
            )
        if self.cfg.MODEL.DECODER_2D:
            pe_layer_2d = PositionEmbeddingSine3D(
                N_steps, normalize=True, add_temporal=self.cfg.ADD_TEMPORAL)
        return pe_layer_3d, pe_layer_2d

    def voxel_map_to_source(self, voxel_map, poin2voxel):
        """
        Input:
            voxel_map (B, N1, C)
            point2voxel (B, N)
        Output:
            src_new (B, N, C)
        """
        bs, n, c = voxel_map.shape
        src_new = torch.stack([voxel_map[i, poin2voxel[i]] for i in range(bs)])
        return src_new


    def open_vocab_class_pred(self, decoder_output, text_feats, positive_map_od=None, num_classes=None):
        class_embed = self.class_embed(decoder_output)

        query_feats = F.normalize(class_embed, dim=-1)
        text_feats = F.normalize(text_feats, dim=-1)

        output_class = torch.einsum("bqc,sbc->bqs", query_feats / 0.07, text_feats) 
            
        output_class = convert_grounding_to_od_logits_batched(
            logits=output_class,
            num_class=num_classes,
            positive_map_od=torch.stack(positive_map_od).to(output_class.device, torch.long),
        )
        return output_class


    def forward(
        self, x, mask_features, shape=None,
        x_xyz=None, mask = None, mask_features_xyz=None,
        multiview_data=None, 
        segments=None, scannet_p2v=None, decoder_3d=False, 
        captions=None, positive_map_od=None, num_classes=None,
        scannet_pc=None):

        voxelize = decoder_3d and self.cfg.INPUT.VOXELIZE

        if shape is None:
            assert not decoder_3d
            shape = [x[0].shape[0], 1]
        bs, v  = shape
        pe_layer = self.pe_layer if decoder_3d else self.pe_layer_2d

        
        if not decoder_3d:
            bv, c_m, h_m, w_m = mask_features.shape
            mask_features = mask_features.view(bs, v, c_m, h_m, w_m)
            self.forward_prediction_heads = self.forward_prediction_heads2D
            if voxelize:
                mask_features = scatter_mean(
                    mask_features.permute(0, 1, 3, 4, 2).flatten(1, 3),
                    multiview_data['multi_scale_p2v'][-1],
                    dim=1,
                ) # b, n, c

        else:
            if ((self.cfg.HIGH_RES_INPUT and not self.training and not self.cfg.USE_GHOST_POINTS)) and self.cfg.INPUT.VOXELIZE or self.cfg.DO_FEATURE_INTERPOLATION_LATER:
                mask_features = mask_features.reshape(bs, v, -1, mask_features.shape[-2], mask_features.shape[-1]).permute(0, 1, 3, 4, 2).flatten(1, 3)
                mask_features = scatter_mean(
                    mask_features,
                    multiview_data['multi_scale_p2v'][-1],
                    dim=1,
                )[..., None].permute(0, 2, 1, 3) # b, c, n, 1
            
            if self.cfg.INPUT.VOXELIZE:
                assert mask_features.shape[-1] == 1, mask_features.shape
                mask_features = mask_features[..., 0]
                # voxelize (mask features are already voxelized)
                assert mask_features_xyz.shape[-2] == mask_features.shape[-1]
                if self.cfg.USE_SEGMENTS and not self.cfg.DO_FEATURE_INTERPOLATION_LATER:
                    mask_features = scatter_mean(
                        mask_features.permute(0, 2, 1),
                        segments, dim=1
                    ).permute(0, 2, 1) # B, C, N

                self.forward_prediction_heads = self.forward_prediction_heads3D
            else:
                bv, c_m, h_m, w_m = mask_features.shape
                mask_features = mask_features.view(bs, v, c_m, h_m, w_m)
                self.forward_prediction_heads = self.forward_prediction_heads2D

        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])

            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])
            _, c, hw = src[-1].shape
            # NTxCxHW => NxTxCxHW => (TxHW)xNxC
            src[-1] = src[-1].view(bs, v, c, hw).permute(1, 3, 0, 2).flatten(0, 1)

            if decoder_3d:
                pos.append(pe_layer(x_xyz[i].reshape(bs, -1, 3)).permute(1, 0, 2)) # THW X B X C
            else:
                pos.append(pe_layer(x[i].view(bs, v, -1, size_list[-1][0], size_list[-1][1]), None).flatten(3))
                pos[-1] = pos[-1].view(bs, v, c, hw).permute(1, 3, 0, 2).flatten(0, 1)

            if voxelize:
                p2v = multiview_data['multi_scale_p2v'][i]
                src[-1] = scatter_mean(src[-1], p2v.T, dim=0)
                
                # sometimes the pos can be big and lead to overflows
                with autocast(enabled=False):
                    pos[-1] = scatter_mean(pos[-1].float(), p2v.T, dim=0).to(pos[-1].dtype)

        # QxNxC
        # query pos
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)

        # query feats
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        query_pad_mask = None
        text_feats = None
        text_attn_mask = None
        if self.cfg.MODEL.OPEN_VOCAB:
            assert captions is not None
            text_feats, text_attn_mask = self.lang_encoder(captions) # B X S X C
            text_feats = text_feats.permute(1, 0, 2) # S X B X C

            # add these text features as text queries
            bs = output.shape[1]
            lang_pos_embed = self.lang_pos_embed.weight[:, None].repeat(1, bs, 1)[:text_feats.shape[0]]
        
        predictions_class = []
        predictions_mask = []

        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
            output, mask_features, attn_mask_target_size=size_list[0], 
            multiview_data=multiview_data, 
            next_level_index=0,
            voxelize=voxelize,
            mask_features_xyz=mask_features_xyz,
            segments=segments,
            scannet_p2v=scannet_p2v,
            decoder_3d=decoder_3d,
            text_feats=text_feats,
            positive_map_od=positive_map_od,
            num_classes=num_classes
            )

        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            # attention: cross-attention first

            if decoder_3d and self.cfg.SAMPLED_CROSS_ATTENTION and self.training:
                src_ = src[level_index]
                pos_ = pos[level_index]
                idx = torch.randperm(src_.shape[0], device=src_.device)[:self.cfg.SAMPLE_SIZES[level_index]]
                src_ = src_[idx]
                pos_ = pos_[idx]
                attn_mask = attn_mask[..., idx]
            else:
                src_ = src[level_index]
                pos_ = pos[level_index]

            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            if self.cfg.MODEL.OPEN_VOCAB:
                # attention to text tokens
                output = self.transformer_text_cross_attention_layers[i](
                    output, text_feats,
                    memory_mask=None,
                    memory_key_padding_mask=text_attn_mask, 
                    pos=lang_pos_embed, query_pos=query_embed
                )

            output = self.transformer_cross_attention_layers[i](
                output, src_,
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos_, query_pos=query_embed
            )
                
            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=query_pad_mask,
                query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
                output,
                mask_features,
                attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels],
                multiview_data=multiview_data,
                next_level_index=(i + 1) % self.num_feature_levels,
                voxelize=voxelize,
                mask_features_xyz=mask_features_xyz,
                segments=segments,
                scannet_p2v=scannet_p2v,
                decoder_3d=decoder_3d,
                text_feats=text_feats,
                positive_map_od=positive_map_od,
                num_classes=num_classes
                )

            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        assert len(predictions_class) == self.num_layers + 1
        
        if self.cfg.DO_FEATURE_INTERPOLATION_LATER:
            predictions_mask = [
                interpolate_feats_3d(
                    predictions_mask[i].permute(0, 2, 1),
                    mask_features_xyz,
                    torch.arange(mask_features_xyz.shape[1], device=mask_features_xyz.device)[None].repeat(bs, 1),
                    scannet_pc, 
                    scannet_p2v,
                    [bs, 1],
                    num_neighbors=self.cfg.INTERP_NEIGHBORS,
                    voxelize=self.cfg.INPUT.VOXELIZE,
                )
                for i in range(len(predictions_mask))
            ]
            
            if self.cfg.USE_SEGMENTS:
                with autocast(enabled=False):
                    predictions_mask = [
                        scatter_mean(
                            predictions_mask[i].permute(0, 2, 1).float(),
                            segments, dim=1
                        ).permute(0, 2, 1).to(predictions_mask[i].dtype)
                        for i in range(len(predictions_mask))
                    ]
            

        out = {
            'text_attn_mask': text_attn_mask,
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask
            )
        }
        return out

    def forward_prediction_heads3D(
        self, output, mask_features, attn_mask_target_size,
        multiview_data, next_level_index, voxelize=False, mask_features_xyz=None,
        segments=None, scannet_p2v=None, decoder_3d=False, 
        text_feats=None, positive_map_od=None, num_classes=None
        ):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)

        if self.cfg.MODEL.OPEN_VOCAB:
            outputs_class = self.open_vocab_class_pred(
                decoder_output, text_feats, positive_map_od, num_classes)

        else:
            outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)

        if self.cfg.USE_SEGMENTS and not self.cfg.DO_FEATURE_INTERPOLATION_LATER:
            segment_mask = torch.einsum("bqc,bcn->bqn", mask_embed, mask_features)
            output_mask = self.voxel_map_to_source(
                segment_mask.permute(0, 2, 1), segments
            ).permute(0, 2, 1)

        else:
            output_mask = torch.einsum("bqc,bcn->bqn", mask_embed, mask_features)
            segment_mask = None
        mask_features_p2v = torch.arange(output_mask.shape[2], device=mask_features.device).unsqueeze(0).repeat(output_mask.shape[0], 1)
        shape = multiview_data['multi_scale_xyz'][0].shape[:2]
        
        attn_mask = interpolate_feats_3d(
            output_mask.permute(0, 2, 1), mask_features_xyz,
            mask_features_p2v, multiview_data['multi_scale_xyz'][next_level_index],
            multiview_data['multi_scale_p2v'][next_level_index], shape=shape,
            num_neighbors=self.cfg.INTERP_NEIGHBORS,
            voxelize=True, 
            return_voxelized=True,
        )
        attn_mask = (attn_mask.sigmoid().unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        if self.cfg.USE_SEGMENTS and not self.cfg.DO_FEATURE_INTERPOLATION_LATER:
            output_mask = segment_mask

        return outputs_class, output_mask, attn_mask


    def forward_prediction_heads2D(
        self, output, mask_features, attn_mask_target_size,
        multiview_data, next_level_index, voxelize=False, mask_features_xyz=None,
        segments=None, scannet_p2v=None, decoder_3d=False, 
        text_feats=None, positive_map_od=None, num_classes=None
        ):
        assert decoder_3d == False or self.cfg.INPUT.VOXELIZE == False, "use forward_prediction_heads3D for 3D prediction by adding NO_POINTREND=True"
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        
        if self.cfg.MODEL.OPEN_VOCAB:
            outputs_class = self.open_vocab_class_pred(
                decoder_output, text_feats, positive_map_od, num_classes)
        else:
            outputs_class = self.class_embed(decoder_output)
            
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,btchw->bqthw", mask_embed, mask_features)
            
        b, q, t, _, _ = outputs_mask.shape

        # NOTE: prediction is of higher-resolution
        # [B, Q, T, H, W] -> [B, Q, T*H*W] -> [B, h, Q, T*H*W] -> [B*h, Q, T*HW]
        
        attn_mask = F.interpolate(outputs_mask.flatten(0, 1), size=attn_mask_target_size, mode="bilinear", align_corners=False).view(
            b, q, t, attn_mask_target_size[0], attn_mask_target_size[1])
            
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]
