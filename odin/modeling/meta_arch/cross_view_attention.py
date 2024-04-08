import torch
from torch import nn
from torch_scatter import scatter_mean

import libs.pointops2.functions.pointops as pointops

from odin.modeling.meta_arch.self_cross_attention_layers import CrossAttentionLayer, FFNLayer
from odin.modeling.transformer_decoder.position_encoding import PositionEmbeddingLearned, PositionEmbeddingLearnedMLP

from detectron2.utils.registry import Registry

CROSS_VIEW_PANET = Registry("CROSS_VIEW_PANET")
CROSS_VIEW_PANET.__doc__ = """
Registry for cross view panet attention module in MaskFormer.
"""

import ipdb
st = ipdb.set_trace


@CROSS_VIEW_PANET.register()
class CrossViewPAnet(nn.Module):
    def __init__(self, latent_dim, num_layers=6, nheads=8, nsample=16, dropout=0.0, dim_feedforward=None,
        cfg=None):
        super().__init__()
        self.cross_view_attention_layers = nn.ModuleList([
            CrossAttentionLayer(
                d_model=latent_dim,
                nhead=nheads,
                dropout=dropout,
                normalize_before=True,
                activation='relu',
            ) for _ in range(num_layers)
            ])
        if dim_feedforward is None:
            dim_feedforward = 4 * latent_dim
        self.ffn_layers = nn.ModuleList([
            FFNLayer(
                d_model=latent_dim,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                normalize_before=True,
                activation='relu',
            ) for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(latent_dim) for _ in range(num_layers)])
        self.nsample = nsample
        print(self.nsample)
        self.num_layers = num_layers
        self.cfg = cfg
        self.pe_layer = self.init_pe(latent_dim)

    def init_pe(self, latent_dim):
        if self.cfg.USE_MLP_POSITIONAL_ENCODING:
            pe_layer = PositionEmbeddingLearnedMLP(
                dim=3, num_pos_feats=latent_dim
        )
        else:
            pe_layer = PositionEmbeddingLearned(
                    dim=3, num_pos_feats=latent_dim
            )
        return pe_layer

    def encode_pe(self, xyz=None):
        return self.pe_layer(xyz)
        
    def forward(
        self, feature_list=None, xyz_list=None,
        shape=None, multiview_data=None, voxelize=None,
        ) -> torch.Tensor:
        """
        Args:
            feature_list: list of tensor (B*V, C, H, W)
            xyz_list: list of tensor (B*V, H, W, 3)
            shape: (B, V)
        """
        out_features = []
        bs, v = shape

        for j, (feature, xyz) in enumerate(zip(feature_list, xyz_list)):
            # B*V, F, H, W -> B, V, F, H, W -> B, V*H*W, F
            bv, f, h, w = feature.shape
            feature = feature.reshape(bs, v, f, h, w).permute(0, 1, 3, 4, 2).flatten(1, 3) # B, VHW, F
            xyz = xyz.reshape(bs, v, h, w, 3).flatten(1, 3) # B, VHW, 3
            if voxelize:
                p2v = multiview_data['multi_scale_p2v'][j] # B, N
                try:
                    feature = torch.cat(
                        [scatter_mean(feature[b], p2v[b], dim=0) for b in range(len(feature))]) # bn, F
                except:
                    st()
                xyz = torch.cat(
                    [scatter_mean(xyz[b], p2v[b], dim=0) for b in range(len(xyz))])
                batch_offset = ((p2v).max(1)[0] + 1).cumsum(0).to(torch.int32)
            else:
                # queryandgroup expects N, F and N, 3 with additional batch offset
                xyz = xyz.flatten(0, 1).contiguous()
                feature = feature.flatten(0, 1).contiguous()
                batch_offset = (torch.arange(bs, dtype=torch.int32, device=xyz.device) + 1) * v * h * w

            knn_points_feats, idx = pointops.queryandgroup(
                self.nsample, xyz, xyz, feature, None, batch_offset, batch_offset, use_xyz=True, return_indx=True
            ) # (B*n, nsample, 3+c)

            knn_points = knn_points_feats[..., 0:3] # B*N, nsample, 3
            knn_feats = knn_points_feats[..., 3:] # B*N, nsample, c
            
            # encode_pe expects B, N, 3
            query_pe = self.encode_pe(torch.zeros_like(xyz[:, None])).permute(1, 0, 2)
            knn_pe = self.encode_pe(knn_points).permute(1, 0, 2)

            output = feature[:, None] # B*N, 1, c

            bn, _, c = output.shape

            for i in range(self.num_layers):
                # get knn features from updated output
                key = output.flatten(0, 1)[idx.view(-1).long(), :].reshape(bn, self.nsample, c).permute(1, 0, 2)
                output = self.cross_view_attention_layers[i](
                    tgt=output.permute(1, 0, 2),
                    memory=key,
                    query_pos=query_pe,
                    pos=knn_pe,
                )
                output = self.ffn_layers[i](output).permute(1, 0, 2) 
                output = self.layer_norms[i](output) # new

            if voxelize:
                out_new = []
                idx = 0
                point2voxel = multiview_data['multi_scale_p2v'][j]
                output = output.squeeze(1)
                for i, b in enumerate(batch_offset):
                    out_new.append(output[idx:b][point2voxel[i]])
                    idx = b
                output = torch.stack(out_new, 0)
                    
            output = output.reshape(bs, v, h, w, c).permute(0, 1, 4, 2, 3).flatten(0, 1)
            out_features.append(output)
        return out_features