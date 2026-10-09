"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import math 
import copy 
import functools
from collections import OrderedDict

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.nn.init as init 
from typing import List

from .denoising import get_contrastive_denoising_training_group
from .utils import deformable_attention_core_func_v2, get_activation, inverse_sigmoid
from .utils import bias_init_with_prob

from ...core import register

__all__ = ['RTDETRTransformerv2']


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act='relu'):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MSDeformableAttention(nn.Module):
    def __init__(
        self, 
        embed_dim=256, 
        num_heads=8, 
        num_levels=4, 
        num_points=4, 
        method='default',
        offset_scale=0.5,
    ):
        """Multi-Scale Deformable Attention
        """
        super(MSDeformableAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.offset_scale = offset_scale

        if isinstance(num_points, list):
            assert len(num_points) == num_levels, ''
            num_points_list = num_points
        else:
            num_points_list = [num_points for _ in range(num_levels)]

        self.num_points_list = num_points_list
        
        num_points_scale = [1/n for n in num_points_list for _ in range(n)]
        self.register_buffer('num_points_scale', torch.tensor(num_points_scale, dtype=torch.float32))

        self.total_points = num_heads * sum(num_points_list)
        self.method = method

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.ms_deformable_attn_core = functools.partial(deformable_attention_core_func_v2, method=self.method) 

        self._reset_parameters()

        if method == 'discrete':
            for p in self.sampling_offsets.parameters():
                p.requires_grad = False

    def _reset_parameters(self):
        # sampling_offsets
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 2).tile([1, sum(self.num_points_list), 1])
        scaling = torch.concat([torch.arange(1, n + 1) for n in self.num_points_list]).reshape(1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        # attention_weights
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)

        # proj
        init.xavier_uniform_(self.value_proj.weight)
        init.constant_(self.value_proj.bias, 0)
        init.xavier_uniform_(self.output_proj.weight)
        init.constant_(self.output_proj.bias, 0)


    def forward(self,
                query: torch.Tensor,
                reference_points: torch.Tensor,
                value: torch.Tensor,
                value_spatial_shapes: List[int],
                value_mask: torch.Tensor=None):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]
        Len_v = value.shape[1]

        value = self.value_proj(value)
        if value_mask is not None:
            value = value * value_mask.to(value.dtype).unsqueeze(-1)

        value = value.reshape(bs, Len_v, self.num_heads, self.head_dim)

        sampling_offsets: torch.Tensor = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.reshape(bs, Len_q, self.num_heads, sum(self.num_points_list), 2)

        attention_weights = self.attention_weights(query).reshape(bs, Len_q, self.num_heads, sum(self.num_points_list))
        attention_weights = F.softmax(attention_weights, dim=-1).reshape(bs, Len_q, self.num_heads, sum(self.num_points_list))

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(1, 1, 1, self.num_levels, 1, 2)
            sampling_locations = reference_points.reshape(bs, Len_q, 1, self.num_levels, 1, 2) + sampling_offsets / offset_normalizer
        elif reference_points.shape[-1] == 4:
            # reference_points [8, 480, None, 1,  4]
            # sampling_offsets [8, 480, 8,    12, 2]
            num_points_scale = self.num_points_scale.to(dtype=query.dtype).unsqueeze(-1)
            offset = sampling_offsets * num_points_scale * reference_points[:, :, None, :, 2:] * self.offset_scale
            sampling_locations = reference_points[:, :, None, :, :2] + offset
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".
                format(reference_points.shape[-1]))

        output = self.ms_deformable_attn_core(value, value_spatial_shapes, sampling_locations, attention_weights, self.num_points_list)

        output = self.output_proj(output)

        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation='relu',
                 n_levels=4,
                 n_points=4,
                 cross_attn_method='default'):
        super(TransformerDecoderLayer, self).__init__()

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points, method=cross_attn_method)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = get_activation(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        
        self._reset_parameters()

    def _reset_parameters(self):
        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(self,
                target,
                reference_points,
                memory,
                memory_spatial_shapes,
                attn_mask=None,
                memory_mask=None,
                query_pos_embed=None):
        # self attention
        q = k = self.with_pos_embed(target, query_pos_embed)

        target2, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask)
        target = target + self.dropout1(target2)
        target = self.norm1(target)

        # cross attention
        target2 = self.cross_attn(\
            self.with_pos_embed(target, query_pos_embed), 
            reference_points, 
            memory, 
            memory_spatial_shapes, 
            memory_mask)
        target = target + self.dropout2(target2)
        target = self.norm2(target)

        # ffn
        target2 = self.forward_ffn(target)
        target = target + self.dropout4(target2)
        target = self.norm3(target)

        return target


class QueryP2DetailSampler(nn.Module):
    """Query-level P2 detail compensation for SA-P2Q.

    The module samples local P2 detail around each regular query reference box
    and applies a small-object gated residual to the query content.
    """

    def __init__(
        self,
        hidden_dim=256,
        num_points=9,
        sample_scale=0.75,
        small_object_scale=0.08,
        small_object_tau=0.02,
        gate_init=0.03,
        gate_max=0.15,
        eps=1e-6,
    ):
        super().__init__()
        assert num_points in (1, 5, 9), 'num_points must be 1, 5, or 9'
        if num_points == 1:
            offsets = [[0., 0.]]
        elif num_points == 5:
            offsets = [[0., 0.], [1., 0.], [-1., 0.], [0., 1.], [0., -1.]]
        else:
            offsets = [
                [-1., -1.], [0., -1.], [1., -1.],
                [-1.,  0.], [0.,  0.], [1.,  0.],
                [-1.,  1.], [0.,  1.], [1.,  1.],
            ]
        self.register_buffer('offsets', torch.tensor(offsets, dtype=torch.float32))
        self.hidden_dim = hidden_dim
        self.sample_scale = float(sample_scale)
        self.small_object_scale = float(small_object_scale)
        self.small_object_tau = float(small_object_tau)
        self.gate_max = float(gate_max)
        self.eps = eps

        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.detail_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gate_mlp = MLP(hidden_dim * 2 + 1, hidden_dim, 1, 2, act='gelu')

        init.xavier_uniform_(self.query_proj.weight)
        init.xavier_uniform_(self.detail_proj.weight)
        init.xavier_uniform_(self.out_proj.weight)
        init.constant_(self.query_proj.bias, 0.)
        init.constant_(self.detail_proj.bias, 0.)
        init.constant_(self.out_proj.bias, 0.)
        self.gate_max = min(max(self.gate_max, 2. * self.eps), 1.)
        gate_init = min(max(float(gate_init), self.eps), self.gate_max - self.eps)
        gate_prob = gate_init / self.gate_max
        init.constant_(self.gate_mlp.layers[-1].weight, 0.)
        init.constant_(self.gate_mlp.layers[-1].bias, math.log(gate_prob / (1. - gate_prob)))

    def forward(self, content, ref_boxes, p2_detail, detail_mask=None):
        if p2_detail is None or content.shape[1] == 0:
            return content

        ref_boxes = ref_boxes.clamp(0., 1.)
        centers = ref_boxes[..., :2]
        wh = ref_boxes[..., 2:].clamp_min(self.eps)
        scale = torch.sqrt((wh[..., :1] * wh[..., 1:2]).clamp_min(self.eps))
        small_gate = torch.sigmoid((self.small_object_scale - scale) / max(self.small_object_tau, self.eps))
        if detail_mask is not None:
            small_gate = small_gate * detail_mask.to(device=content.device, dtype=content.dtype).unsqueeze(-1)

        offsets = self.offsets.to(device=content.device, dtype=content.dtype)
        points = centers.unsqueeze(2) + offsets.view(1, 1, -1, 2) * wh.unsqueeze(2) * self.sample_scale
        points = points.clamp(0., 1.)

        B, N, K = points.shape[:3]
        grid = points.mul(2.).sub(1.).reshape(B, N * K, 1, 2)
        sampled = F.grid_sample(
            p2_detail.to(dtype=content.dtype),
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False,
        )
        sampled = sampled.reshape(B, self.hidden_dim, N, K).permute(0, 2, 3, 1).contiguous()

        q = self.query_proj(content).unsqueeze(2)
        v = self.detail_proj(sampled)
        attn = (q * v).sum(dim=-1) / math.sqrt(self.hidden_dim)
        attn = F.softmax(attn, dim=-1).unsqueeze(-1)
        detail = self.out_proj((attn * sampled).sum(dim=2))

        gate_input = torch.concat([content, detail, small_gate.to(dtype=content.dtype)], dim=-1)
        gate = self.gate_max * torch.sigmoid(self.gate_mlp(gate_input))
        return content + gate * small_gate.to(dtype=content.dtype) * detail


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim,
        decoder_layer,
        num_layers,
        eval_idx=-1,
        num_queries=300,
        nhead=8,
        use_sa_p2q_detail=False,
        sa_p2q_detail_num_points=9,
        sa_p2q_detail_sample_scale=0.75,
        sa_p2q_detail_small_scale=0.08,
        sa_p2q_detail_small_tau=0.02,
        sa_p2q_detail_gate_init=0.03,
        sa_p2q_detail_gate_max=0.15,
    ):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.num_queries = num_queries
        self.nhead = nhead
        self.use_sa_p2q_detail = use_sa_p2q_detail

        if use_sa_p2q_detail:
            self.p2_detail_sampler = QueryP2DetailSampler(
                hidden_dim=hidden_dim,
                num_points=sa_p2q_detail_num_points,
                sample_scale=sa_p2q_detail_sample_scale,
                small_object_scale=sa_p2q_detail_small_scale,
                small_object_tau=sa_p2q_detail_small_tau,
                gate_init=sa_p2q_detail_gate_init,
                gate_max=sa_p2q_detail_gate_max,
            )
        else:
            self.p2_detail_sampler = None

    def forward(self,
                target,
                ref_points_unact,
                memory,
                memory_spatial_shapes,
                bbox_head,
                score_head,
                query_pos_head,
                attn_mask=None,
                memory_mask=None,
                p2_detail=None,
                p2_query_mask=None):
        dec_out_bboxes = []
        dec_out_logits = []
        ref_points_detach = F.sigmoid(ref_points_unact)

        output = target
        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach)
            layer_attn_mask = attn_mask
            total_queries = output.shape[1]
            regular_start = max(total_queries - self.num_queries, 0)
            if self.use_sa_p2q_detail and self.p2_detail_sampler is not None and p2_detail is not None:
                regular_output = self.p2_detail_sampler(
                    output[:, regular_start:],
                    ref_points_detach[:, regular_start:],
                    p2_detail,
                    detail_mask=p2_query_mask,
                )
                if regular_start > 0:
                    output = torch.concat([output[:, :regular_start], regular_output], dim=1)
                else:
                    output = regular_output

            output = layer(output, ref_points_input, memory, memory_spatial_shapes, layer_attn_mask, memory_mask, query_pos_embed)

            inter_ref_bbox = F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points_detach))

            if self.training:
                dec_out_logits.append(score_head[i](output))
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    dec_out_bboxes.append(F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points)))

            elif i == self.eval_idx:
                dec_out_logits.append(score_head[i](output))
                dec_out_bboxes.append(inter_ref_bbox)
                break

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach()

        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits)


@register()
class RTDETRTransformerv2(nn.Module):
    __share__ = ['num_classes', 'eval_spatial_size']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_levels=3,
                 num_points=4,
                 nhead=8,
                 num_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 learn_query_content=False,
                 eval_spatial_size=None,
                 eval_idx=-1,
                 eps=1e-2,
                 aux_loss=True,
                 cross_attn_method='default',
                 query_select_method='default',
                 use_sa_p2q=False,
                 sa_p2q_recall_weight=0.08,
                 sa_p2q_recall_min_weight=0.01,
                 sa_p2q_recall_detach=True,
                 sa_p2q_recall_eps=1e-4,
                 sa_p2q_recall_clip=2.0,
                 sa_p2q_base_stride=8,
                 sa_p2q_p2_min_low=12,
                 sa_p2q_p2_min_high=48,
                 sa_p2q_p2_max_queries=64,
                 sa_p2q_p2_recall_mass_topk=300,
                 sa_p2q_p2_recall_mass_low=0.03,
                 sa_p2q_p2_recall_mass_high=0.15,
                 sa_p2q_refill_margin=0.12,
                 sa_p2q_level_calibration_weight=0.5,
                 sa_p2q_prior_warm_iters=65000,
                 sa_p2q_prior_anneal_iters=32000,
                 use_sa_p2q_detail=True,
                 sa_p2q_detail_num_points=9,
                 sa_p2q_detail_sample_scale=0.75,
                 sa_p2q_detail_small_scale=0.08,
                 sa_p2q_detail_small_tau=0.02,
                 sa_p2q_detail_gate_init=0.03,
                 sa_p2q_detail_gate_max=0.15):
        super().__init__()
        feat_strides = list(feat_strides)
        self.use_sa_p2q = use_sa_p2q
        if use_sa_p2q:
            assert len(feat_channels) + 1 == num_levels, \
                "SCB-P2Q expects decoder levels [S3, S4, S5, P2]."
            assert len(feat_strides) == num_levels, \
                "SCB-P2Q feat_strides must be ordered as [8, 16, 32, 4]."
        else:
            assert len(feat_channels) <= num_levels
            assert len(feat_strides) == len(feat_channels)
            for _ in range(num_levels - len(feat_strides)):
                feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_layers = num_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss

        assert query_select_method in ('default', 'one2many', 'agnostic'), ''
        assert cross_attn_method in ('default', 'discrete'), ''
        self.cross_attn_method = cross_attn_method
        self.query_select_method = query_select_method
        self.sa_p2q_recall_weight = float(sa_p2q_recall_weight)
        self.sa_p2q_recall_min_weight = float(sa_p2q_recall_min_weight)
        self.sa_p2q_recall_detach = bool(sa_p2q_recall_detach)
        self.sa_p2q_recall_eps = float(sa_p2q_recall_eps)
        self.sa_p2q_recall_clip = float(sa_p2q_recall_clip)
        self.sa_p2q_base_stride = float(sa_p2q_base_stride)
        self.sa_p2q_p2_min_low = int(sa_p2q_p2_min_low)
        self.sa_p2q_p2_min_high = int(sa_p2q_p2_min_high)
        self.sa_p2q_p2_max_queries = int(sa_p2q_p2_max_queries)
        self.sa_p2q_p2_recall_mass_topk = int(sa_p2q_p2_recall_mass_topk)
        self.sa_p2q_p2_recall_mass_low = float(sa_p2q_p2_recall_mass_low)
        self.sa_p2q_p2_recall_mass_high = float(sa_p2q_p2_recall_mass_high)
        self.sa_p2q_refill_margin = float(sa_p2q_refill_margin)
        self.sa_p2q_level_calibration_weight = float(sa_p2q_level_calibration_weight)
        self.sa_p2q_prior_warm_iters = int(sa_p2q_prior_warm_iters)
        self.sa_p2q_prior_anneal_iters = int(sa_p2q_prior_anneal_iters)
        self._last_sa_p2q_p2_query_start = 0
        self._last_sa_p2q_p2_query_count = 0
        self._last_sa_p2q_p2_query_mean = 0.0
        self._last_sa_p2q_level_count_mean = []
        self._last_sa_p2q_prior_weight = 0.0
        self._last_sa_p2q_q_min_mean = 0.0
        self.register_buffer('_sa_p2q_step', torch.zeros((), dtype=torch.long), persistent=False)
        if use_sa_p2q:
            assert 0 <= self.sa_p2q_p2_min_low <= self.sa_p2q_p2_min_high <= num_queries, \
                "SCB-P2Q p2 min band must satisfy 0 <= low <= high <= num_queries."
            assert 0 <= self.sa_p2q_p2_max_queries <= num_queries, \
                "SCB-P2Q p2 max queries must be in [0, num_queries]."
            assert self.sa_p2q_p2_min_high <= self.sa_p2q_p2_max_queries, \
                "SCB-P2Q p2 min high should not exceed p2 max queries."

        # backbone feature projection
        self._build_input_proj_layer(feat_channels)

        # Transformer module
        decoder_layer = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, \
            activation, num_levels, num_points, cross_attn_method=cross_attn_method)
        self.decoder = TransformerDecoder(
            hidden_dim,
            decoder_layer,
            num_layers,
            eval_idx,
            num_queries=num_queries,
            nhead=nhead,
            use_sa_p2q_detail=use_sa_p2q and use_sa_p2q_detail,
            sa_p2q_detail_num_points=sa_p2q_detail_num_points,
            sa_p2q_detail_sample_scale=sa_p2q_detail_sample_scale,
            sa_p2q_detail_small_scale=sa_p2q_detail_small_scale,
            sa_p2q_detail_small_tau=sa_p2q_detail_small_tau,
            sa_p2q_detail_gate_init=sa_p2q_detail_gate_init,
            sa_p2q_detail_gate_max=sa_p2q_detail_gate_max,
        )

        # denoising
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        if num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(num_classes+1, hidden_dim, padding_idx=num_classes)
            init.normal_(self.denoising_class_embed.weight[:-1])

        # decoder embedding
        self.learn_query_content = learn_query_content
        if learn_query_content:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, 2)

        # if num_select_queries != self.num_queries:
        #     layer = TransformerEncoderLayer(hidden_dim, nhead, dim_feedforward, activation='gelu')
        #     self.encoder = TransformerEncoder(layer, 1)

        self.enc_output = nn.Sequential(OrderedDict([
            ('proj', nn.Linear(hidden_dim, hidden_dim)),
            ('norm', nn.LayerNorm(hidden_dim,)),
        ]))

        if query_select_method == 'agnostic':
            self.enc_score_head = nn.Linear(hidden_dim, 1)
        else:
            self.enc_score_head = nn.Linear(hidden_dim, num_classes)

        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3)

        # decoder head
        self.dec_score_head = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes) for _ in range(num_layers)
        ])
        self.dec_bbox_head = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 4, 3) for _ in range(num_layers)
        ])

        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            anchors, valid_mask = self._generate_anchors()
            self.register_buffer('anchors', anchors)
            self.register_buffer('valid_mask', valid_mask)

        self._reset_parameters()

    def _reset_parameters(self):
        bias = bias_init_with_prob(0.01)
        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)

        for _cls, _reg in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(_cls.bias, bias)
            init.constant_(_reg.layers[-1].weight, 0)
            init.constant_(_reg.layers[-1].bias, 0)

        init.xavier_uniform_(self.enc_output[0].weight)
        if self.learn_query_content:
            init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)
        for m in self.input_proj:
            init.xavier_uniform_(m[0].weight)

    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            self.input_proj.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)),
                    ('norm', nn.BatchNorm2d(self.hidden_dim,))])
                )
            )

        in_channels = feat_channels[-1]

        extra_levels = 0 if self.use_sa_p2q else self.num_levels - len(feat_channels)
        for _ in range(extra_levels):
            self.input_proj.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)),
                    ('norm', nn.BatchNorm2d(self.hidden_dim))])
                )
            )
            in_channels = self.hidden_dim

    def _get_encoder_input(self, feats: List[torch.Tensor], p2_query=None, p2_stride=None):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if (not self.use_sa_p2q) and self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
        if self.use_sa_p2q:
            assert p2_query is not None, "SCB-P2Q requires p2_query from the encoder."
            _, _, h, w = p2_query.shape
            feat_flatten.append(p2_query.flatten(2).permute(0, 2, 1))
            spatial_shapes.append([h, w])
        # [b, l, c]
        feat_flatten = torch.concat(feat_flatten, 1)
        return feat_flatten, spatial_shapes

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,
                          dtype=torch.float32,
                          device='cpu'):
        if spatial_shapes is None:
            spatial_shapes = []
            eval_h, eval_w = self.eval_spatial_size
            for s in self.feat_strides:
                spatial_shapes.append([int(eval_h / s), int(eval_w / s)])

        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], dim=-1)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / torch.tensor([w, h], dtype=dtype)
            if self.use_sa_p2q:
                stride = float(self.feat_strides[lvl])
                wh_scale = stride / self.sa_p2q_base_stride
            else:
                wh_scale = 2.0 ** lvl
            wh = torch.ones_like(grid_xy) * grid_size * wh_scale
            lvl_anchors = torch.concat([grid_xy, wh], dim=-1).reshape(-1, h * w, 4)
            anchors.append(lvl_anchors)

        anchors = torch.concat(anchors, dim=1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.inf)

        return anchors, valid_mask

    def _get_query_base_score(self, outputs_logits: torch.Tensor):
        if self.query_select_method == 'agnostic':
            return outputs_logits.squeeze(-1)
        return outputs_logits.max(-1).values

    def _get_sa_p2q_prior_weight(self):
        if not self.training:
            return self.sa_p2q_recall_min_weight
        if self.sa_p2q_prior_anneal_iters <= 0:
            return self.sa_p2q_recall_weight
        step = int(self._sa_p2q_step.item())
        if step < self.sa_p2q_prior_warm_iters:
            factor = 1.0
        else:
            progress = min(
                1.0,
                max(0.0, (step - self.sa_p2q_prior_warm_iters) / float(self.sa_p2q_prior_anneal_iters)),
            )
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.sa_p2q_recall_min_weight + \
            (self.sa_p2q_recall_weight - self.sa_p2q_recall_min_weight) * factor

    def _get_sa_p2q_prior(self, p2_recall_logits, spatial_shapes, dtype):
        if (not self.use_sa_p2q) or p2_recall_logits is None:
            return None, None, None, None

        recall = p2_recall_logits.detach() if self.sa_p2q_recall_detach else p2_recall_logits
        p2_h, p2_w = spatial_shapes[-1]
        if recall.shape[-2:] != (p2_h, p2_w):
            recall = F.interpolate(recall, size=(p2_h, p2_w), mode='bilinear', align_corners=False)

        p2_prob = recall.sigmoid().to(dtype=dtype).flatten(1)
        phi = recall.to(dtype=dtype).flatten(1)
        phi = phi - phi.mean(dim=1, keepdim=True)
        phi = phi / phi.std(dim=1, keepdim=True, unbiased=False).clamp_min(self.sa_p2q_recall_eps)
        phi = phi.clamp(-self.sa_p2q_recall_clip, self.sa_p2q_recall_clip)

        prefix_len = sum(int(h) * int(w) for h, w in spatial_shapes[:-1])
        total_len = prefix_len + int(p2_h) * int(p2_w)
        prior = phi.new_zeros(phi.shape[0], total_len)
        p2_mask = phi.new_zeros(phi.shape[0], total_len)
        prior[:, prefix_len:] = phi
        p2_mask[:, prefix_len:] = 1.0
        return prior, p2_mask, prefix_len, p2_prob

    def _get_level_calibrated_score(self, base_score: torch.Tensor, spatial_shapes):
        eta = self.sa_p2q_level_calibration_weight
        if (not self.use_sa_p2q) or eta <= 0:
            return base_score

        global_mean = base_score.mean(dim=1, keepdim=True)
        global_std = base_score.std(dim=1, keepdim=True, unbiased=False).clamp_min(self.sa_p2q_recall_eps)
        calibrated = base_score.clone()
        start = 0
        for h, w in spatial_shapes:
            length = int(h) * int(w)
            level_score = base_score[:, start:start + length]
            level_mean = level_score.mean(dim=1, keepdim=True)
            level_std = level_score.std(dim=1, keepdim=True, unbiased=False).clamp_min(self.sa_p2q_recall_eps)
            level_norm = global_mean + global_std * (level_score - level_mean) / level_std
            calibrated[:, start:start + length] = level_score + eta * (level_norm - level_score)
            start += length
        return calibrated

    def _get_sa_p2q_dynamic_min(self, p2_prob: torch.Tensor):
        if p2_prob is None:
            return None
        k = min(max(self.sa_p2q_p2_recall_mass_topk, 1), p2_prob.shape[1])
        recall_mass = torch.topk(p2_prob, k, dim=1).values.mean(dim=1)
        denom = max(
            self.sa_p2q_p2_recall_mass_high - self.sa_p2q_p2_recall_mass_low,
            self.sa_p2q_recall_eps,
        )
        alpha = ((recall_mass - self.sa_p2q_p2_recall_mass_low) / denom).clamp(0., 1.)
        q_min = self.sa_p2q_p2_min_low + \
            (self.sa_p2q_p2_min_high - self.sa_p2q_p2_min_low) * alpha
        q_min = q_min.round().to(torch.long)
        q_min = q_min.clamp(0, min(self.sa_p2q_p2_max_queries, self.num_queries))
        return q_min

    @staticmethod
    def _level_slices(spatial_shapes):
        slices = []
        start = 0
        for h, w in spatial_shapes:
            length = int(h) * int(w)
            slices.append((start, start + length))
            start += length
        return slices


    def _get_decoder_input(self,
                           memory: torch.Tensor,
                           spatial_shapes,
                           p2_recall_logits=None,
                           denoising_logits=None,
                           denoising_bbox_unact=None):

        # prepare input for decoder
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors = self.anchors
            valid_mask = self.valid_mask

        # memory = torch.where(valid_mask, memory, 0)
        # TODO fix type error for onnx export
        memory = valid_mask.to(memory.dtype) * memory

        output_memory :torch.Tensor = self.enc_output(memory)
        enc_outputs_logits :torch.Tensor = self.enc_score_head(output_memory)
        enc_outputs_coord_unact :torch.Tensor = self.enc_bbox_head(output_memory) + anchors
        recall_prior, recall_p2_mask, recall_p2_start, p2_prob = self._get_sa_p2q_prior(
            p2_recall_logits,
            spatial_shapes,
            enc_outputs_logits.dtype,
        )

        enc_topk_bboxes_list, enc_topk_logits_list = [], []
        enc_topk_memory, enc_topk_logits, enc_topk_bbox_unact, topk_p2_mask = \
            self._select_topk(
                output_memory,
                enc_outputs_logits,
                enc_outputs_coord_unact,
                self.num_queries,
                recall_prior=recall_prior,
                recall_p2_mask=recall_p2_mask,
                recall_p2_start=recall_p2_start,
                p2_prob=p2_prob,
                spatial_shapes=spatial_shapes,
            )

        if self.training:
            enc_topk_bboxes = F.sigmoid(enc_topk_bbox_unact)
            enc_topk_bboxes_list.append(enc_topk_bboxes)
            enc_topk_logits_list.append(enc_topk_logits)

        # if self.num_select_queries != self.num_queries:
        #     raise NotImplementedError('')

        if self.learn_query_content:
            content = self.tgt_embed.weight.unsqueeze(0).tile([memory.shape[0], 1, 1])
        else:
            content = enc_topk_memory.detach()

        enc_topk_bbox_unact = enc_topk_bbox_unact.detach()

        if denoising_bbox_unact is not None:
            enc_topk_bbox_unact = torch.concat([denoising_bbox_unact, enc_topk_bbox_unact], dim=1)
            content = torch.concat([denoising_logits, content], dim=1)

        return content, enc_topk_bbox_unact, enc_topk_bboxes_list, enc_topk_logits_list, topk_p2_mask

    def _select_topk(self,
                     memory: torch.Tensor,
                     outputs_logits: torch.Tensor,
                     outputs_coords_unact: torch.Tensor,
                     topk: int,
                     recall_prior: torch.Tensor=None,
                     recall_p2_mask: torch.Tensor=None,
                     recall_p2_start: int=None,
                     p2_prob: torch.Tensor=None,
                     spatial_shapes=None):
        self._last_sa_p2q_p2_query_start = 0
        self._last_sa_p2q_p2_query_count = 0
        self._last_sa_p2q_p2_query_mean = 0.0
        self._last_sa_p2q_level_count_mean = []
        self._last_sa_p2q_prior_weight = 0.0
        self._last_sa_p2q_q_min_mean = 0.0

        if self.use_sa_p2q and recall_prior is not None and recall_p2_start is not None:
            base_score = self._get_query_base_score(outputs_logits)
            base_score = self._get_level_calibrated_score(base_score, spatial_shapes)
            prior_weight = self._get_sa_p2q_prior_weight()
            self._last_sa_p2q_prior_weight = float(prior_weight)
            selection_score = base_score + prior_weight * recall_prior
            _, topk_ind = torch.topk(selection_score, topk, dim=-1)

            p2_mask_bool = recall_p2_mask > 0
            p2_max = min(self.sa_p2q_p2_max_queries, topk)
            q_min = self._get_sa_p2q_dynamic_min(p2_prob)
            if q_min is None:
                q_min = torch.zeros(selection_score.shape[0], dtype=torch.long, device=selection_score.device)

            final_indices = []
            p2_counts = []
            q_min_values = []
            level_count_values = []
            level_slices = self._level_slices(spatial_shapes)
            min_value = torch.finfo(selection_score.dtype).min
            for b in range(selection_score.shape[0]):
                ind_b = topk_ind[b]
                score_b = selection_score[b]
                p2_selected = p2_mask_bool[b, ind_b]
                p2_count = int(p2_selected.sum().item())
                q_min_b = min(int(q_min[b].item()), p2_max, topk)

                if p2_count > p2_max:
                    p2_ind = ind_b[p2_selected]
                    _, keep_order = torch.topk(score_b[p2_ind], p2_max, dim=0)
                    keep_p2 = p2_ind[keep_order]
                    fill_score = score_b.masked_fill(p2_mask_bool[b], min_value)
                    _, fill_main = torch.topk(fill_score, topk - p2_max, dim=0)
                    final_ind = torch.cat([keep_p2, fill_main], dim=0)
                    p2_count = p2_max
                else:
                    final_ind = ind_b

                if p2_count < q_min_b:
                    selected = torch.zeros_like(score_b, dtype=torch.bool)
                    selected.scatter_(0, final_ind, True)
                    available_p2 = p2_mask_bool[b] & (~selected)
                    candidate_num = min(q_min_b - p2_count, int(available_p2.sum().item()))
                    nonp2_selected = final_ind[~p2_mask_bool[b, final_ind]]

                    if candidate_num > 0 and nonp2_selected.numel() > 0:
                        candidate_score = score_b.masked_fill(~available_p2, min_value)
                        cand_score, cand_ind = torch.topk(candidate_score, candidate_num, dim=0)
                        replace_order = torch.argsort(score_b[nonp2_selected], descending=False)
                        replace_ind = nonp2_selected[replace_order]
                        final_list = final_ind.clone()

                        max_pairs = min(cand_ind.numel(), replace_ind.numel())
                        for j in range(max_pairs):
                            cand = cand_ind[j]
                            rep = replace_ind[j]
                            if cand_score[j] < score_b[rep] - self.sa_p2q_refill_margin:
                                break
                            pos = torch.nonzero(final_list == rep, as_tuple=False)
                            if pos.numel() == 0:
                                continue
                            final_list[pos[0, 0]] = cand
                            p2_count += 1
                        final_ind = final_list

                order = torch.argsort(score_b[final_ind], descending=True)
                final_ind = final_ind[order]
                final_indices.append(final_ind)
                p2_counts.append(int(p2_mask_bool[b, final_ind].sum().item()))
                q_min_values.append(q_min_b)

                level_counts = []
                for start, end in level_slices:
                    level_counts.append(int(((final_ind >= start) & (final_ind < end)).sum().item()))
                level_count_values.append(level_counts)

            topk_ind = torch.stack(final_indices, dim=0)
            self._last_sa_p2q_p2_query_count = max(p2_counts) if len(p2_counts) > 0 else 0
            self._last_sa_p2q_p2_query_mean = float(sum(p2_counts) / max(len(p2_counts), 1))
            self._last_sa_p2q_q_min_mean = float(sum(q_min_values) / max(len(q_min_values), 1))
            if len(level_count_values) > 0:
                n_levels = len(level_count_values[0])
                self._last_sa_p2q_level_count_mean = [
                    float(sum(counts[i] for counts in level_count_values) / len(level_count_values))
                    for i in range(n_levels)
                ]
            self._last_sa_p2q_p2_query_start = -1

        elif self.query_select_method == 'default':
            _, topk_ind = torch.topk(outputs_logits.max(-1).values, topk, dim=-1)

        elif self.query_select_method == 'one2many':
            _, topk_ind = torch.topk(outputs_logits.flatten(1), topk, dim=-1)
            topk_ind = topk_ind // self.num_classes

        elif self.query_select_method == 'agnostic':
            _, topk_ind = torch.topk(outputs_logits.squeeze(-1), topk, dim=-1)

        topk_ind: torch.Tensor

        topk_coords = outputs_coords_unact.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_coords_unact.shape[-1]))

        topk_logits = outputs_logits.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_logits.shape[-1]))

        topk_memory = memory.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, memory.shape[-1]))

        topk_p2_mask = None
        if recall_p2_mask is not None:
            topk_p2_mask = recall_p2_mask.gather(dim=1, index=topk_ind)

        return topk_memory, topk_logits, topk_coords, topk_p2_mask


    def forward(self, feats, targets=None, p2_query=None,
                p2_detail=None, p2_stride=None, p2_recall_logits=None):
        # input projection and embedding
        memory, spatial_shapes = self._get_encoder_input(feats, p2_query=p2_query, p2_stride=p2_stride)

        # prepare denoising training
        if self.training and self.num_denoising > 0:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(targets, \
                    self.num_classes,
                    self.num_queries,
                    self.denoising_class_embed,
                    num_denoising=self.num_denoising,
                    label_noise_ratio=self.label_noise_ratio,
                    box_noise_scale=self.box_noise_scale, )
        else:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        init_ref_contents, init_ref_points_unact, enc_topk_bboxes_list, enc_topk_logits_list, topk_p2_mask = \
            self._get_decoder_input(
                memory,
                spatial_shapes,
                p2_recall_logits=p2_recall_logits,
                denoising_logits=denoising_logits,
                denoising_bbox_unact=denoising_bbox_unact,
            )

        # decoder
        out_bboxes, out_logits = self.decoder(
            init_ref_contents,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
            p2_detail=p2_detail,
            p2_query_mask=topk_p2_mask)

        if self.training and dn_meta is not None:
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)

        out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1]}
        if self.training and self.use_sa_p2q:
            out['sa_p2q_p2_query_start'] = self._last_sa_p2q_p2_query_start
            out['sa_p2q_p2_query_count'] = self._last_sa_p2q_p2_query_count
            stat_device = out_logits[-1].device
            out['sa_p2q_p2_query_count_mean'] = torch.tensor(
                self._last_sa_p2q_p2_query_mean, device=stat_device)
            out['sa_p2q_q_min_mean'] = torch.tensor(
                self._last_sa_p2q_q_min_mean, device=stat_device)
            out['sa_p2q_prior_weight'] = torch.tensor(
                self._last_sa_p2q_prior_weight, device=stat_device)
            for idx, value in enumerate(self._last_sa_p2q_level_count_mean):
                out[f'sa_p2q_level_{idx}_count_mean'] = torch.tensor(value, device=stat_device)
            self._sa_p2q_step += 1

        if self.training and self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1])
            out['enc_aux_outputs'] = self._set_aux_loss(enc_topk_logits_list, enc_topk_bboxes_list)
            out['enc_meta'] = {'class_agnostic': self.query_select_method == 'agnostic'}

            if dn_meta is not None:
                out['dn_aux_outputs'] = self._set_aux_loss(dn_out_logits, dn_out_bboxes)
                out['dn_meta'] = dn_meta

        return out


    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class, outputs_coord)]
