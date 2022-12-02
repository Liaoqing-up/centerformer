# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

from ..functions import MSDeformAttnFunction
from einops import rearrange, repeat


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, d_head = 64, n_levels=4, n_heads=8, n_points=4, out_sample_loc=False):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        # if d_model % n_heads != 0:
        #     raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        # _d_per_head = d_model // n_heads
        # # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        # if not _is_power_of_2(_d_per_head):
        #     warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
        #                   "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64

        self.d_model = d_model
        self.d_head = d_head
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.out_sample_loc = out_sample_loc

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_head*n_heads)
        self.output_proj = nn.Linear(d_head*n_heads, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_head)
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1).to(sampling_offsets)

            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        output = MSDeformAttnFunction.apply(
            value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        output = self.output_proj(output)
        if self.out_sample_loc:
            return output, torch.cat((sampling_locations,attention_weights[:,:,:,:,:,None]),dim=-1)
        else:
            return output, None


class CSDeformAttn(nn.Module):
    def __init__(self, d_model=256, d_head = 64, n_levels=4, n_heads=8, n_points=4, out_sample_loc=False):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        # if d_model % n_heads != 0:
        #     raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        # _d_per_head = d_model // n_heads
        # # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        # if not _is_power_of_2(_d_per_head):
        #     warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
        #                   "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64

        self.d_model = d_model
        self.d_head = d_head
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.scale = d_head ** -0.5

        self.out_sample_loc = out_sample_loc

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        # self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.query_proj = nn.Linear(d_model, d_head*n_heads)
        self.key_proj_heads = nn.ModuleList([nn.Linear(d_model, d_head) for _ in range(n_heads)])
        self.value_proj_heads = nn.ModuleList([nn.Linear(d_model, d_head) for _ in range(n_heads)])
        self.attend = nn.Softmax(dim=-1)
        self.output_proj = nn.Linear(d_head*n_heads, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        # constant_(self.attention_weights.weight.data, 0.)
        # constant_(self.attention_weights.bias.data, 0.)
        for key_proj in self.key_proj_heads:
            xavier_uniform_(key_proj.weight.data)
            constant_(key_proj.bias.data, 0.)
        for value_proj in self.value_proj_heads:
            xavier_uniform_(value_proj.weight.data)
            constant_(value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, input_query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        """
        :param input_query                       (N, Length_{input_query}, C)
        :param reference_points            (N, Length_{input_query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{input_query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{input_query}, C)
        """
        N, Len_q, _ = input_query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in


        # if input_padding_mask is not None:
        #     value = value.masked_fill(input_padding_mask[..., None], float(0))
        ## [1, 500, 6, 4, 15, 2]
        sampling_offsets = self.sampling_offsets(input_query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)

        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1).to(sampling_offsets)

            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        ##todo: gather the key and value input_flatten feature,  points_sample=15, dim=256
        ##todo(chenzhao): need refactor this part because of the inefficient 'for' loop
        sampling_input_features_list = []
        for lid_, (H, W) in enumerate(input_spatial_shapes):
            input_features_ = input_flatten[:, input_level_start_index[lid_]:input_level_start_index[lid_]+H*W, :].reshape(
                N, H, W, input_flatten.shape[-1]
            ).unsqueeze(1).repeat(1, self.n_heads, 1, 1, 1)       ## B H W dim  -> B h H W dim
            input_features_ = rearrange(input_features_, 'b n h w c -> (b n) c h w')    ## N_*M_, D_, H_, W_
            sampling_grid_l_ = sampling_locations[:,:,:,lid_].transpose(1, 2).flatten(0, 1) ## N_*M_, Lq_, P_, 2
            sampling__input_features_l_ = F.grid_sample(input_features_, sampling_grid_l_,
                                              mode='bilinear', padding_mode='zeros')    ## N_*M_, D_, Lq_, P_
            sampling_input_features_list.append(sampling__input_features_l_)
        ## (N_*M_, D_, Lq_, L_*P_): [6, 256, 500, 60]
        sampling_input_features_all = torch.stack(sampling_input_features_list, dim=-2).flatten(-2)
        # sampling_input_features_all = rearrange(sampling_input_features_all, '(b n) c lq (l p) -> b n lq ')
        ## (N_*M_, Lq_, L_*P_, D_): [6, 500, 60, 256]
        sampling_input_features_all = sampling_input_features_all.permute(0,2,3,1).contiguous()
        ##  [1, 6, 500, 60, 256]
        sampling_input_features_all = rearrange(sampling_input_features_all, "(b h) n m d -> b h n m d", b=N)

        ##  [1, 6, 500, 60, 256] ->  [1, 500, 60, 64] -> [1, 6, 500, 60, 64]
        ##todo(chenzhao): need refactor this part multi-head loop
        key = torch.stack(
            [k_heads(sampling_input_features_all[:, i,]) for i, k_heads in enumerate(self.key_proj_heads)], dim=1)
        ## [1, 6, 500, 60, 64] -> [1*500, 6, 60, 64]
        key = rearrange(key, "b h n m d -> (b n) h m d")
        ##  [1, 6, 500, 60, 64] ->  [1, 500, 60, 64] -> [1, 6, 500, 60, 64]
        value = torch.stack(
            [v_heads(sampling_input_features_all[:, i,]) for i, v_heads in enumerate(self.value_proj_heads)], dim=1)
        ## [1, 6, 500, 60, 64] -> [1*500, 6, 60, 64]
        value = rearrange(value, "b h n m d -> (b n) h m d")
        ## [1, 500, 6*64] -> [1*500, 6, 1, 64]
        query =rearrange(self.query_proj(input_query), "b n (h d) -> (b n) h 1 d", h=self.n_heads)
        dots = einsum("b h i d, b h j d -> b h i j", query, key) * self.scale
        ## b h i j
        attn = self.attend(dots)
        output = einsum("b h i j, b h j d -> b h i d", attn, value)
        ## [1, 500, 6*384]
        output = rearrange(output, "(b n) h 1 d -> b n (h d)", b=N)

        output = self.output_proj(output)

        ## [1, 500, 6, 4, 15]
        attention_weights = rearrange(attn, "(b n) h 1 (l p) -> b n h l p", b=N, l=self.n_levels)
        if self.out_sample_loc:
            return output, torch.cat((sampling_locations,attention_weights[:,:,:,:,:,None]),dim=-1)
        else:
            return output, None
