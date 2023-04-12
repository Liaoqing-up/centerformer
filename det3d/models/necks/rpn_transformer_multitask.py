import time
import numpy as np

import torch

from torch import nn
from torch.nn import functional as F

from det3d.torchie.cnn import xavier_init
from det3d.models.utils import Sequential
from det3d.models.utils import Transformer, Deform_Transformer

from .. import builder
from ..registry import NECKS, MULTISCALENECKS
from ..utils import build_norm_layer
import  math

# ### add for visualize featuremap experiment
# import math
# import matplotlib.pyplot as plt
# ## plot gt box in featuremap
# from nuscenes import NuScenes
# from nuscenes.eval.common.loaders import load_gt, add_center_dist, filter_eval_boxes
# from nuscenes.eval.detection.data_classes import DetectionBox
# from nuscenes.eval.common.utils import boxes_to_sensor
# from nuscenes.eval.common.config import config_factory
#
# nusc_render = NuScenes(version='v1.0-mini', verbose=1, dataroot='data/nuScenes/')
# gt_boxes = load_gt(nusc_render, 'mini_val', DetectionBox, verbose=1)
# gt_boxes = add_center_dist(nusc_render, gt_boxes)
# cfg_ = config_factory('detection_cvpr_2019')
# gt_boxes = filter_eval_boxes(nusc_render, gt_boxes, cfg_.class_range, verbose=1)
#
# # Render GT boxes.
# def render_box_in_ax(ax, boxes_gt, scale=4):
#     for box in boxes_gt:
#         corners = change_corners_to_featurmap(box.corners(), scale)
#         box.render(ax, corners, view=np.eye(4), colors=('r', 'r', 'r'), linewidth=1)
#
# ### Render x_up multi channels
# def change_for_vis(feature, mode='max'):
#     assert len(feature.shape) in [2, 3]
#     if len(feature.shape) == 3:
#         if mode == 'max':
#             feature = torch.max(feature, dim=0)[0]
#         elif mode == 'mean':
#             feature = torch.mean(feature, dim=0)
#         else:
#             raise ValueError("mode only support 'max' and 'mean'!")
#     feature_norm = (feature - feature.min()) / (feature.max() - feature.min())
#     feature_show = (feature_norm * 255).cpu().numpy().astype(np.uint8)
#     return feature_show
#
# def change_corners_to_featurmap(corners, scales):
#     corners = (corners + 54) / 0.075 / scales
#     return corners
#
# def get_boxes_gt_from_sample_token(sample_token):
#     boxes_gt_global = gt_boxes[sample_token]
#     sample_rec = nusc_render.get('sample', sample_token)
#     sd_record = nusc_render.get('sample_data', sample_rec['data']['LIDAR_TOP'])
#     cs_record = nusc_render.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
#     pose_record = nusc_render.get('ego_pose', sd_record['ego_pose_token'])
#     boxes_gt = boxes_to_sensor(boxes_gt_global, pose_record, cs_record)
#     return boxes_gt
#
# def multiscale_featuremaps_render(feature_dict, sample_token, show=False, save=False):
#     assert len(list(feature_dict.values())[0]) == 2
#     boxes_gt = get_boxes_gt_from_sample_token(sample_token)
#
#     titles, features_scales = map(list, zip(*feature_dict.items()))
#     features, scales = map(list, zip(*features_scales))
#     plt.figure()
#     for i, feature in enumerate([change_for_vis(feature, mode='mean') for feature in features]):
#         ax = plt.subplot(math.ceil(len(titles)/3), 3, i+1)
#         ax.set_title(titles[i])
#         render_box_in_ax(ax, boxes_gt, scales[i])
#         plt.imshow(feature, cmap='jet')
#     if save:
#         plt.savefig(
#             f'/opt/sdatmp/lq/project/centerformer/debug/multiscale/{sample_token}.jpg',
#             bbox_inches = 'tight',
#             dpi = 600
#         )
#     if show:
#         plt.show()
#     plt.close()
#
# def multichannel_featuremaps_render(featuremap, scale, channels, sample_token, show=False, save=False):
#     boxes_gt = get_boxes_gt_from_sample_token(sample_token)
#     titles = [str(channel) for channel in channels]
#     plt.figure()
#     for i in channels:
#         ax = plt.subplot(math.ceil(len(channels)/3), 3, i+1)
#         ax.set_title(titles[i])
#         render_box_in_ax(ax, boxes_gt)
#         plt.imshow(change_for_vis(featuremap[i]), cmap='jet')
#     if save:
#         plt.savefig(
#             f'/opt/sdatmp/lq/project/centerformer/debug/channels/{sample_token}.jpg',
#             bbox_inches='tight',
#             dpi = 600
#         )
#     if show:
#         plt.show()
#     plt.close()
#
#
# def heatmaps_render(heatmaps, sample_token, show=False, save=False):
#     ## add for heatmap visualize experiment
#     boxes_gt = get_boxes_gt_from_sample_token(sample_token)
#     heatmaps_show = change_for_vis(heatmaps, mode='mean')
#     plt.figure()
#     plt.axis('off')
#     for i, heatmaps in enumerate([heatmaps_show]):
#         ax = plt.subplot(1, 1, i + 1)
#         ax.set_title('no TSA')
#         # render_box_in_ax(ax, boxes_gt, 4)
#         plt.imshow(heatmaps, cmap='jet')
#     if save:
#         plt.savefig(
#             f'/opt/sdatmp/lq/project/centerformer/debug/heatmaps/no_tsa/{sample_token}.jpg',
#             bbox_inches='tight',
#             dpi=600
#         )
#         print(f'save to {sample_token}')
#     if show:
#         plt.show()
#     plt.close()
#     # heatmaps_COLORMAP_HSV = cv2.applyColorMap(heatmaps, cv2.COLORMAP_JET)
#     # cv2.imwrite('/opt/sdatmp/lq/project/centerformer/debug/heatmap_hsv.jpg', heatmaps_COLORMAP_HSV)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // 16, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv1(y)
        return self.sigmoid(y) * x


class SpatialAttention_mtf(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention_mtf, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, curr, prev):
        avg_out = torch.mean(curr, dim=1, keepdim=True)
        max_out, _ = torch.max(curr, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv1(y)
        return self.sigmoid(y) * prev


class SpatialAttention_mtf_custom(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=7):
        super(SpatialAttention_mtf_custom, self).__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        return self.sigmoid(x)


class SpatialAttention_mtf_custom2(nn.Module):
    def __init__(self, in_channel_1, in_channel_2, kernel_size=7):
        super(SpatialAttention_mtf_custom2, self).__init__()

        self.conv1 = nn.Conv2d(in_channel_1, 1, 1)
        self.conv2 = nn.Conv2d(in_channel_2, in_channel_2, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, K, C, H, W = x.shape
        x = (self.conv1(x.reshape(-1, C, H, W))).reshape(B, K, H, W)
        x = self.conv2(x)
        return self.sigmoid(x)


@MULTISCALENECKS.register_module
class MultiscaleNeckCenterformer(nn.Module):
    def __init__(self, num_input_features, layer_nums, ds_num_filters, norm_cfg=None):
        super(MultiscaleNeckCenterformer, self).__init__()

        self._layer_strides = [1, 2, -4]
        self._num_filters = ds_num_filters
        self._layer_nums = layer_nums
        self._num_input_features = num_input_features

        assert len(self._layer_strides) == len(self._layer_nums)
        assert len(self._num_filters) == len(self._layer_nums)

        in_filters = [
            self._num_input_features,
            self._num_filters[0],
            self._num_filters[1],
        ]
        blocks = []

        if norm_cfg is None:
            norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self._norm_cfg = norm_cfg

        for i, layer_num in enumerate(self._layer_nums):
            block, num_out_filters = self._make_layer(
                in_filters[i],
                self._num_filters[i],
                layer_num,
                stride=self._layer_strides[i],
            )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)
        self.up = Sequential(
            nn.ConvTranspose2d(
                self._num_filters[0], self._num_filters[2], 2, stride=2, bias=False
            ),
            build_norm_layer(self._norm_cfg, self._num_filters[2])[1],
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.blocks[0](x)
        x_down = self.blocks[1](x)
        x_up = torch.cat([self.blocks[2](x_down), self.up(x)], dim=1)
        return [x, x_down, x_up]

    def _make_layer(self, inplanes, planes, num_blocks, stride=1):

        if stride > 0:
            block = Sequential(
                nn.ZeroPad2d(1),
                nn.Conv2d(inplanes, planes, 3, stride=stride, bias=False),
                build_norm_layer(self._norm_cfg, planes)[1],
                nn.ReLU(),
            )
        else:
            block = Sequential(
                nn.ConvTranspose2d(
                    inplanes, planes, -stride, stride=-stride, bias=False
                ),
                build_norm_layer(self._norm_cfg, planes)[1],
                nn.ReLU(),
            )

        for j in range(num_blocks):
            block.add(nn.Conv2d(planes, planes, 3, padding=1, bias=False))
            block.add(
                build_norm_layer(self._norm_cfg, planes)[1],
            )
            block.add(nn.ReLU())

        block.add(ChannelAttention(planes))
        block.add(SpatialAttention())

        return block, planes


@MULTISCALENECKS.register_module
class MultiscaleNeckPillarNet(nn.Module):
    def __init__(self, num_input_features, layer_nums, ds_num_filters, norm_cfg=None):
        super(MultiscaleNeckPillarNet, self).__init__()

        self._layer_strides = [1, 2, -2]
        self._num_filters = ds_num_filters
        self._layer_nums = layer_nums
        self._num_input_features = num_input_features

        assert len(self._layer_strides) == len(self._layer_nums)
        assert len(self._num_filters) == len(self._layer_nums)

        in_filters = [
            self._num_input_features,
            self._num_filters[0],
            self._num_filters[1],
        ]
        blocks = []

        if norm_cfg is None:
            norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self._norm_cfg = norm_cfg

        # fpn neck from pillarnet
        for i, layer_num in enumerate(self._layer_nums):
            block, num_out_filters = self._make_layer(
                in_filters[i],
                self._num_filters[0], #todo:[i],
                layer_num,
                stride=self._layer_strides[i],
            )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

        self.mix = Sequential(
                nn.Conv2d(in_filters[-1], in_filters[-1], 3, padding=1, bias=False),
                build_norm_layer(self._norm_cfg, in_filters[-1])[1],
                nn.ReLU(),
            )

        self.up = Sequential(
            nn.ConvTranspose2d(
                self._num_filters[0], self._num_filters[2]*2, 2, stride=2, bias=False
            ),
            build_norm_layer(self._norm_cfg, self._num_filters[2]*2)[1],
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.blocks[0](x)
        x_down = self.blocks[1](x)
        x_down_up = self.blocks[2](x_down)
        x_mix = self.mix(x + x_down_up)
        x_up = self.up(x_mix)
        return [x, x_down, x_up]

    def _make_layer(self, inplanes, planes, num_blocks, stride=1):

        if stride > 0:
            block = Sequential(
                nn.ZeroPad2d(1),
                nn.Conv2d(inplanes, planes, 3, stride=stride, bias=False),
                build_norm_layer(self._norm_cfg, planes)[1],
                nn.ReLU(),
            )
        else:
            block = Sequential(
                nn.ConvTranspose2d(
                    inplanes, planes, -stride, stride=-stride, bias=False
                ),
                build_norm_layer(self._norm_cfg, planes)[1],
                nn.ReLU(),
            )

        for j in range(num_blocks):
            block.add(nn.Conv2d(planes, planes, 3, padding=1, bias=False))
            block.add(
                build_norm_layer(self._norm_cfg, planes)[1],
            )
            block.add(nn.ReLU())

        block.add(ChannelAttention(planes))
        block.add(SpatialAttention())

        return block, planes




@NECKS.register_module
class RPN_transformer_base_multitask(nn.Module):
    def __init__(
        self,
        layer_nums,  # [2,2,2]
        ds_num_filters,  # [128,256,64]
        num_input_features,  # 256
        transformer_config=None,
        hm_head_layer=2,
        corner_head_layer=2,
        corner=False,
        assign_label_window_size=1,
        # classes=3,
        tasks=[],
        use_gt_training=False,
        norm_cfg=None,
        logger=None,
        init_bias=-2.19,
        score_threshold=0.1,
        obj_num=500,
        **kwargs
    ):
        super(RPN_transformer_base_multitask, self).__init__()
        self._layer_strides = [1, 2, -4]    #[1, 2, -2], [1, 2, -4]
        self._num_filters = ds_num_filters
        self._layer_nums = layer_nums
        self._num_input_features = num_input_features
        self.score_threshold = score_threshold
        self.transformer_config = transformer_config
        self.corner = corner
        self.obj_num = obj_num
        self.use_gt_training = use_gt_training
        self.window_size = assign_label_window_size**2
        self.cross_attention_kernel_size = [3, 3, 3]
        self.batch_id = None
        self.tasks = tasks

        if norm_cfg is None:
            norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self._norm_cfg = norm_cfg

        assert len(self._layer_strides) == len(self._layer_nums)
        assert len(self._num_filters) == len(self._layer_nums)
        assert self.transformer_config is not None

        in_filters = [
            self._num_input_features,
            self._num_filters[0],
            self._num_filters[1],
        ]
        blocks = []

        for i, layer_num in enumerate(self._layer_nums):
            block, num_out_filters = self._make_layer(
                in_filters[i],
                self._num_filters[i],
                layer_num,
                stride=self._layer_strides[i],
            )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)
        self.up = Sequential(
            nn.ConvTranspose2d(
                self._num_filters[0], self._num_filters[2], 2, stride=2, bias=False
            ),
            build_norm_layer(self._norm_cfg, self._num_filters[2])[1],
            nn.ReLU(),
        )

        # # fpn neck from pillarnet
        # for i, layer_num in enumerate(self._layer_nums):
        #     block, num_out_filters = self._make_layer(
        #         in_filters[i],
        #         self._num_filters[0], #todo:[i],
        #         layer_num,
        #         stride=self._layer_strides[i],
        #     )
        #     blocks.append(block)
        # self.blocks = nn.ModuleList(blocks)
        #
        # self.mix = Sequential(
        #         nn.Conv2d(in_filters[-1], in_filters[-1], 3, padding=1, bias=False),
        #         build_norm_layer(self._norm_cfg, in_filters[-1])[1],
        #         nn.ReLU(),
        #     )
        #
        # self.up = Sequential(
        #     nn.ConvTranspose2d(
        #         self._num_filters[0], self._num_filters[2]*2, 2, stride=2, bias=False
        #     ),
        #     # torch.nn.Upsample(size=None, scale_factor=2, mode='nearest', align_corners=None),
        #     build_norm_layer(self._norm_cfg, self._num_filters[2]*2)[1],
        #     nn.ReLU(),
        # )

        # heatmap prediction
        self.hm_heads = nn.ModuleList()
        for task in self.tasks:
            hm_head = Sequential()
            for i in range(hm_head_layer - 1):
                hm_head.add(
                    nn.Conv2d(
                        self._num_filters[-1] * 2,
                        64,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                    )
                )
                hm_head.add(build_norm_layer(self._norm_cfg, 64)[1])
                hm_head.add(nn.ReLU())

            hm_head.add(
                nn.Conv2d(64, task['num_class'], kernel_size=3, stride=1, padding=1, bias=True)
            )
            hm_head[-1].bias.data.fill_(init_bias)
            self.hm_heads.append(hm_head)

        if self.corner:
            self.corner_heads = nn.ModuleList()
            for task in self.tasks:
                corner_head = Sequential()
                for i in range(corner_head_layer - 1):
                    corner_head.add(
                        nn.Conv2d(
                            self._num_filters[-1] * 2,
                            64,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=True,
                        )
                    )
                    corner_head.add(build_norm_layer(self._norm_cfg, 64)[1])
                    corner_head.add(nn.ReLU())

                corner_head.add(
                    nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=True)
                )
                corner_head[-1].bias.data.fill_(init_bias)
                self.corner_heads.append(corner_head)

    def _make_layer(self, inplanes, planes, num_blocks, stride=1):

        if stride > 0:
            block = Sequential(
                nn.ZeroPad2d(1),
                nn.Conv2d(inplanes, planes, 3, stride=stride, bias=False),
                build_norm_layer(self._norm_cfg, planes)[1],
                nn.ReLU(),
            )
        else:
            # block = F.interpolate(x_down, scale_factor=-stride, mode='bilinear')
            # torch.nn.Upsample(size=None, scale_factor=-stride, mode='nearest', align_corners=None)
            block = Sequential(
                nn.ConvTranspose2d(
                    inplanes, planes, -stride, stride=-stride, bias=False
                ),
                # torch.nn.Upsample(size=None, scale_factor=-stride, mode='nearest', align_corners=None),
                build_norm_layer(self._norm_cfg, planes)[1],
                nn.ReLU(),
            )

        for j in range(num_blocks):
            block.add(nn.Conv2d(planes, planes, 3, padding=1, bias=False))
            block.add(
                build_norm_layer(self._norm_cfg, planes)[1],
            )
            block.add(nn.ReLU())

        block.add(ChannelAttention(planes))
        block.add(SpatialAttention())

        return block, planes

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")

    def forward(self, x, example=None):
        pass

    def get_multi_scale_feature(self, center_pos, feats):
        """
        Args:
            center_pos: center coor at the lowest scale feature map [B 500 2]
            feats: multi scale BEV feature 3*[B C H W]
        Returns:
            neighbor_feat: [B 500 K C]
            neighbor_pos: [B 500 K 2]
        """
        kernel_size = self.cross_attention_kernel_size
        batch, num_cls, H, W = feats[0].size()

        center_num = center_pos.shape[1]

        relative_pos_list = []
        neighbor_feat_list = []
        for i, k in enumerate(kernel_size):
            neighbor_coords = torch.arange(-(k // 2), (k // 2) + 1)
            neighbor_coords = torch.flatten(
                torch.stack(torch.meshgrid([neighbor_coords, neighbor_coords]), dim=0),
                1,
            )  # [2, k]
            neighbor_coords = (
                neighbor_coords.permute(1, 0).contiguous().to(center_pos)
            )  # relative coordinate [k, 2]
            neighbor_coords = (
                center_pos[:, :, None, :] // (2**i)
                + neighbor_coords[None, None, :, :]
            )  # coordinates [B, 500, k, 2]
            neighbor_coords = torch.clamp(
                neighbor_coords, min=0, max=H // (2**i) - 1
            )  # prevent out of bound
            feat_id = (
                neighbor_coords[:, :, :, 1] * (W // (2**i))
                + neighbor_coords[:, :, :, 0]
            )  # pixel id [B, 500, k]
            feat_id = feat_id.reshape(batch, -1)  # pixel id [B, 500*k]
            # selected_feat = torch.gather(feats[i].reshape(batch, num_cls,(H*W)//(4**i)).permute(0, 2, 1).contiguous(),1,feat_id)
            selected_feat = (
                feats[i]
                .reshape(batch, num_cls, (H * W) // (4**i))
                .permute(0, 2, 1)
                .contiguous()[self.batch_id.repeat(1, k**2), feat_id]
            )  # B, 500*k, C
            neighbor_feat_list.append(
                selected_feat.reshape(batch, center_num, -1, num_cls)
            )  # B, 500, k, C
            relative_pos_list.append(neighbor_coords * (2**i))  # B, 500, k, 2
            # relative_pos_list.append(F.pad(neighbor_coords*(2**i), (0,1), "constant", i)) # B, 500, k, 3

        neighbor_pos = torch.cat(relative_pos_list, dim=2)  # B, 500, K, 2/3
        neighbor_feats = torch.cat(neighbor_feat_list, dim=2)  # B, 500, K, C
        return neighbor_feats, neighbor_pos

    def get_multi_scale_feature_multiframe(self, center_pos, feats, timeframe):
        """
        Args:
            center_pos: center coor at the lowest scale feature map [B 500 2]
            feats: multi scale BEV feature (3+k)*[B C H W]
            timeframe: timeframe [B,k]
        Returns:
            neighbor_feat: [B 500 K C]
            neighbor_pos: [B 500 K 2]
            neighbor_time: [B 500 K 1]
        """
        kernel_size = self.cross_attention_kernel_size
        batch, num_cls, H, W = feats[0].size()

        center_num = center_pos.shape[1]

        relative_pos_list = []
        neighbor_feat_list = []
        timeframe_list = []
        for i, k in enumerate(kernel_size):
            neighbor_coords = torch.arange(-(k // 2), (k // 2) + 1)
            neighbor_coords = torch.flatten(
                torch.stack(torch.meshgrid([neighbor_coords, neighbor_coords]), dim=0),
                1,
            )  # [2, k]
            neighbor_coords = (
                neighbor_coords.permute(1, 0).contiguous().to(center_pos)
            )  # relative coordinate [k, 2]
            neighbor_coords = (
                center_pos[:, :, None, :] // (2**i)
                + neighbor_coords[None, None, :, :]
            )  # coordinates [B, 500, k, 2]
            neighbor_coords = torch.clamp(
                neighbor_coords, min=0, max=H // (2**i) - 1
            )  # prevent out of bound
            feat_id = (
                neighbor_coords[:, :, :, 1] * (W // (2**i))
                + neighbor_coords[:, :, :, 0]
            )  # pixel id [B, 500, k]
            feat_id = feat_id.reshape(batch, -1)  # pixel id [B, 500*k]
            selected_feat = (
                feats[i]
                .reshape(batch, num_cls, (H * W) // (4**i))
                .permute(0, 2, 1)
                .contiguous()[self.batch_id.repeat(1, k**2), feat_id]
            )  # B, 500*k, C
            neighbor_feat_list.append(
                selected_feat.reshape(batch, center_num, -1, num_cls)
            )  # B, 500, k, C
            relative_pos_list.append(neighbor_coords * (2**i))  # B, 500, k, 2
            timeframe_list.append(
                torch.full_like(neighbor_coords[:, :, :, 0:1], 0)
            )  # B, 500, k
            if i == 0:
                # add previous frame feature
                for frame_num in range(feats[-1].shape[1]):
                    selected_feat = (
                        feats[-1][:, frame_num, :, :, :]
                        .reshape(batch, num_cls, (H * W) // (4**i))
                        .permute(0, 2, 1)
                        .contiguous()[self.batch_id.repeat(1, k**2), feat_id]
                    )  # B, 500*k, C
                    neighbor_feat_list.append(
                        selected_feat.reshape(batch, center_num, -1, num_cls)
                    )
                    relative_pos_list.append(neighbor_coords * (2**i))
                    time = timeframe[:, frame_num + 1].to(selected_feat)  # B
                    timeframe_list.append(
                        time[:, None, None, None]
                        * torch.full_like(neighbor_coords[:, :, :, 0:1], 1)
                    )  # B, 500, k

        neighbor_pos = torch.cat(relative_pos_list, dim=2)  # B, 500, K, 2/3
        neighbor_feats = torch.cat(neighbor_feat_list, dim=2)  # B, 500, K, C
        neighbor_time = torch.cat(timeframe_list, dim=2)  # B, 500, K, 1

        return neighbor_feats, neighbor_pos, neighbor_time


@NECKS.register_module
class RPN_transformer_base_multitask_refactor(nn.Module):
    def __init__(
        self,
        ds_num_filters,
        multiscale_neck=None,
        transformer_config=None,
        hm_head_layer=2,
        corner_head_layer=2,
        corner=False,
        assign_label_window_size=1,
        # classes=3,
        tasks=[],
        use_gt_training=False,
        norm_cfg=None,
        logger=None,
        init_bias=-2.19,
        score_threshold=0.1,
        obj_num=500,
        **kwargs
    ):
        super(RPN_transformer_base_multitask_refactor, self).__init__()
        if multiscale_neck is not None:
            self.multiscale_neck = builder.build_multiscale_neck(multiscale_neck)
        self.score_threshold = score_threshold
        self.transformer_config = transformer_config
        self.corner = corner
        self.obj_num = obj_num
        self.use_gt_training = use_gt_training
        self.window_size = assign_label_window_size**2
        self.cross_attention_kernel_size = [3, 3, 3]
        self.batch_id = None
        self.tasks = tasks
        self._num_filters = ds_num_filters

        if norm_cfg is None:
            norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self._norm_cfg = norm_cfg

        assert self.transformer_config is not None

        # heatmap prediction
        self.hm_heads = nn.ModuleList()
        for task in self.tasks:
            hm_head = Sequential()
            for i in range(hm_head_layer - 1):
                hm_head.add(
                    nn.Conv2d(
                        self._num_filters[-1] * 2,
                        64,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                    )
                )
                hm_head.add(build_norm_layer(self._norm_cfg, 64)[1])
                hm_head.add(nn.ReLU())

            hm_head.add(
                nn.Conv2d(64, task['num_class'], kernel_size=3, stride=1, padding=1, bias=True)
            )
            hm_head[-1].bias.data.fill_(init_bias)
            self.hm_heads.append(hm_head)

        if self.corner:
            self.corner_heads = nn.ModuleList()
            for task in self.tasks:
                corner_head = Sequential()
                for i in range(corner_head_layer - 1):
                    corner_head.add(
                        nn.Conv2d(
                            self._num_filters[-1] * 2,
                            64,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=True,
                        )
                    )
                    corner_head.add(build_norm_layer(self._norm_cfg, 64)[1])
                    corner_head.add(nn.ReLU())

                corner_head.add(
                    nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=True)
                )
                corner_head[-1].bias.data.fill_(init_bias)
                self.corner_heads.append(corner_head)

    def _make_layer(self, inplanes, planes, num_blocks, stride=1):

        if stride > 0:
            block = Sequential(
                nn.ZeroPad2d(1),
                nn.Conv2d(inplanes, planes, 3, stride=stride, bias=False),
                build_norm_layer(self._norm_cfg, planes)[1],
                nn.ReLU(),
            )
        else:
            block = Sequential(
                nn.ConvTranspose2d(
                    inplanes, planes, -stride, stride=-stride, bias=False
                ),
                build_norm_layer(self._norm_cfg, planes)[1],
                nn.ReLU(),
            )

        for j in range(num_blocks):
            block.add(nn.Conv2d(planes, planes, 3, padding=1, bias=False))
            block.add(
                build_norm_layer(self._norm_cfg, planes)[1],
            )
            block.add(nn.ReLU())

        block.add(ChannelAttention(planes))
        block.add(SpatialAttention())

        return block, planes

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")

    def forward(self, x, example=None):
        pass

    def get_multi_scale_feature(self, center_pos, feats):
        """
        Args:
            center_pos: center coor at the lowest scale feature map [B 500 2]
            feats: multi scale BEV feature 3*[B C H W]
        Returns:
            neighbor_feat: [B 500 K C]
            neighbor_pos: [B 500 K 2]
        """
        kernel_size = self.cross_attention_kernel_size
        batch, num_cls, H, W = feats[0].size()

        center_num = center_pos.shape[1]

        relative_pos_list = []
        neighbor_feat_list = []
        for i, k in enumerate(kernel_size):
            neighbor_coords = torch.arange(-(k // 2), (k // 2) + 1)
            neighbor_coords = torch.flatten(
                torch.stack(torch.meshgrid([neighbor_coords, neighbor_coords]), dim=0),
                1,
            )  # [2, k]
            neighbor_coords = (
                neighbor_coords.permute(1, 0).contiguous().to(center_pos)
            )  # relative coordinate [k, 2]
            neighbor_coords = (
                center_pos[:, :, None, :] // (2**i)
                + neighbor_coords[None, None, :, :]
            )  # coordinates [B, 500, k, 2]
            neighbor_coords = torch.clamp(
                neighbor_coords, min=0, max=H // (2**i) - 1
            )  # prevent out of bound
            feat_id = (
                neighbor_coords[:, :, :, 1] * (W // (2**i))
                + neighbor_coords[:, :, :, 0]
            )  # pixel id [B, 500, k]
            feat_id = feat_id.reshape(batch, -1)  # pixel id [B, 500*k]
            # selected_feat = torch.gather(feats[i].reshape(batch, num_cls,(H*W)//(4**i)).permute(0, 2, 1).contiguous(),1,feat_id)
            selected_feat = (
                feats[i]
                .reshape(batch, num_cls, (H * W) // (4**i))
                .permute(0, 2, 1)
                .contiguous()[self.batch_id.repeat(1, k**2), feat_id]
            )  # B, 500*k, C
            neighbor_feat_list.append(
                selected_feat.reshape(batch, center_num, -1, num_cls)
            )  # B, 500, k, C
            relative_pos_list.append(neighbor_coords * (2**i))  # B, 500, k, 2
            # relative_pos_list.append(F.pad(neighbor_coords*(2**i), (0,1), "constant", i)) # B, 500, k, 3

        neighbor_pos = torch.cat(relative_pos_list, dim=2)  # B, 500, K, 2/3
        neighbor_feats = torch.cat(neighbor_feat_list, dim=2)  # B, 500, K, C
        return neighbor_feats, neighbor_pos

    def get_multi_scale_feature_multiframe(self, center_pos, feats, timeframe):
        """
        Args:
            center_pos: center coor at the lowest scale feature map [B 500 2]
            feats: multi scale BEV feature (3+k)*[B C H W]
            timeframe: timeframe [B,k]
        Returns:
            neighbor_feat: [B 500 K C]
            neighbor_pos: [B 500 K 2]
            neighbor_time: [B 500 K 1]
        """
        kernel_size = self.cross_attention_kernel_size
        batch, num_cls, H, W = feats[0].size()

        center_num = center_pos.shape[1]

        relative_pos_list = []
        neighbor_feat_list = []
        timeframe_list = []
        for i, k in enumerate(kernel_size):
            neighbor_coords = torch.arange(-(k // 2), (k // 2) + 1)
            neighbor_coords = torch.flatten(
                torch.stack(torch.meshgrid([neighbor_coords, neighbor_coords]), dim=0),
                1,
            )  # [2, k]
            neighbor_coords = (
                neighbor_coords.permute(1, 0).contiguous().to(center_pos)
            )  # relative coordinate [k, 2]
            neighbor_coords = (
                center_pos[:, :, None, :] // (2**i)
                + neighbor_coords[None, None, :, :]
            )  # coordinates [B, 500, k, 2]
            neighbor_coords = torch.clamp(
                neighbor_coords, min=0, max=H // (2**i) - 1
            )  # prevent out of bound
            feat_id = (
                neighbor_coords[:, :, :, 1] * (W // (2**i))
                + neighbor_coords[:, :, :, 0]
            )  # pixel id [B, 500, k]
            feat_id = feat_id.reshape(batch, -1)  # pixel id [B, 500*k]
            selected_feat = (
                feats[i]
                .reshape(batch, num_cls, (H * W) // (4**i))
                .permute(0, 2, 1)
                .contiguous()[self.batch_id.repeat(1, k**2), feat_id]
            )  # B, 500*k, C
            neighbor_feat_list.append(
                selected_feat.reshape(batch, center_num, -1, num_cls)
            )  # B, 500, k, C
            relative_pos_list.append(neighbor_coords * (2**i))  # B, 500, k, 2
            timeframe_list.append(
                torch.full_like(neighbor_coords[:, :, :, 0:1], 0)
            )  # B, 500, k
            if i == 0:
                # add previous frame feature
                for frame_num in range(feats[-1].shape[1]):
                    selected_feat = (
                        feats[-1][:, frame_num, :, :, :]
                        .reshape(batch, num_cls, (H * W) // (4**i))
                        .permute(0, 2, 1)
                        .contiguous()[self.batch_id.repeat(1, k**2), feat_id]
                    )  # B, 500*k, C
                    neighbor_feat_list.append(
                        selected_feat.reshape(batch, center_num, -1, num_cls)
                    )
                    relative_pos_list.append(neighbor_coords * (2**i))
                    time = timeframe[:, frame_num + 1].to(selected_feat)  # B
                    timeframe_list.append(
                        time[:, None, None, None]
                        * torch.full_like(neighbor_coords[:, :, :, 0:1], 1)
                    )  # B, 500, k

        neighbor_pos = torch.cat(relative_pos_list, dim=2)  # B, 500, K, 2/3
        neighbor_feats = torch.cat(neighbor_feat_list, dim=2)  # B, 500, K, C
        neighbor_time = torch.cat(timeframe_list, dim=2)  # B, 500, K, 1

        return neighbor_feats, neighbor_pos, neighbor_time


@NECKS.register_module
class RPN_transformer_multitask(RPN_transformer_base_multitask):
    def __init__(
        self,
        layer_nums,  # [2,2,2]
        ds_num_filters,  # [128,256,64]
        num_input_features,  # 256
        transformer_config=None,
        hm_head_layer=2,
        corner_head_layer=2,
        corner=False,
        assign_label_window_size=1,
        tasks=[],
        use_gt_training=False,
        norm_cfg=None,
        name="rpn_transformer_multitask",
        logger=None,
        init_bias=-2.19,
        score_threshold=0.1,
        obj_num=500,
        parametric_embedding=False,
        **kwargs
    ):
        super(RPN_transformer_multitask, self).__init__(
            layer_nums,
            ds_num_filters,
            num_input_features,
            transformer_config,
            hm_head_layer,
            corner_head_layer,
            corner,
            assign_label_window_size,
            tasks,
            use_gt_training,
            norm_cfg,
            logger,
            init_bias,
            score_threshold,
            obj_num,
        )

        self.transformer_layer = Transformer(
            self._num_filters[-1] * 2,
            depth=transformer_config.depth,
            heads=transformer_config.heads,
            dim_head=transformer_config.dim_head,
            mlp_dim=transformer_config.MLP_dim,
            dropout=transformer_config.DP_rate,
            out_attention=transformer_config.out_att,
        )
        self.pos_embedding_type = transformer_config.get(
            "pos_embedding_type", "linear"
        )
        if self.pos_embedding_type == "linear":
            if len(self.tasks)>1:
                self.pos_embedding = nn.Linear(3, self._num_filters[-1] * 2)
            else:
                self.pos_embedding = nn.Linear(2, self._num_filters[-1] * 2)
        elif self.pos_embedding_type == "none":
            self.pos_embedding = None
        else:
            raise NotImplementedError()
        self.cross_attention_kernel_size = transformer_config.cross_attention_kernel_size
        self.parametric_embedding = parametric_embedding
        if self.parametric_embedding:
            self.query_embed = nn.Embedding(self.obj_num * len(self.tasks), self._num_filters[-1] * 2)
            nn.init.uniform_(self.query_embed.weight, -1.0, 1.0)

        logger.info("Finish RPN_transformer Initialization")

    def forward(self, x, example=None):

        # FPN
        x = self.blocks[0](x)
        x_down = self.blocks[1](x)
        x_up = torch.cat([self.blocks[2](x_down), self.up(x)], dim=1)

        order_list = []
        out_dict_list = []
        for idx, task in enumerate(self.tasks):
            # heatmap head
            hm = self.hm_heads[idx](x_up)

            if self.corner and self.corner_heads[0].training:
                corner_hm = self.corner_heads[idx](x_up)
                corner_hm = torch.sigmoid(corner_hm)

            # find top K center location
            hm = torch.sigmoid(hm)
            batch, num_cls, H, W = hm.size()

            scores, labels = torch.max(hm.reshape(batch, num_cls, H * W), dim=1)  # b,H*W

            if self.use_gt_training and self.hm_heads[0].training:
                gt_inds = example["ind"][idx][:, (self.window_size // 2) :: self.window_size]
                gt_masks = example["mask"][idx][
                    :, (self.window_size // 2) :: self.window_size
                ]
                batch_id_gt = torch.from_numpy(np.indices((batch, gt_inds.shape[1]))[0]).to(
                    labels
                )
                scores[batch_id_gt, gt_inds] = scores[batch_id_gt, gt_inds] + gt_masks
                order = scores.sort(1, descending=True)[1]
                order = order[:, : self.obj_num]
                scores[batch_id_gt, gt_inds] = scores[batch_id_gt, gt_inds] - gt_masks
            else:
                order = scores.sort(1, descending=True)[1]
                order = order[:, : self.obj_num]

            scores = torch.gather(scores, 1, order)
            labels = torch.gather(labels, 1, order)
            mask = scores > self.score_threshold
            order_list.append(order)

            out_dict = {}
            out_dict.update(
                {
                    "hm": hm,
                    "scores": scores,
                    "labels": labels,
                    "order": order,
                    "mask": mask,
                    "BEV_feat": x_up,
                    "H": H,
                    "W": W,
                }
            )
            if self.corner and self.corner_heads[0].training:
                out_dict.update({"corner_hm": corner_hm})
            out_dict_list.append(out_dict)

        self.batch_id = torch.from_numpy(np.indices((batch, self.obj_num * len(self.tasks)))[0]).to(
                labels
            )
        order_all = torch.cat(order_list,dim=1)

        ct_feat = (
            x_up.reshape(batch, -1, H * W)
            .transpose(2, 1)
            .contiguous()[self.batch_id, order_all]
        )  # B, 500, C

        # create position embedding for each center
        y_coor = order_all // W
        # x_coor = order - y_coor*W
        x_coor = order_all % W
        pos_features = torch.stack([x_coor, y_coor], dim=2)

        if self.parametric_embedding:
            ct_feat = self.query_embed.weight
            ct_feat = ct_feat.unsqueeze(0).expand(batch, -1, -1)

        # run transformer
        neighbor_feat, neighbor_pos = self.get_multi_scale_feature(
            pos_features, [x_up, x, x_down]
        )

        if len(self.tasks) > 1:
            task_ids = torch.repeat_interleave(torch.arange(len(self.tasks)).repeat(batch,1), self.obj_num, dim=1).to(neighbor_pos) # B, 500
            neighbor_task_ids = task_ids.unsqueeze(2).repeat(1, 1, neighbor_pos.shape[2]) # B, 500, K
            pos_features = torch.cat([pos_features, task_ids[:, :, None]],dim=-1)
            neighbor_pos = torch.cat([neighbor_pos, neighbor_task_ids[:, :, :, None]],dim=-1)

        transformer_out = self.transformer_layer(
            ct_feat,
            pos_embedding=self.pos_embedding,
            center_pos=pos_features.to(ct_feat),
            y=neighbor_feat,
            neighbor_pos=neighbor_pos.to(ct_feat),
        )  # (B,N,C)

        ct_feat = (
            transformer_out["ct_feat"].transpose(2, 1).contiguous()
        )  # B, C, 500

        for idx, task in enumerate(self.tasks):
            out_dict_list[idx]["ct_feat"] = ct_feat[:, :, idx * self.obj_num : (idx+1) * self.obj_num]

        return out_dict_list

@NECKS.register_module
class RPN_transformer_deformable_multitask(RPN_transformer_base_multitask):
    def __init__(
        self,
        layer_nums,  # [2,2,2]
        ds_num_filters,  # [128,256,64]
        num_input_features,  # 256
        transformer_config=None,
        hm_head_layer=2,
        corner_head_layer=2,
        corner=False,
        assign_label_window_size=1,
        tasks=[],
        use_gt_training=False,
        norm_cfg=None,
        name="rpn_transformer_deformable_multitask",
        logger=None,
        init_bias=-2.19,
        score_threshold=0.1,
        obj_num=500,
        parametric_embedding=False,
        **kwargs
    ):
        super(RPN_transformer_deformable_multitask, self).__init__(
            layer_nums,
            ds_num_filters,
            num_input_features,
            transformer_config,
            hm_head_layer,
            corner_head_layer,
            corner,
            assign_label_window_size,
            tasks,
            use_gt_training,
            norm_cfg,
            logger,
            init_bias,
            score_threshold,
            obj_num,
        )

        self.transformer_layer = Deform_Transformer(
            self._num_filters[-1] * 2,
            depth=transformer_config.depth,
            heads=transformer_config.heads,
            dim_head=transformer_config.dim_head,
            mlp_dim=transformer_config.MLP_dim,
            dropout=transformer_config.DP_rate,
            out_attention=transformer_config.out_att,
            n_points=transformer_config.get("n_points", 9),
        )
        self.pos_embedding_type = transformer_config.get(
            "pos_embedding_type", "linear"
        )
        if self.pos_embedding_type == "linear":
            if len(self.tasks)>1:
                self.pos_embedding = nn.Linear(3, self._num_filters[-1] * 2)
            else:
                self.pos_embedding = nn.Linear(2, self._num_filters[-1] * 2)
        else:
            raise NotImplementedError()
        self.parametric_embedding = parametric_embedding
        if self.parametric_embedding:
            self.query_embed = nn.Embedding(self.obj_num * len(self.tasks), self._num_filters[-1] * 2)
            nn.init.uniform_(self.query_embed.weight, -1.0, 1.0)

        logger.info("Finish RPN_transformer_deformable Initialization")

    def forward(self, x, example=None):

        # FPN
        x = self.blocks[0](x)
        x_down = self.blocks[1](x)
        x_up = torch.cat([self.blocks[2](x_down), self.up(x)], dim=1)

        order_list = []
        out_dict_list = []
        for idx, task in enumerate(self.tasks):
            # heatmap head
            hm = self.hm_heads[idx](x_up)

            if self.corner and self.corner_heads[0].training:
                corner_hm = self.corner_heads[idx](x_up)
                corner_hm = torch.sigmoid(corner_hm)

            # find top K center location
            hm = torch.sigmoid(hm)
            batch, num_cls, H, W = hm.size()

            scores, labels = torch.max(hm.reshape(batch, num_cls, H * W), dim=1)  # b,H*W

            if self.use_gt_training and self.hm_heads[0].training:
                gt_inds = example["ind"][idx][:, (self.window_size // 2) :: self.window_size]
                gt_masks = example["mask"][idx][
                    :, (self.window_size // 2) :: self.window_size
                ]
                batch_id_gt = torch.from_numpy(np.indices((batch, gt_inds.shape[1]))[0]).to(
                    labels
                )
                scores[batch_id_gt, gt_inds] = scores[batch_id_gt, gt_inds] + gt_masks
                order = scores.sort(1, descending=True)[1]
                order = order[:, : self.obj_num]
                scores[batch_id_gt, gt_inds] = scores[batch_id_gt, gt_inds] - gt_masks
            else:
                order = scores.sort(1, descending=True)[1]
                order = order[:, : self.obj_num]

            scores = torch.gather(scores, 1, order)
            labels = torch.gather(labels, 1, order)
            mask = scores > self.score_threshold
            order_list.append(order)

            out_dict = {}
            out_dict.update(
                {
                    "hm": hm,
                    "scores": scores,
                    "labels": labels,
                    "order": order,
                    "mask": mask,
                    "BEV_feat": x_up,
                    "H": H,
                    "W": W,
                }
            )
            if self.corner and self.corner_heads[0].training:
                out_dict.update({"corner_hm": corner_hm})
            out_dict_list.append(out_dict)

        self.batch_id = torch.from_numpy(np.indices((batch, self.obj_num * len(self.tasks)))[0]).to(
                labels
            )
        order_all = torch.cat(order_list,dim=1)

        ct_feat = (
            x_up.reshape(batch, -1, H * W)
            .transpose(2, 1)
            .contiguous()[self.batch_id, order_all]
        )  # B, 500, C

        # create position embedding for each center
        y_coor = order_all // W
        x_coor = order_all - y_coor * W
        y_coor, x_coor = y_coor.to(ct_feat), x_coor.to(ct_feat)
        y_coor, x_coor = y_coor / H, x_coor / W
        pos_features = torch.stack([x_coor, y_coor], dim=2)

        if self.parametric_embedding:
            ct_feat = self.query_embed.weight
            ct_feat = ct_feat.unsqueeze(0).expand(batch, -1, -1)

        # run transformer
        src = torch.cat(
            (
                x_up.reshape(batch, -1, H * W).transpose(2, 1).contiguous(),
                x.reshape(batch, -1, (H * W) // 4).transpose(2, 1).contiguous(),
                x_down.reshape(batch, -1, (H * W) // 16)
                .transpose(2, 1)
                .contiguous(),
            ),
            dim=1,
        )  # B ,sum(H*W), C
        spatial_shapes = torch.as_tensor(
            [(H, W), (H // 2, W // 2), (H // 4, W // 4)],
            dtype=torch.long,
            device=ct_feat.device,
        )
        level_start_index = torch.cat(
            (
                spatial_shapes.new_zeros((1,)),
                spatial_shapes.prod(1).cumsum(0)[:-1],
            )
        )

        if len(self.tasks) > 1:
            task_ids = torch.repeat_interleave(torch.arange(len(self.tasks)).repeat(batch,1), self.obj_num, dim=1).to(pos_features) # B, 500
            pos_features = torch.cat([pos_features, task_ids[:, :, None]],dim=-1)

        transformer_out = self.transformer_layer(
            ct_feat,
            self.pos_embedding,
            src,
            spatial_shapes,
            level_start_index,
            center_pos=pos_features,
        )  # (B,N,C)

        ct_feat = (
            transformer_out["ct_feat"].transpose(2, 1).contiguous()
        )  # B, C, 500

        for idx, task in enumerate(self.tasks):
            out_dict_list[idx]["ct_feat"] = ct_feat[:, :, idx * self.obj_num : (idx+1) * self.obj_num]

        return out_dict_list


@NECKS.register_module
class RPN_transformer_deformable_multitask_refactor_singlestage(RPN_transformer_base_multitask_refactor):
    def __init__(
        self,
        ds_num_filters,
        multiscale_neck=None,
        transformer_config=None,
        hm_head_layer=2,
        corner_head_layer=2,
        corner=False,
        assign_label_window_size=1,
        tasks=[],
        use_gt_training=False,
        norm_cfg=None,
        name="RPN_transformer_deformable_multitask_refactor_singlestage",
        logger=None,
        init_bias=-2.19,
        score_threshold=0.1,
        obj_num=500,
        parametric_embedding=False,
        **kwargs
    ):
        super(RPN_transformer_deformable_multitask_refactor_singlestage, self).__init__(
            ds_num_filters,
            multiscale_neck,
            transformer_config,
            hm_head_layer,
            corner_head_layer,
            corner,
            assign_label_window_size,
            tasks,
            use_gt_training,
            norm_cfg,
            logger,
            init_bias,
            score_threshold,
            obj_num,
        )

        self.transformer_layer = Deform_Transformer(
            self._num_filters[-1] * 2,
            depth=transformer_config.depth,
            heads=transformer_config.heads,
            dim_head=transformer_config.dim_head,
            mlp_dim=transformer_config.MLP_dim,
            dropout=transformer_config.DP_rate,
            out_attention=transformer_config.out_att,
            n_points=transformer_config.get("n_points", 9),
        )
        self.pos_embedding_type = transformer_config.get(
            "pos_embedding_type", "linear"
        )
        if self.pos_embedding_type == "linear":
            if len(self.tasks)>1:
                self.pos_embedding = nn.Linear(3, self._num_filters[-1] * 2)
            else:
                self.pos_embedding = nn.Linear(2, self._num_filters[-1] * 2)
        else:
            raise NotImplementedError()
        self.parametric_embedding = parametric_embedding
        if self.parametric_embedding:
            self.query_embed = nn.Embedding(self.obj_num * len(self.tasks), self._num_filters[-1] * 2)
            nn.init.uniform_(self.query_embed.weight, -1.0, 1.0)

        logger.info("Finish RPN_transformer_deformable Initialization")

    def forward(self, x, example=None):

        # FPN
        x, x_down, x_up = self.multiscale_neck(x)

        order_list = []
        out_dict_list = []
        for idx, task in enumerate(self.tasks):
            # heatmap head
            hm = self.hm_heads[idx](x_up)

            if self.corner and self.corner_heads[0].training:
                corner_hm = self.corner_heads[idx](x_up)
                corner_hm = torch.sigmoid(corner_hm)

            # find top K center location
            hm = torch.sigmoid(hm)
            batch, num_cls, H, W = hm.size()

            scores, labels = torch.max(hm.reshape(batch, num_cls, H * W), dim=1)  # b,H*W

            if self.use_gt_training and self.hm_heads[0].training:
                gt_inds = example["ind"][idx][:, (self.window_size // 2) :: self.window_size]
                gt_masks = example["mask"][idx][
                    :, (self.window_size // 2) :: self.window_size
                ]
                batch_id_gt = torch.from_numpy(np.indices((batch, gt_inds.shape[1]))[0]).to(
                    labels
                )
                scores[batch_id_gt, gt_inds] = scores[batch_id_gt, gt_inds] + gt_masks
                order = scores.sort(1, descending=True)[1]
                order = order[:, : self.obj_num]
                scores[batch_id_gt, gt_inds] = scores[batch_id_gt, gt_inds] - gt_masks
            else:
                order = scores.sort(1, descending=True)[1]
                order = order[:, : self.obj_num]

            scores = torch.gather(scores, 1, order)
            labels = torch.gather(labels, 1, order)
            mask = scores > self.score_threshold
            order_list.append(order)

            out_dict = {}
            out_dict.update(
                {
                    "hm": hm,
                    "scores": scores,
                    "labels": labels,
                    "order": order,
                    "mask": mask,
                    "BEV_feat": x_up,
                    "H": H,
                    "W": W,
                }
            )
            if self.corner and self.corner_heads[0].training:
                out_dict.update({"corner_hm": corner_hm})
            out_dict_list.append(out_dict)

        self.batch_id = torch.from_numpy(np.indices((batch, self.obj_num * len(self.tasks)))[0]).to(
                labels
            )
        order_all = torch.cat(order_list,dim=1)

        ct_feat = (
            x_up.reshape(batch, -1, H * W)
            .transpose(2, 1)
            .contiguous()[self.batch_id, order_all]
        )  # B, 500, C

        # create position embedding for each center
        y_coor = order_all // W
        x_coor = order_all - y_coor * W
        y_coor, x_coor = y_coor.to(ct_feat), x_coor.to(ct_feat)
        y_coor, x_coor = y_coor / H, x_coor / W
        pos_features = torch.stack([x_coor, y_coor], dim=2)

        if self.parametric_embedding:
            ct_feat = self.query_embed.weight
            ct_feat = ct_feat.unsqueeze(0).expand(batch, -1, -1)

        # run transformer
        src = torch.cat(
            (
                x_up.reshape(batch, -1, H * W).transpose(2, 1).contiguous(),
            ),
            dim=1,
        )  # B ,sum(H*W), C
        spatial_shapes = torch.as_tensor(
            [(H, W)],
            dtype=torch.long,
            device=ct_feat.device,
        )
        level_start_index = torch.cat(
            (
                spatial_shapes.new_zeros((1,)),
                spatial_shapes.prod(1).cumsum(0)[:-1],
            )
        )

        if len(self.tasks) > 1:
            task_ids = torch.repeat_interleave(torch.arange(len(self.tasks)).repeat(batch,1), self.obj_num, dim=1).to(pos_features) # B, 500
            pos_features = torch.cat([pos_features, task_ids[:, :, None]],dim=-1)

        transformer_out = self.transformer_layer(
            ct_feat,
            self.pos_embedding,
            src,
            spatial_shapes,
            level_start_index,
            center_pos=pos_features,
        )  # (B,N,C)

        ct_feat = (
            transformer_out["ct_feat"].transpose(2, 1).contiguous()
        )  # B, C, 500

        for idx, task in enumerate(self.tasks):
            out_dict_list[idx]["ct_feat"] = ct_feat[:, :, idx * self.obj_num : (idx+1) * self.obj_num]

        return out_dict_list


@NECKS.register_module
class RPN_transformer_deformable_multitask_mtf(RPN_transformer_base_multitask):
    def __init__(
        self,
        layer_nums,  # [2,2,2]
        ds_num_filters,  # [128,256,64]
        num_input_features,  # 256
        transformer_config=None,
        hm_head_layer=2,
        corner_head_layer=2,
        corner=False,
        assign_label_window_size=1,
        tasks=[],
        use_gt_training=False,
        norm_cfg=None,
        name="RPN_transformer_deformable_multitask_mtf",
        logger=None,
        init_bias=-2.19,
        score_threshold=0.1,
        obj_num=500,
        frame=1,
        parametric_embedding=False,
        **kwargs
    ):
        super(RPN_transformer_deformable_multitask_mtf, self).__init__(
            layer_nums,
            ds_num_filters,
            num_input_features,
            transformer_config,
            hm_head_layer,
            corner_head_layer,
            corner,
            assign_label_window_size,
            tasks,
            use_gt_training,
            norm_cfg,
            logger,
            init_bias,
            score_threshold,
            obj_num,
        )
        self.frame = frame
        self.out = Sequential(
            nn.Conv2d(
                self._num_filters[0] * frame,
                self._num_filters[0],
                3,
                padding=1,
                bias=False,
            ),
            build_norm_layer(self._norm_cfg, self._num_filters[0])[1],
            nn.ReLU(),
        )
        self.mtf_attention = SpatialAttention_mtf()
        self.time_embedding = nn.Linear(1, self._num_filters[0])

        self.transformer_layer = Deform_Transformer(
            self._num_filters[-1] * 2,
            depth=transformer_config.depth,
            heads=transformer_config.heads,
            levels=2 + self.frame,
            dim_head=transformer_config.dim_head,
            mlp_dim=transformer_config.MLP_dim,
            dropout=transformer_config.DP_rate,
            out_attention=transformer_config.out_att,
            n_points=transformer_config.get("n_points", 9),
        )
        self.pos_embedding_type = transformer_config.get(
            "pos_embedding_type", "linear"
        )
        if self.pos_embedding_type == "linear":
            if len(self.tasks)>1:
                self.pos_embedding = nn.Linear(3, self._num_filters[-1] * 2)
            else:
                self.pos_embedding = nn.Linear(2, self._num_filters[-1] * 2)
        else:
            raise NotImplementedError()
        self.parametric_embedding = parametric_embedding
        if self.parametric_embedding:
            self.query_embed = nn.Embedding(self.obj_num * len(self.tasks), self._num_filters[-1] * 2)
            nn.init.uniform_(self.query_embed.weight, -1.0, 1.0)

        logger.info("Finish RPN_transformer_deformable Initialization")

    def forward(self, x, example=None):

        # FPN
        x = self.blocks[0](x)
        x_down = self.blocks[1](x)
        x_up = torch.cat([self.blocks[2](x_down), self.up(x)], dim=1)

        # take out the BEV feature on current frame
        x = torch.split(x, self.frame)
        x_up = torch.split(x_up, self.frame)
        x_down = torch.split(x_down, self.frame)
        x_prev = torch.stack([t[1:] for t in x_up], dim=0)  # B,K,C,H,W
        x = torch.stack([t[0] for t in x], dim=0)
        x_down = torch.stack([t[0] for t in x_down], dim=0)

        x_up = torch.stack([t[0] for t in x_up], dim=0)  # B,C,H,W
        # use spatial attention in current frame on previous feature
        x_prev_cat = self.mtf_attention(
            x_up, x_prev.reshape(x_up.shape[0], -1, x_up.shape[2], x_up.shape[3])
        )  # B,K*C,H,W
        # time embedding
        x_up_fuse = torch.cat((x_up, x_prev_cat), dim=1) + self.time_embedding(
            example["times"][:, :, None].to(x_up)
        ).reshape(x_up.shape[0], -1, 1, 1)
        # fuse mtf feature
        x_up_fuse = self.out(x_up_fuse)

        order_list = []
        out_dict_list = []
        for idx, task in enumerate(self.tasks):
            # heatmap head
            hm = self.hm_heads[idx](x_up_fuse)

            if self.corner and self.corner_heads[0].training:
                corner_hm = self.corner_heads[idx](x_up)
                corner_hm = torch.sigmoid(corner_hm)

            # find top K center location
            hm = torch.sigmoid(hm)
            batch, num_cls, H, W = hm.size()

            scores, labels = torch.max(hm.reshape(batch, num_cls, H * W), dim=1)  # b,H*W

            if self.use_gt_training and self.hm_heads[0].training:
                gt_inds = example["ind"][idx][:, (self.window_size // 2) :: self.window_size]
                gt_masks = example["mask"][idx][
                    :, (self.window_size // 2) :: self.window_size
                ]
                batch_id_gt = torch.from_numpy(np.indices((batch, gt_inds.shape[1]))[0]).to(
                    labels
                )
                scores[batch_id_gt, gt_inds] = scores[batch_id_gt, gt_inds] + gt_masks
                order = scores.sort(1, descending=True)[1]
                order = order[:, : self.obj_num]
                scores[batch_id_gt, gt_inds] = scores[batch_id_gt, gt_inds] - gt_masks
            else:
                order = scores.sort(1, descending=True)[1]
                order = order[:, : self.obj_num]

            scores = torch.gather(scores, 1, order)
            labels = torch.gather(labels, 1, order)
            mask = scores > self.score_threshold
            order_list.append(order)

            out_dict = {}
            out_dict.update(
                {
                    "hm": hm,
                    "scores": scores,
                    "labels": labels,
                    "order": order,
                    "mask": mask,
                    "BEV_feat": x_up,
                    "H": H,
                    "W": W,
                }
            )
            if self.corner and self.corner_heads[0].training:
                out_dict.update({"corner_hm": corner_hm})
            out_dict_list.append(out_dict)

        self.batch_id = torch.from_numpy(np.indices((batch, self.obj_num * len(self.tasks)))[0]).to(
                labels
            )
        order_all = torch.cat(order_list,dim=1)

        ct_feat = (
            x_up.reshape(batch, -1, H * W)
            .transpose(2, 1)
            .contiguous()[self.batch_id, order_all]
        )  # B, 500, C

        # create position embedding for each center
        y_coor = order_all // W
        x_coor = order_all - y_coor * W
        y_coor, x_coor = y_coor.to(ct_feat), x_coor.to(ct_feat)
        y_coor, x_coor = y_coor / H, x_coor / W
        pos_features = torch.stack([x_coor, y_coor], dim=2)

        if self.parametric_embedding:
            ct_feat = self.query_embed.weight
            ct_feat = ct_feat.unsqueeze(0).expand(batch, -1, -1)

        # run transformer
        # src = torch.cat(
        #     (
        #         x_up.reshape(batch, -1, H * W).transpose(2, 1).contiguous(),
        #         x.reshape(batch, -1, (H * W) // 4).transpose(2, 1).contiguous(),
        #         x_down.reshape(batch, -1, (H * W) // 16)
        #         .transpose(2, 1)
        #         .contiguous(),
        #     ),
        #     dim=1,
        # )  # B ,sum(H*W), C
        src_list = [
            x_up.reshape(batch, -1, H * W).transpose(2, 1).contiguous(),
            x.reshape(batch, -1, (H * W) // 4).transpose(2, 1).contiguous(),
            x_down.reshape(batch, -1, (H * W) // 16)
            .transpose(2, 1)
            .contiguous(),
        ]
        for frame in range(x_prev.shape[1]):
            src_list.append(
                x_prev[:, frame]
                .reshape(batch, -1, (H * W))
                .transpose(2, 1)
                .contiguous()
            )
        src = torch.cat(src_list, dim=1)  # B ,sum(H*W), C
        spatial_list = [(H, W), (H // 2, W // 2), (H // 4, W // 4)]
        spatial_list += [(H, W) for frame in range(x_prev.shape[1])]
        spatial_shapes = torch.as_tensor(
            spatial_list,
            dtype=torch.long,
            device=ct_feat.device,
        )
        level_start_index = torch.cat(
            (
                spatial_shapes.new_zeros((1,)),
                spatial_shapes.prod(1).cumsum(0)[:-1],
            )
        )

        if len(self.tasks) > 1:
            task_ids = torch.repeat_interleave(torch.arange(len(self.tasks)).repeat(batch,1), self.obj_num, dim=1).to(pos_features) # B, 500
            pos_features = torch.cat([pos_features, task_ids[:, :, None]],dim=-1)

        transformer_out = self.transformer_layer(
            ct_feat,
            self.pos_embedding,
            src,
            spatial_shapes,
            level_start_index,
            center_pos=pos_features,
        )  # (B,N,C)

        ct_feat = (
            transformer_out["ct_feat"].transpose(2, 1).contiguous()
        )  # B, C, 500

        for idx, task in enumerate(self.tasks):
            out_dict_list[idx]["ct_feat"] = ct_feat[:, :, idx * self.obj_num : (idx+1) * self.obj_num]

        return out_dict_list


@NECKS.register_module
class RPN_transformer_deformable_multitask_mtf_custom(RPN_transformer_base_multitask):
    def __init__(
        self,
        layer_nums,  # [2,2,2]
        ds_num_filters,  # [128,256,64]
        num_input_features,  # 256
        transformer_config=None,
        hm_head_layer=2,
        corner_head_layer=2,
        corner=False,
        assign_label_window_size=1,
        tasks=[],
        use_gt_training=False,
        norm_cfg=None,
        name="RPN_transformer_deformable_multitask_mtf",
        logger=None,
        init_bias=-2.19,
        score_threshold=0.1,
        obj_num=500,
        frame=1,
        parametric_embedding=False,
        **kwargs
    ):
        super(RPN_transformer_deformable_multitask_mtf_custom, self).__init__(
            layer_nums,
            ds_num_filters,
            num_input_features,
            transformer_config,
            hm_head_layer,
            corner_head_layer,
            corner,
            assign_label_window_size,
            tasks,
            use_gt_training,
            norm_cfg,
            logger,
            init_bias,
            score_threshold,
            obj_num,
        )
        self.frame = frame
        self.out = Sequential(
            nn.Conv2d(
                self._num_filters[0] * frame,
                self._num_filters[0],
                3,
                padding=1,
                bias=False,
            ),
            build_norm_layer(self._norm_cfg, self._num_filters[0])[1],
            nn.ReLU(),
        )
        self.mtf_attention = SpatialAttention_mtf_custom(frame*ds_num_filters[0], frame)
        # self.mtf_attention = SpatialAttention_mtf()
        self.time_embedding = nn.Linear(1, self._num_filters[0])

        self.transformer_layer = Deform_Transformer(
            self._num_filters[-1] * 2,
            depth=transformer_config.depth,
            heads=transformer_config.heads,
            levels=2 + self.frame,
            dim_head=transformer_config.dim_head,
            mlp_dim=transformer_config.MLP_dim,
            dropout=transformer_config.DP_rate,
            out_attention=transformer_config.out_att,
            n_points=transformer_config.get("n_points", 9),
        )
        self.pos_embedding_type = transformer_config.get(
            "pos_embedding_type", "linear"
        )
        if self.pos_embedding_type == "linear":
            if len(self.tasks)>1:
                self.pos_embedding = nn.Linear(3, self._num_filters[-1] * 2)
            else:
                self.pos_embedding = nn.Linear(2, self._num_filters[-1] * 2)
        else:
            raise NotImplementedError()
        self.parametric_embedding = parametric_embedding
        if self.parametric_embedding:
            self.query_embed = nn.Embedding(self.obj_num * len(self.tasks), self._num_filters[-1] * 2)
            nn.init.uniform_(self.query_embed.weight, -1.0, 1.0)

        logger.info("Finish RPN_transformer_deformable Initialization")

    def forward(self, x, example=None):

        # FPN
        x = self.blocks[0](x)
        x_down = self.blocks[1](x)
        x_up = torch.cat([self.blocks[2](x_down), self.up(x)], dim=1)

        # take out the BEV feature on current frame
        x = torch.split(x, self.frame)
        x_up = torch.split(x_up, self.frame)
        x_down = torch.split(x_down, self.frame)
        x_prev = torch.stack([t[1:] for t in x_up], dim=0)  # B,K,C,H,W
        x = torch.stack([t[0] for t in x], dim=0)
        x_down = torch.stack([t[0] for t in x_down], dim=0)

        x_up = torch.stack([t[0] for t in x_up], dim=0)  # B,C,H,W
        # # use spatial attention in current frame on previous feature
        # x_prev_cat = self.mtf_attention(
        #     x_up, x_prev.reshape(x_up.shape[0], -1, x_up.shape[2], x_up.shape[3])
        # )  # B,K*C,H,W
        # # time embedding
        # x_up_fuse = torch.cat((x_up, x_prev_cat), dim=1) + self.time_embedding(
        #     example["times"][:, :, None].to(x_up)
        # ).reshape(x_up.shape[0], -1, 1, 1)
        # # fuse mtf feature
        # x_up_fuse = self.out(x_up_fuse)

        ### self-design mtf network
        B, K, C, H, W = x_prev.shape
        x_frames_cat = torch.cat((x_up, x_prev.reshape(B,-1,H,W)), dim=1) + self.time_embedding(
            example["times"][:, :, None].to(x_up)
        ).reshape(x_up.shape[0], -1, 1, 1)
        x_frames_att = self.mtf_attention(x_frames_cat)
        x_frames_att = x_frames_att.unsqueeze(2).expand(-1,-1,C,-1,-1).reshape(B,-1,H,W)
        x_up_fuse = self.out(x_frames_cat * x_frames_att)


        order_list = []
        out_dict_list = []
        for idx, task in enumerate(self.tasks):
            # heatmap head
            hm = self.hm_heads[idx](x_up_fuse)

            if self.corner and self.corner_heads[0].training:
                corner_hm = self.corner_heads[idx](x_up)
                corner_hm = torch.sigmoid(corner_hm)

            # find top K center location
            hm = torch.sigmoid(hm)
            batch, num_cls, H, W = hm.size()

            scores, labels = torch.max(hm.reshape(batch, num_cls, H * W), dim=1)  # b,H*W

            if self.use_gt_training and self.hm_heads[0].training:
                gt_inds = example["ind"][idx][:, (self.window_size // 2) :: self.window_size]
                gt_masks = example["mask"][idx][
                    :, (self.window_size // 2) :: self.window_size
                ]
                batch_id_gt = torch.from_numpy(np.indices((batch, gt_inds.shape[1]))[0]).to(
                    labels
                )
                scores[batch_id_gt, gt_inds] = scores[batch_id_gt, gt_inds] + gt_masks
                order = scores.sort(1, descending=True)[1]
                order = order[:, : self.obj_num]
                scores[batch_id_gt, gt_inds] = scores[batch_id_gt, gt_inds] - gt_masks
            else:
                order = scores.sort(1, descending=True)[1]
                order = order[:, : self.obj_num]

            scores = torch.gather(scores, 1, order)
            labels = torch.gather(labels, 1, order)
            mask = scores > self.score_threshold
            order_list.append(order)

            out_dict = {}
            out_dict.update(
                {
                    "hm": hm,
                    "scores": scores,
                    "labels": labels,
                    "order": order,
                    "mask": mask,
                    "BEV_feat": x_up,
                    "H": H,
                    "W": W,
                }
            )
            if self.corner and self.corner_heads[0].training:
                out_dict.update({"corner_hm": corner_hm})
            out_dict_list.append(out_dict)

        self.batch_id = torch.from_numpy(np.indices((batch, self.obj_num * len(self.tasks)))[0]).to(
                labels
            )
        order_all = torch.cat(order_list,dim=1)

        ct_feat = (
            x_up.reshape(batch, -1, H * W)
            .transpose(2, 1)
            .contiguous()[self.batch_id, order_all]
        )  # B, 500, C

        # create position embedding for each center
        y_coor = order_all // W
        x_coor = order_all - y_coor * W
        y_coor, x_coor = y_coor.to(ct_feat), x_coor.to(ct_feat)
        y_coor, x_coor = y_coor / H, x_coor / W
        pos_features = torch.stack([x_coor, y_coor], dim=2)

        if self.parametric_embedding:
            ct_feat = self.query_embed.weight
            ct_feat = ct_feat.unsqueeze(0).expand(batch, -1, -1)

        # run transformer
        # src = torch.cat(
        #     (
        #         x_up.reshape(batch, -1, H * W).transpose(2, 1).contiguous(),
        #         x.reshape(batch, -1, (H * W) // 4).transpose(2, 1).contiguous(),
        #         x_down.reshape(batch, -1, (H * W) // 16)
        #         .transpose(2, 1)
        #         .contiguous(),
        #     ),
        #     dim=1,
        # )  # B ,sum(H*W), C
        src_list = [
            x_up.reshape(batch, -1, H * W).transpose(2, 1).contiguous(),
            x.reshape(batch, -1, (H * W) // 4).transpose(2, 1).contiguous(),
            x_down.reshape(batch, -1, (H * W) // 16)
            .transpose(2, 1)
            .contiguous(),
        ]
        for frame in range(x_prev.shape[1]):
            src_list.append(
                x_prev[:, frame]
                .reshape(batch, -1, (H * W))
                .transpose(2, 1)
                .contiguous()
            )
        src = torch.cat(src_list, dim=1)  # B ,sum(H*W), C
        spatial_list = [(H, W), (H // 2, W // 2), (H // 4, W // 4)]
        spatial_list += [(H, W) for frame in range(x_prev.shape[1])]
        spatial_shapes = torch.as_tensor(
            spatial_list,
            dtype=torch.long,
            device=ct_feat.device,
        )
        level_start_index = torch.cat(
            (
                spatial_shapes.new_zeros((1,)),
                spatial_shapes.prod(1).cumsum(0)[:-1],
            )
        )

        if len(self.tasks) > 1:
            task_ids = torch.repeat_interleave(torch.arange(len(self.tasks)).repeat(batch,1), self.obj_num, dim=1).to(pos_features) # B, 500
            pos_features = torch.cat([pos_features, task_ids[:, :, None]],dim=-1)

        transformer_out = self.transformer_layer(
            ct_feat,
            self.pos_embedding,
            src,
            spatial_shapes,
            level_start_index,
            center_pos=pos_features,
        )  # (B,N,C)

        ct_feat = (
            transformer_out["ct_feat"].transpose(2, 1).contiguous()
        )  # B, C, 500

        for idx, task in enumerate(self.tasks):
            out_dict_list[idx]["ct_feat"] = ct_feat[:, :, idx * self.obj_num : (idx+1) * self.obj_num]

        return out_dict_list


@NECKS.register_module
class RPN_transformer_deformable_multitask_mtf_custom2(RPN_transformer_base_multitask):
    def __init__(
        self,
        layer_nums,  # [2,2,2]
        ds_num_filters,  # [128,256,64]
        num_input_features,  # 256
        transformer_config=None,
        hm_head_layer=2,
        corner_head_layer=2,
        corner=False,
        assign_label_window_size=1,
        tasks=[],
        use_gt_training=False,
        norm_cfg=None,
        name="RPN_transformer_deformable_multitask_mtf",
        logger=None,
        init_bias=-2.19,
        score_threshold=0.1,
        obj_num=500,
        frame=1,
        parametric_embedding=False,
        **kwargs
    ):
        super(RPN_transformer_deformable_multitask_mtf_custom2, self).__init__(
            layer_nums,
            ds_num_filters,
            num_input_features,
            transformer_config,
            hm_head_layer,
            corner_head_layer,
            corner,
            assign_label_window_size,
            tasks,
            use_gt_training,
            norm_cfg,
            logger,
            init_bias,
            score_threshold,
            obj_num,
        )
        self.frame = frame
        self.out = Sequential(
            nn.Conv2d(
                self._num_filters[0] * frame,
                self._num_filters[0],
                3,
                padding=1,
                bias=False,
            ),
            build_norm_layer(self._norm_cfg, self._num_filters[0])[1],
            nn.ReLU(),
        )
        self.mtf_attention = SpatialAttention_mtf_custom2(ds_num_filters[0], frame)
        # self.mtf_attention = SpatialAttention_mtf()
        self.time_embedding = nn.Linear(1, self._num_filters[0])

        self.transformer_layer = Deform_Transformer(
            self._num_filters[-1] * 2,
            depth=transformer_config.depth,
            heads=transformer_config.heads,
            levels=2 + self.frame,
            dim_head=transformer_config.dim_head,
            mlp_dim=transformer_config.MLP_dim,
            dropout=transformer_config.DP_rate,
            out_attention=transformer_config.out_att,
            n_points=transformer_config.get("n_points", 9),
        )
        self.pos_embedding_type = transformer_config.get(
            "pos_embedding_type", "linear"
        )
        if self.pos_embedding_type == "linear":
            if len(self.tasks)>1:
                self.pos_embedding = nn.Linear(3, self._num_filters[-1] * 2)
            else:
                self.pos_embedding = nn.Linear(2, self._num_filters[-1] * 2)
        else:
            raise NotImplementedError()
        self.parametric_embedding = parametric_embedding
        if self.parametric_embedding:
            self.query_embed = nn.Embedding(self.obj_num * len(self.tasks), self._num_filters[-1] * 2)
            nn.init.uniform_(self.query_embed.weight, -1.0, 1.0)

        logger.info("Finish RPN_transformer_deformable Initialization")

    def forward(self, x, example=None):

        # FPN
        x = self.blocks[0](x)
        x_down = self.blocks[1](x)
        x_up = torch.cat([self.blocks[2](x_down), self.up(x)], dim=1)

        # take out the BEV feature on current frame
        x = torch.split(x, self.frame)
        x_up = torch.split(x_up, self.frame)
        x_down = torch.split(x_down, self.frame)
        x_prev = torch.stack([t[1:] for t in x_up], dim=0)  # B,K,C,H,W
        x = torch.stack([t[0] for t in x], dim=0)
        x_down = torch.stack([t[0] for t in x_down], dim=0)

        x_up = torch.stack([t[0] for t in x_up], dim=0)  # B,C,H,W
        # # use spatial attention in current frame on previous feature
        # x_prev_cat = self.mtf_attention(
        #     x_up, x_prev.reshape(x_up.shape[0], -1, x_up.shape[2], x_up.shape[3])
        # )  # B,K*C,H,W
        # # time embedding
        # x_up_fuse = torch.cat((x_up, x_prev_cat), dim=1) + self.time_embedding(
        #     example["times"][:, :, None].to(x_up)
        # ).reshape(x_up.shape[0], -1, 1, 1)
        # # fuse mtf feature
        # x_up_fuse = self.out(x_up_fuse)

        ### self-design mtf network
        B, K, C, H, W = x_prev.shape

        x_frames_cat = torch.cat((x_up.unsqueeze(1), x_prev), dim=1) + self.time_embedding(
            example["times"][:, :, None].to(x_up))[:,:,:,None,None]
        x_frames_att = self.mtf_attention(x_frames_cat)
        x_frames_fused = (x_frames_cat * x_frames_att.unsqueeze(2)).reshape(B,-1,H,W)
        x_up_fuse = self.out(x_frames_fused)


        order_list = []
        out_dict_list = []
        for idx, task in enumerate(self.tasks):
            # heatmap head
            hm = self.hm_heads[idx](x_up_fuse)

            if self.corner and self.corner_heads[0].training:
                corner_hm = self.corner_heads[idx](x_up)
                corner_hm = torch.sigmoid(corner_hm)

            # find top K center location
            hm = torch.sigmoid(hm)
            batch, num_cls, H, W = hm.size()

            scores, labels = torch.max(hm.reshape(batch, num_cls, H * W), dim=1)  # b,H*W

            if self.use_gt_training and self.hm_heads[0].training:
                gt_inds = example["ind"][idx][:, (self.window_size // 2) :: self.window_size]
                gt_masks = example["mask"][idx][
                    :, (self.window_size // 2) :: self.window_size
                ]
                batch_id_gt = torch.from_numpy(np.indices((batch, gt_inds.shape[1]))[0]).to(
                    labels
                )
                scores[batch_id_gt, gt_inds] = scores[batch_id_gt, gt_inds] + gt_masks
                order = scores.sort(1, descending=True)[1]
                order = order[:, : self.obj_num]
                scores[batch_id_gt, gt_inds] = scores[batch_id_gt, gt_inds] - gt_masks
            else:
                order = scores.sort(1, descending=True)[1]
                order = order[:, : self.obj_num]

            scores = torch.gather(scores, 1, order)
            labels = torch.gather(labels, 1, order)
            mask = scores > self.score_threshold
            order_list.append(order)

            out_dict = {}
            out_dict.update(
                {
                    "hm": hm,
                    "scores": scores,
                    "labels": labels,
                    "order": order,
                    "mask": mask,
                    "BEV_feat": x_up,
                    "H": H,
                    "W": W,
                }
            )
            if self.corner and self.corner_heads[0].training:
                out_dict.update({"corner_hm": corner_hm})
            out_dict_list.append(out_dict)

        self.batch_id = torch.from_numpy(np.indices((batch, self.obj_num * len(self.tasks)))[0]).to(
                labels
            )
        order_all = torch.cat(order_list,dim=1)

        ct_feat = (
            x_up.reshape(batch, -1, H * W)
            .transpose(2, 1)
            .contiguous()[self.batch_id, order_all]
        )  # B, 500, C

        # create position embedding for each center
        y_coor = order_all // W
        x_coor = order_all - y_coor * W
        y_coor, x_coor = y_coor.to(ct_feat), x_coor.to(ct_feat)
        y_coor, x_coor = y_coor / H, x_coor / W
        pos_features = torch.stack([x_coor, y_coor], dim=2)

        if self.parametric_embedding:
            ct_feat = self.query_embed.weight
            ct_feat = ct_feat.unsqueeze(0).expand(batch, -1, -1)

        # run transformer
        # src = torch.cat(
        #     (
        #         x_up.reshape(batch, -1, H * W).transpose(2, 1).contiguous(),
        #         x.reshape(batch, -1, (H * W) // 4).transpose(2, 1).contiguous(),
        #         x_down.reshape(batch, -1, (H * W) // 16)
        #         .transpose(2, 1)
        #         .contiguous(),
        #     ),
        #     dim=1,
        # )  # B ,sum(H*W), C
        src_list = [
            x_up.reshape(batch, -1, H * W).transpose(2, 1).contiguous(),
            x.reshape(batch, -1, (H * W) // 4).transpose(2, 1).contiguous(),
            x_down.reshape(batch, -1, (H * W) // 16)
            .transpose(2, 1)
            .contiguous(),
        ]
        for frame in range(x_prev.shape[1]):
            src_list.append(
                x_prev[:, frame]
                .reshape(batch, -1, (H * W))
                .transpose(2, 1)
                .contiguous()
            )
        src = torch.cat(src_list, dim=1)  # B ,sum(H*W), C
        spatial_list = [(H, W), (H // 2, W // 2), (H // 4, W // 4)]
        spatial_list += [(H, W) for frame in range(x_prev.shape[1])]
        spatial_shapes = torch.as_tensor(
            spatial_list,
            dtype=torch.long,
            device=ct_feat.device,
        )
        level_start_index = torch.cat(
            (
                spatial_shapes.new_zeros((1,)),
                spatial_shapes.prod(1).cumsum(0)[:-1],
            )
        )

        if len(self.tasks) > 1:
            task_ids = torch.repeat_interleave(torch.arange(len(self.tasks)).repeat(batch,1), self.obj_num, dim=1).to(pos_features) # B, 500
            pos_features = torch.cat([pos_features, task_ids[:, :, None]],dim=-1)

        transformer_out = self.transformer_layer(
            ct_feat,
            self.pos_embedding,
            src,
            spatial_shapes,
            level_start_index,
            center_pos=pos_features,
        )  # (B,N,C)

        ct_feat = (
            transformer_out["ct_feat"].transpose(2, 1).contiguous()
        )  # B, C, 500

        for idx, task in enumerate(self.tasks):
            out_dict_list[idx]["ct_feat"] = ct_feat[:, :, idx * self.obj_num : (idx+1) * self.obj_num]

        return out_dict_list


@NECKS.register_module
class RPN_transformer_deformable_multitask_mtf_custom3(RPN_transformer_base_multitask):
    ###  not change SpatialAttention_mtf, but pillarnet neck
    def __init__(
        self,
        layer_nums,  # [2,2,2]
        ds_num_filters,  # [128,256,64]
        num_input_features,  # 256
        transformer_config=None,
        hm_head_layer=2,
        corner_head_layer=2,
        corner=False,
        assign_label_window_size=1,
        tasks=[],
        use_gt_training=False,
        norm_cfg=None,
        name="RPN_transformer_deformable_multitask_mtf",
        logger=None,
        init_bias=-2.19,
        score_threshold=0.1,
        obj_num=500,
        frame=1,
        parametric_embedding=False,
        **kwargs
    ):
        super(RPN_transformer_deformable_multitask_mtf_custom3, self).__init__(
            layer_nums,
            ds_num_filters,
            num_input_features,
            transformer_config,
            hm_head_layer,
            corner_head_layer,
            corner,
            assign_label_window_size,
            tasks,
            use_gt_training,
            norm_cfg,
            logger,
            init_bias,
            score_threshold,
            obj_num,
        )
        self.frame = frame
        self.out = Sequential(
            nn.Conv2d(
                self._num_filters[0] * frame,
                self._num_filters[0],
                3,
                padding=1,
                bias=False,
            ),
            build_norm_layer(self._norm_cfg, self._num_filters[0])[1],
            nn.ReLU(),
        )
        # self.mtf_attention = SpatialAttention_mtf_custom2(ds_num_filters[0], frame)
        self.mtf_attention = SpatialAttention_mtf()
        self.time_embedding = nn.Linear(1, self._num_filters[0])

        self.transformer_layer = Deform_Transformer(
            self._num_filters[-1] * 2,
            depth=transformer_config.depth,
            heads=transformer_config.heads,
            levels=2 + self.frame,
            dim_head=transformer_config.dim_head,
            mlp_dim=transformer_config.MLP_dim,
            dropout=transformer_config.DP_rate,
            out_attention=transformer_config.out_att,
            n_points=transformer_config.get("n_points", 9),
        )
        self.pos_embedding_type = transformer_config.get(
            "pos_embedding_type", "linear"
        )
        if self.pos_embedding_type == "linear":
            if len(self.tasks)>1:
                self.pos_embedding = nn.Linear(3, self._num_filters[-1] * 2)
            else:
                self.pos_embedding = nn.Linear(2, self._num_filters[-1] * 2)
        else:
            raise NotImplementedError()
        self.parametric_embedding = parametric_embedding
        if self.parametric_embedding:
            self.query_embed = nn.Embedding(self.obj_num * len(self.tasks), self._num_filters[-1] * 2)
            nn.init.uniform_(self.query_embed.weight, -1.0, 1.0)

        logger.info("Finish RPN_transformer_deformable Initialization")

    def forward(self, x, example=None):
        ### add for visualize featuremap experiment
        x_in = x

        # FPN for pillarnet
        x = self.blocks[0](x)
        x_down = self.blocks[1](x)
        # x_down_up = F.interpolate(x_down, scale_factor=2, mode='bilinear')
        x_down_up = self.blocks[2](x_down)
        x_mix = self.mix(x + x_down_up)
        x_up = self.up(x_mix)

        # # fpn original
        # x = self.blocks[0](x)
        # x_down = self.blocks[1](x)
        # x_up = torch.cat([self.blocks[2](x_down), self.up(x)], dim=1)


        # take out the BEV feature on current frame
        x = torch.split(x, self.frame)
        x_up = torch.split(x_up, self.frame)
        x_down = torch.split(x_down, self.frame)
        x_prev = torch.stack([t[1:] for t in x_up], dim=0)  # B,K,C,H,W
        x = torch.stack([t[0] for t in x], dim=0)
        x_down = torch.stack([t[0] for t in x_down], dim=0)

        x_up = torch.stack([t[0] for t in x_up], dim=0)  # B,C,H,W
        # # use spatial attention in current frame on previous feature
        # x_prev_cat = self.mtf_attention(
        #     x_up, x_prev.reshape(x_up.shape[0], -1, x_up.shape[2], x_up.shape[3])
        # )  # B,K*C,H,W
        # # time embedding
        # x_up_fuse = torch.cat((x_up, x_prev_cat), dim=1) + self.time_embedding(
        #     example["times"][:, :, None].to(x_up)
        # ).reshape(x_up.shape[0], -1, 1, 1)
        # # fuse mtf feature
        # x_up_fuse = self.out(x_up_fuse)

        ### self-design mtf network
        B, K, C, H, W = x_prev.shape

        # x_frames_cat = torch.cat((x_up.unsqueeze(1), x_prev), dim=1) + self.time_embedding(
        #     example["times"][:, :, None].to(x_up))[:,:,:,None,None]
        # x_frames_att = self.mtf_attention(x_frames_cat)
        # use spatial attention in current frame on previous feature
        x_prev_cat = self.mtf_attention(
            x_up, x_prev.reshape(x_up.shape[0], -1, x_up.shape[2], x_up.shape[3])
        )  # B,K*C,H,W
        # time embedding
        x_up_fuse = torch.cat((x_up, x_prev_cat), dim=1) + self.time_embedding(
            example["times"][:, :, None].to(x_up)
        ).reshape(x_up.shape[0], -1, 1, 1)
        # fuse mtf feature
        x_up_fuse = self.out(x_up_fuse)

        order_list = []
        out_dict_list = []
        for idx, task in enumerate(self.tasks):
            # heatmap head
            hm = self.hm_heads[idx](x_up_fuse)

            if self.corner and self.corner_heads[0].training:
                corner_hm = self.corner_heads[idx](x_up)
                corner_hm = torch.sigmoid(corner_hm)

            # find top K center location
            hm = torch.sigmoid(hm)
            batch, num_cls, H, W = hm.size()

            scores, labels = torch.max(hm.reshape(batch, num_cls, H * W), dim=1)  # b,H*W

            if self.use_gt_training and self.hm_heads[0].training:
                gt_inds = example["ind"][idx][:, (self.window_size // 2) :: self.window_size]
                gt_masks = example["mask"][idx][
                    :, (self.window_size // 2) :: self.window_size
                ]
                batch_id_gt = torch.from_numpy(np.indices((batch, gt_inds.shape[1]))[0]).to(
                    labels
                )
                scores[batch_id_gt, gt_inds] = scores[batch_id_gt, gt_inds] + gt_masks
                order = scores.sort(1, descending=True)[1]
                order = order[:, : self.obj_num]
                scores[batch_id_gt, gt_inds] = scores[batch_id_gt, gt_inds] - gt_masks
            else:
                order = scores.sort(1, descending=True)[1]
                order = order[:, : self.obj_num]

            scores = torch.gather(scores, 1, order)
            labels = torch.gather(labels, 1, order)
            mask = scores > self.score_threshold
            order_list.append(order)

            out_dict = {}
            out_dict.update(
                {
                    "hm": hm,
                    "scores": scores,
                    "labels": labels,
                    "order": order,
                    "mask": mask,
                    "BEV_feat": x_up,
                    "H": H,
                    "W": W,
                }
            )
            if self.corner and self.corner_heads[0].training:
                out_dict.update({"corner_hm": corner_hm})
            out_dict_list.append(out_dict)

        ## add for heatmap visualize experiment
        ## featuremaps render
        sample_token = example['metadata'][0]['token']
        feature_dict = dict(
            x_in=[x_in[0], 8],
            x=[x[0], 8],
            x_down=[x_down[0], 16],
            x_down_up=[x_down_up[0], 8],
            x_mix=[x_mix[0], 8],
            x_up=[x_up[0], 4],

        )
        feature_save = False
        heatmap_save = True
        multiscale_featuremaps_render(feature_dict, sample_token, save=feature_save)
        multichannel_featuremaps_render(x_up[0], 4, range(10), sample_token, save=feature_save)

        heatmaps = torch.cat([out_task['hm'][0] for out_task in out_dict_list], dim=0)
        heatmaps_render(heatmaps, sample_token, save=heatmap_save)
        heatmaps_show = change_for_vis(heatmaps, mode='mean')

        self.batch_id = torch.from_numpy(np.indices((batch, self.obj_num * len(self.tasks)))[0]).to(
                labels
            )
        order_all = torch.cat(order_list,dim=1)

        ct_feat = (
            x_up.reshape(batch, -1, H * W)
            .transpose(2, 1)
            .contiguous()[self.batch_id, order_all]
        )  # B, 500, C

        # create position embedding for each center
        y_coor = order_all // W
        x_coor = order_all - y_coor * W
        y_coor, x_coor = y_coor.to(ct_feat), x_coor.to(ct_feat)
        y_coor, x_coor = y_coor / H, x_coor / W
        pos_features = torch.stack([x_coor, y_coor], dim=2)

        if self.parametric_embedding:
            ct_feat = self.query_embed.weight
            ct_feat = ct_feat.unsqueeze(0).expand(batch, -1, -1)

        # run transformer
        # src = torch.cat(
        #     (
        #         x_up.reshape(batch, -1, H * W).transpose(2, 1).contiguous(),
        #         x.reshape(batch, -1, (H * W) // 4).transpose(2, 1).contiguous(),
        #         x_down.reshape(batch, -1, (H * W) // 16)
        #         .transpose(2, 1)
        #         .contiguous(),
        #     ),
        #     dim=1,
        # )  # B ,sum(H*W), C
        src_list = [
            x_up.reshape(batch, -1, H * W).transpose(2, 1).contiguous(),
            x.reshape(batch, -1, (H * W) // 4).transpose(2, 1).contiguous(),
            x_down.reshape(batch, -1, (H * W) // 16)
            .transpose(2, 1)
            .contiguous(),
        ]
        for frame in range(x_prev.shape[1]):
            src_list.append(
                x_prev[:, frame]
                .reshape(batch, -1, (H * W))
                .transpose(2, 1)
                .contiguous()
            )
        src = torch.cat(src_list, dim=1)  # B ,sum(H*W), C
        spatial_list = [(H, W), (H // 2, W // 2), (H // 4, W // 4)]
        spatial_list += [(H, W) for frame in range(x_prev.shape[1])]
        spatial_shapes = torch.as_tensor(
            spatial_list,
            dtype=torch.long,
            device=ct_feat.device,
        )
        level_start_index = torch.cat(
            (
                spatial_shapes.new_zeros((1,)),
                spatial_shapes.prod(1).cumsum(0)[:-1],
            )
        )

        if len(self.tasks) > 1:
            task_ids = torch.repeat_interleave(torch.arange(len(self.tasks)).repeat(batch,1), self.obj_num, dim=1).to(pos_features) # B, 500
            pos_features = torch.cat([pos_features, task_ids[:, :, None]],dim=-1)

        transformer_out = self.transformer_layer(
            ct_feat,
            self.pos_embedding,
            src,
            spatial_shapes,
            level_start_index,
            center_pos=pos_features,
        )  # (B,N,C)

        ct_feat = (
            transformer_out["ct_feat"].transpose(2, 1).contiguous()
        )  # B, C, 500

        for idx, task in enumerate(self.tasks):
            out_dict_list[idx]["ct_feat"] = ct_feat[:, :, idx * self.obj_num : (idx+1) * self.obj_num]

        return out_dict_list


@NECKS.register_module
class RPN_transformer_deformable_multitask_mtf_refactor(RPN_transformer_base_multitask_refactor):
    def __init__(
        self,
        ds_num_filters,
        multiscale_neck=None,
        transformer_config=None,
        hm_head_layer=2,
        corner_head_layer=2,
        corner=False,
        assign_label_window_size=1,
        tasks=[],
        use_gt_training=False,
        norm_cfg=None,
        name="RPN_transformer_deformable_multitask_mtf_refactor",
        logger=None,
        init_bias=-2.19,
        score_threshold=0.1,
        obj_num=500,
        frame=1,
        parametric_embedding=False,
        **kwargs
    ):
        super(RPN_transformer_deformable_multitask_mtf_refactor, self).__init__(
            ds_num_filters,
            multiscale_neck,
            transformer_config,
            hm_head_layer,
            corner_head_layer,
            corner,
            assign_label_window_size,
            tasks,
            use_gt_training,
            norm_cfg,
            logger,
            init_bias,
            score_threshold,
            obj_num,
        )
        self.frame = frame
        self.out = Sequential(
            nn.Conv2d(
                self._num_filters[0] * frame,
                self._num_filters[0],
                3,
                padding=1,
                bias=False,
            ),
            build_norm_layer(self._norm_cfg, self._num_filters[0])[1],
            nn.ReLU(),
        )
        # self.mtf_attention = SpatialAttention_mtf_custom2(ds_num_filters[0], frame)
        self.mtf_attention = SpatialAttention_mtf()
        self.time_embedding = nn.Linear(1, self._num_filters[0])

        self.transformer_layer = Deform_Transformer(
            self._num_filters[-1] * 2,
            depth=transformer_config.depth,
            heads=transformer_config.heads,
            levels=2 + self.frame,
            dim_head=transformer_config.dim_head,
            mlp_dim=transformer_config.MLP_dim,
            dropout=transformer_config.DP_rate,
            out_attention=transformer_config.out_att,
            n_points=transformer_config.get("n_points", 9),
        )
        self.pos_embedding_type = transformer_config.get(
            "pos_embedding_type", "linear"
        )
        if self.pos_embedding_type == "linear":
            if len(self.tasks)>1:
                self.pos_embedding = nn.Linear(3, self._num_filters[-1] * 2)
            else:
                self.pos_embedding = nn.Linear(2, self._num_filters[-1] * 2)
        else:
            raise NotImplementedError()
        self.parametric_embedding = parametric_embedding
        if self.parametric_embedding:
            self.query_embed = nn.Embedding(self.obj_num * len(self.tasks), self._num_filters[-1] * 2)
            nn.init.uniform_(self.query_embed.weight, -1.0, 1.0)

        logger.info("Finish RPN_transformer_deformable Initialization")

    def forward(self, x, example=None):
        ### add for visualize featuremap experiment
        x_in = x
        # if hasattr(self, "multiscale_neck") and self.multiscale_neck is not None:
        x, x_down, x_up = self.multiscale_neck(x)

        # take out the BEV feature on current frame
        x = torch.split(x, self.frame)
        x_up = torch.split(x_up, self.frame)
        x_down = torch.split(x_down, self.frame)
        x_prev = torch.stack([t[1:] for t in x_up], dim=0)  # B,K,C,H,W
        x = torch.stack([t[0] for t in x], dim=0)
        x_down = torch.stack([t[0] for t in x_down], dim=0)

        x_up = torch.stack([t[0] for t in x_up], dim=0)  # B,C,H,W
        # use spatial attention in current frame on previous feature
        x_prev_cat = self.mtf_attention(
            x_up, x_prev.reshape(x_up.shape[0], -1, x_up.shape[2], x_up.shape[3])
        )  # B,K*C,H,W
        # time embedding
        x_up_fuse = torch.cat((x_up, x_prev_cat), dim=1) + self.time_embedding(
            example["times"][:, :, None].to(x_up)
        ).reshape(x_up.shape[0], -1, 1, 1)
        # fuse mtf feature
        x_up_fuse = self.out(x_up_fuse)

        ### self-design mtf network
        # B, K, C, H, W = x_prev.shape
        # x_frames_cat = torch.cat((x_up.unsqueeze(1), x_prev), dim=1) + self.time_embedding(
        #     example["times"][:, :, None].to(x_up))[:,:,:,None,None]
        # x_frames_att = self.mtf_attention(x_frames_cat)
        # use spatial attention in current frame on previous feature

        order_list = []
        out_dict_list = []
        for idx, task in enumerate(self.tasks):
            # heatmap head
            hm = self.hm_heads[idx](x_up_fuse)

            if self.corner and self.corner_heads[0].training:
                corner_hm = self.corner_heads[idx](x_up)
                corner_hm = torch.sigmoid(corner_hm)

            # find top K center location
            hm = torch.sigmoid(hm)
            batch, num_cls, H, W = hm.size()

            scores, labels = torch.max(hm.reshape(batch, num_cls, H * W), dim=1)  # b,H*W

            if self.use_gt_training and self.hm_heads[0].training:
                gt_inds = example["ind"][idx][:, (self.window_size // 2) :: self.window_size]
                gt_masks = example["mask"][idx][
                    :, (self.window_size // 2) :: self.window_size
                ]
                batch_id_gt = torch.from_numpy(np.indices((batch, gt_inds.shape[1]))[0]).to(
                    labels
                )
                scores[batch_id_gt, gt_inds] = scores[batch_id_gt, gt_inds] + gt_masks
                order = scores.sort(1, descending=True)[1]
                order = order[:, : self.obj_num]
                scores[batch_id_gt, gt_inds] = scores[batch_id_gt, gt_inds] - gt_masks
            else:
                order = scores.sort(1, descending=True)[1]
                order = order[:, : self.obj_num]

            scores = torch.gather(scores, 1, order)
            labels = torch.gather(labels, 1, order)
            mask = scores > self.score_threshold
            order_list.append(order)

            out_dict = {}
            out_dict.update(
                {
                    "hm": hm,
                    "scores": scores,
                    "labels": labels,
                    "order": order,
                    "mask": mask,
                    "BEV_feat": x_up,
                    "H": H,
                    "W": W,
                }
            )
            if self.corner and self.corner_heads[0].training:
                out_dict.update({"corner_hm": corner_hm})
            out_dict_list.append(out_dict)

        ## featuremaps render
        sample_token = example['metadata'][0]['token']
        feature_dict = dict(
            x_in=[x_in[0], 8],
            x=[x[0], 8],
            x_down=[x_down[0], 16],
            # x_down_up=[x_down_up[0], 8],
            # x_mix=[x_mix[0], 8],
            x_up=[x_up[0], 4],
        )
        save_mode=False
        multiscale_featuremaps_render(feature_dict, sample_token, save=save_mode)
        multichannel_featuremaps_render(x_up[0], 4, range(10), sample_token, save=save_mode)

        ## add for heatmap visualize experiment render
        sample_token = example['metadata'][0]['token']
        heatmaps = torch.cat([out_task['hm'][0] for out_task in out_dict_list], dim=0)
        heatmaps_render(heatmaps, sample_token, save=save_mode)
        heatmaps_show = change_for_vis(heatmaps, mode='mean')

        self.batch_id = torch.from_numpy(np.indices((batch, self.obj_num * len(self.tasks)))[0]).to(
                labels
            )
        order_all = torch.cat(order_list,dim=1)

        ct_feat = (
            x_up.reshape(batch, -1, H * W)
            .transpose(2, 1)
            .contiguous()[self.batch_id, order_all]
        )  # B, 500, C

        # create position embedding for each center
        y_coor = order_all // W
        x_coor = order_all - y_coor * W
        y_coor, x_coor = y_coor.to(ct_feat), x_coor.to(ct_feat)
        y_coor, x_coor = y_coor / H, x_coor / W
        pos_features = torch.stack([x_coor, y_coor], dim=2)

        if self.parametric_embedding:
            ct_feat = self.query_embed.weight
            ct_feat = ct_feat.unsqueeze(0).expand(batch, -1, -1)

        # run transformer
        # src = torch.cat(
        #     (
        #         x_up.reshape(batch, -1, H * W).transpose(2, 1).contiguous(),
        #         x.reshape(batch, -1, (H * W) // 4).transpose(2, 1).contiguous(),
        #         x_down.reshape(batch, -1, (H * W) // 16)
        #         .transpose(2, 1)
        #         .contiguous(),
        #     ),
        #     dim=1,
        # )  # B ,sum(H*W), C
        src_list = [
            x_up.reshape(batch, -1, H * W).transpose(2, 1).contiguous(),
            x.reshape(batch, -1, (H * W) // 4).transpose(2, 1).contiguous(),
            x_down.reshape(batch, -1, (H * W) // 16)
            .transpose(2, 1)
            .contiguous(),
        ]
        for frame in range(x_prev.shape[1]):
            src_list.append(
                x_prev[:, frame]
                .reshape(batch, -1, (H * W))
                .transpose(2, 1)
                .contiguous()
            )
        src = torch.cat(src_list, dim=1)  # B ,sum(H*W), C
        spatial_list = [(H, W), (H // 2, W // 2), (H // 4, W // 4)]
        spatial_list += [(H, W) for frame in range(x_prev.shape[1])]
        spatial_shapes = torch.as_tensor(
            spatial_list,
            dtype=torch.long,
            device=ct_feat.device,
        )
        level_start_index = torch.cat(
            (
                spatial_shapes.new_zeros((1,)),
                spatial_shapes.prod(1).cumsum(0)[:-1],
            )
        )

        if len(self.tasks) > 1:
            task_ids = torch.repeat_interleave(torch.arange(len(self.tasks)).repeat(batch,1), self.obj_num, dim=1).to(pos_features) # B, 500
            pos_features = torch.cat([pos_features, task_ids[:, :, None]],dim=-1)

        transformer_out = self.transformer_layer(
            ct_feat,
            self.pos_embedding,
            src,
            spatial_shapes,
            level_start_index,
            center_pos=pos_features,
        )  # (B,N,C)

        ct_feat = (
            transformer_out["ct_feat"].transpose(2, 1).contiguous()
        )  # B, C, 500

        for idx, task in enumerate(self.tasks):
            out_dict_list[idx]["ct_feat"] = ct_feat[:, :, idx * self.obj_num : (idx+1) * self.obj_num]

        return out_dict_list


@NECKS.register_module
class RPN_transformer_deformable_multitask_mtf_refactor_singlestage(RPN_transformer_base_multitask_refactor):
    def __init__(
        self,
        ds_num_filters,
        multiscale_neck=None,
        transformer_config=None,
        hm_head_layer=2,
        corner_head_layer=2,
        corner=False,
        assign_label_window_size=1,
        tasks=[],
        use_gt_training=False,
        norm_cfg=None,
        name="RPN_transformer_deformable_multitask_mtf_refactor",
        logger=None,
        init_bias=-2.19,
        score_threshold=0.1,
        obj_num=500,
        frame=1,
        parametric_embedding=False,
        **kwargs
    ):
        super(RPN_transformer_deformable_multitask_mtf_refactor_singlestage, self).__init__(
            ds_num_filters,
            multiscale_neck,
            transformer_config,
            hm_head_layer,
            corner_head_layer,
            corner,
            assign_label_window_size,
            tasks,
            use_gt_training,
            norm_cfg,
            logger,
            init_bias,
            score_threshold,
            obj_num,
        )
        self.frame = frame
        self.out = Sequential(
            nn.Conv2d(
                self._num_filters[0] * frame,
                self._num_filters[0],
                3,
                padding=1,
                bias=False,
            ),
            build_norm_layer(self._norm_cfg, self._num_filters[0])[1],
            nn.ReLU(),
        )
        self.mtf_attention = SpatialAttention_mtf()
        self.time_embedding = nn.Linear(1, self._num_filters[0])
        # self.mtf_attention = SpatialAttention_mtf_custom2(ds_num_filters[0], frame)

        self.transformer_layer = Deform_Transformer(
            self._num_filters[-1] * 2,
            depth=transformer_config.depth,
            heads=transformer_config.heads,
            levels=0 + self.frame,#2 + self.frame,
            dim_head=transformer_config.dim_head,
            mlp_dim=transformer_config.MLP_dim,
            dropout=transformer_config.DP_rate,
            out_attention=transformer_config.out_att,
            n_points=transformer_config.get("n_points", 9),
            # custom_deformable=True,
        )
        self.pos_embedding_type = transformer_config.get(
            "pos_embedding_type", "linear"
        )
        if self.pos_embedding_type == "linear":
            if len(self.tasks)>1:
                self.pos_embedding = nn.Linear(3, self._num_filters[-1] * 2)
            else:
                self.pos_embedding = nn.Linear(2, self._num_filters[-1] * 2)
        else:
            raise NotImplementedError()
        self.parametric_embedding = parametric_embedding
        if self.parametric_embedding:
            self.query_embed = nn.Embedding(self.obj_num * len(self.tasks), self._num_filters[-1] * 2)
            nn.init.uniform_(self.query_embed.weight, -1.0, 1.0)

        # self.src_pos_embedding = PositionalEncoding2D(256, height=360, width=360)
        # self.src_pos_embedding = PositionEmbeddingSine(num_pos_feats=256/2)

        logger.info("Finish RPN_transformer_deformable Initialization")

    def forward(self, x, example=None):
        # x_in = x
        # if hasattr(self, "multiscale_neck") and self.multiscale_neck is not None:
        x, x_down, x_up = self.multiscale_neck(x)
        # take out the BEV feature on current frame
        # x = torch.split(x, self.frame)
        x_up = torch.split(x_up, self.frame)
        # x_down = torch.split(x_down, self.frame)
        x_prev = torch.stack([t[1:] for t in x_up], dim=0)  # B,K,C,H,W
        # x = torch.stack([t[0] for t in x], dim=0)
        # x_down = torch.stack([t[0] for t in x_down], dim=0)

        x_up = torch.stack([t[0] for t in x_up], dim=0)  # B,C,H,W

        # ### self-design mtf network
        # B, K, C, H, W = x_prev.shape

        ## custom2 mtf_attention
        # x_frames_cat = torch.cat((x_up.unsqueeze(1), x_prev), dim=1) + self.time_embedding(
        #     example["times"][:, :, None].to(x_up))[:,:,:,None,None]
        # x_frames_att = self.mtf_attention(x_frames_cat)

        # use spatial attention in current frame on previous feature
        x_prev_cat = self.mtf_attention(
            x_up, x_prev.reshape(x_up.shape[0], -1, x_up.shape[2], x_up.shape[3])
        )  # B,K*C,H,W
        # time embedding
        x_up_fuse = torch.cat((x_up, x_prev_cat), dim=1) + self.time_embedding(
            example["times"][:, :, None].to(x_up)
        ).reshape(x_up.shape[0], -1, 1, 1)
        # fuse mtf feature
        x_up_fuse = self.out(x_up_fuse)

        # # delete mtf_attention
        # x_up_fuse = x_up

        order_list = []
        out_dict_list = []
        for idx, task in enumerate(self.tasks):
            # heatmap head
            hm = self.hm_heads[idx](x_up_fuse)

            if self.corner and self.corner_heads[0].training:
                corner_hm = self.corner_heads[idx](x_up)
                corner_hm = torch.sigmoid(corner_hm)

            # find top K center location
            hm = torch.sigmoid(hm)
            batch, num_cls, H, W = hm.size()

            scores, labels = torch.max(hm.reshape(batch, num_cls, H * W), dim=1)  # b,H*W

            if self.use_gt_training and self.hm_heads[0].training:
                gt_inds = example["ind"][idx][:, (self.window_size // 2) :: self.window_size]
                gt_masks = example["mask"][idx][
                    :, (self.window_size // 2) :: self.window_size
                ]
                batch_id_gt = torch.from_numpy(np.indices((batch, gt_inds.shape[1]))[0]).to(
                    labels
                )
                scores[batch_id_gt, gt_inds] = scores[batch_id_gt, gt_inds] + gt_masks
                order = scores.sort(1, descending=True)[1]
                order = order[:, : self.obj_num]
                scores[batch_id_gt, gt_inds] = scores[batch_id_gt, gt_inds] - gt_masks
            else:
                order = scores.sort(1, descending=True)[1]
                order = order[:, : self.obj_num]

            scores = torch.gather(scores, 1, order)
            labels = torch.gather(labels, 1, order)
            mask = scores > self.score_threshold
            order_list.append(order)

            out_dict = {}
            out_dict.update(
                {
                    "hm": hm,
                    "scores": scores,
                    "labels": labels,
                    "order": order,
                    "mask": mask,
                    "BEV_feat": x_up,
                    "H": H,
                    "W": W,
                }
            )
            if self.corner and self.corner_heads[0].training:
                out_dict.update({"corner_hm": corner_hm})
            out_dict_list.append(out_dict)

        # ## featuremaps render
        # sample_token = example['metadata'][0]['token']
        # # feature_dict = dict(
        # #     x_in=[x_in[0], 8],
        # #     x=[x[0], 8],
        # #     x_down=[x_down[0], 16],
        # #     # x_down_up=[x_down_up[0], 8],
        # #     # x_mix=[x_mix[0], 8],
        # #     x_up=[x_up[0], 4],
        # #
        # # )
        # # feature_save = False
        # heatmap_save = False
        # # multiscale_featuremaps_render(feature_dict, sample_token, save=feature_save)
        # # multichannel_featuremaps_render(x_up[0], 4, range(10), sample_token, save=feature_save)
        #
        # ## add for heatmap visualize experiment
        # heatmaps = torch.cat([out_task['hm'][0] for out_task in out_dict_list], dim=0)
        # heatmaps_render(heatmaps, sample_token, save=heatmap_save, show=True)

        self.batch_id = torch.from_numpy(np.indices((batch, self.obj_num * len(self.tasks)))[0]).to(
                labels
            )
        order_all = torch.cat(order_list,dim=1)

        ct_feat = (
            # x_up.reshape(batch, -1, H * W)
            x_up_fuse.reshape(batch, -1, H * W)
            .transpose(2, 1)
            .contiguous()[self.batch_id, order_all]
        )  # B, 500, C

        # create position embedding for each center
        y_coor = order_all // W
        x_coor = order_all - y_coor * W
        y_coor, x_coor = y_coor.to(ct_feat), x_coor.to(ct_feat)
        y_coor, x_coor = y_coor / H, x_coor / W
        pos_features = torch.stack([x_coor, y_coor], dim=2)

        if self.parametric_embedding:
            ct_feat = self.query_embed.weight
            ct_feat = ct_feat.unsqueeze(0).expand(batch, -1, -1)

        # run transformer
        # src = torch.cat(
        #     (
        #         x_up.reshape(batch, -1, H * W).transpose(2, 1).contiguous(),
        #         x.reshape(batch, -1, (H * W) // 4).transpose(2, 1).contiguous(),
        #         x_down.reshape(batch, -1, (H * W) // 16)
        #         .transpose(2, 1)
        #         .contiguous(),
        #     ),
        #     dim=1,
        # )  # B ,sum(H*W), C

        ## add time embedding for src
        src_list = [
           (self.time_embedding(example["times"][:, 0:1, None].to(x_up))
            .reshape(x_up.shape[0],-1,1,1) + x_up)
           .reshape(batch, -1, H * W).transpose(2, 1).contiguous(),
        ]
        for frame in range(x_prev.shape[1]):
           src_list.append(
               (self.time_embedding(example["times"][:, frame:frame+1, None].to(x_up))
                .reshape(x_prev.shape[0],-1,1,1) + x_prev[:, frame])
               .reshape(batch, -1, (H * W))
               .transpose(2, 1)
               .contiguous()
           )

        ## add the pos encoding for src

        # src_list = [
        #    (self.src_pos_embedding(x_up)+x_up).reshape(batch, -1, H * W).transpose(2, 1).contiguous(),
        # ]
        # for frame in range(x_prev.shape[1]):
        #    src_list.append(
        #        (self.src_pos_embedding(x_prev[:, frame])+x_prev[:, frame])
        #        .reshape(batch, -1, (H * W))
        #        .transpose(2, 1)
        #        .contiguous()
        #    )

        # src_list = [
        #      self.src_pos_embedding(x_up).reshape(batch, -1, H * W).transpose(2, 1).contiguous(),
        # ]
        # for frame in range(x_prev.shape[1]):
        #     src_list.append(
        #         self.src_pos_embedding(x_prev[:, frame])
        #         .reshape(batch, -1, (H * W))
        #         .transpose(2, 1)
        #         .contiguous()
        #     )

        # src_list = [
        #     x_up.reshape(batch, -1, H * W).transpose(2, 1).contiguous(),
        # ]
        # for frame in range(x_prev.shape[1]):
        #     src_list.append(
        #         x_prev[:, frame]
        #         .reshape(batch, -1, (H * W))
        #         .transpose(2, 1)
        #         .contiguous()
        #     )
        src = torch.cat(src_list, dim=1)  # B ,sum(H*W), C

        spatial_list = [(H, W)]
        spatial_list += [(H, W) for frame in range(x_prev.shape[1])]
        spatial_shapes = torch.as_tensor(
            spatial_list,
            dtype=torch.long,
            device=ct_feat.device,
        )
        level_start_index = torch.cat(
            (
                spatial_shapes.new_zeros((1,)),
                spatial_shapes.prod(1).cumsum(0)[:-1],
            )
        )

        if len(self.tasks) > 1:
            task_ids = torch.repeat_interleave(torch.arange(len(self.tasks)).repeat(batch,1), self.obj_num, dim=1).to(pos_features) # B, 500
            pos_features = torch.cat([pos_features, task_ids[:, :, None]],dim=-1)

        transformer_out = self.transformer_layer(
            ct_feat,
            self.pos_embedding,
            src,
            spatial_shapes,
            level_start_index,
            center_pos=pos_features,
        )  # (B,N,C)

        ct_feat = (
            transformer_out["ct_feat"].transpose(2, 1).contiguous()
        )  # B, C, 500

        for idx, task in enumerate(self.tasks):
            out_dict_list[idx]["ct_feat"] = ct_feat[:, :, idx * self.obj_num : (idx+1) * self.obj_num]

        # ## vis for deformable attention
        # save = True
        # if save:
        #     ## get the absolute pos
        #     # pos_absolute = pos_features[:,:,None,None,None,None,:2] + transformer_out['out_attention'][...,:2]
        #     # pos_absolute_att = torch.cat((pos_absolute, transformer_out['out_attention'][...,2:]), dim=-1)
        #     import os
        #     pos_absolute = torch.cat((
        #         transformer_out['out_attention'][...,:1] * 108 - 54,
        #         transformer_out['out_attention'][...,1:2] * 108 -54,
        #         transformer_out['out_attention'][..., 2:3]
        #          ), dim=-1)
        #     ## filter pred hm mask
        #     pos_absolute_mask_list = []
        #     pos_absolute_all_list = []
        #     k, depth, head = 10, 0, 0
        #     for i in range(len(out_dict_list)):
        #         pos_absolute_mask = pos_absolute[:, i*200:(i+1)*200][:, out_dict_list[i]['mask'][0], ...]
        #         pos_absolute_mask_list.append(pos_absolute_mask[:,:k,...])
        #         pos_absolute_all_list.append(pos_absolute_mask)
        #     pos_absolute_filter = torch.cat(pos_absolute_mask_list, dim=1)
        #     pos_absolute_all = torch.cat(pos_absolute_all_list, dim=1)
        #     B, objn, depths, heads, frames, npoints, _ = pos_absolute_filter.shape
        #     pos_absolute_filter = pos_absolute_filter.view(B, objn, depths, heads, frames*npoints, -1)
        #     B, objn, depths, heads, frames, npoints, _ = pos_absolute_all.shape
        #     pos_absolute_all = pos_absolute_all.view(B, objn, depths, heads, frames*npoints, -1)
        #     k = 100
        #     # pos_save = pos_absolute_filter[0,:k,depth,head].reshape(-1, 3)
        #     pos_save = pos_absolute_filter[0,:k,depth,:].reshape(-1, 3)
        #     pos_save = torch.cat([pos_save[:, :2], torch.zeros(pos_save.shape[0], 1).to(pos_save), pos_save[:, -1:]], dim=-1) ## add z
        #     pos_all_save = pos_absolute_all[0,:,depth,:].reshape(-1, 3)
        #     pos_all_save = torch.cat([pos_all_save[:, :2], torch.zeros(pos_all_save.shape[0], 1).to(pos_all_save), pos_all_save[:, -1:]],
        #                          dim=-1)  ## add z
        #     save_root = '/opt/sdatmp/lq/project/centerformer/vis/meshlab'
        #     token = ''
        #     save_dir = os.path.join(save_root, token)
        #     save_path = os.path.join(save_dir, 'trans_att.txt')
        #     np.savetxt(save_path, pos_save.cpu().numpy())
        #     save_path = os.path.join(save_dir, 'trans_att_all.txt')
        #     np.savetxt(save_path, pos_all_save.cpu().numpy())
        #     print(f'Deformable Transformer attention save to {save_path}')

        return out_dict_list


class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, dropout=0.1, height=10, width=10):
        super(PositionalEncoding2D, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.height = height
        self.width = width

        pe = torch.zeros(height, width, d_model)
        y_pos = torch.arange(0, height, dtype=torch.float).unsqueeze(1)
        x_pos = torch.arange(0, width, dtype=torch.float).unsqueeze(1)
        div_term_y = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        div_term_x = torch.exp(torch.arange(1, d_model + 1, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, :, 0::2] = torch.sin(y_pos * div_term_y).unsqueeze(1)
        pe[:, :, 1::2] = torch.cos(x_pos * div_term_x).unsqueeze(0)
        pe = pe.permute(2,0,1).unsqueeze(0)
        # pe = pe.detach()
        self.register_buffer('pe', pe)


    def forward(self, x):
        x = x + self.pe[..., :x.size(-2), :x.size(-1)].to(x.device)
        return self.dropout(x)


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.shape[0], x.shape[-2], x.shape[-1]), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
