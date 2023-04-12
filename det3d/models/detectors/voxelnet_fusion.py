from ..registry import DETECTORS
from .single_stage import SingleStageDetector
from det3d.torchie.trainer import load_checkpoint
import torch
from copy import deepcopy
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F
import sys
sys.path.append('/opt/sdatmp/lq/project/segproject')
sys.path.append('/opt/sdatmp/lq/project/segproject/CenterNet2/projects/CenterNet2')


class Args_img(object):
    def __init__(self, config):
        self.opts = ["MODEL.WEIGHTS", config.MODEL.WEIGHT]
        self.config_file = config.config_file


def init_detector(img_config):
    from CenterNet2.projects.CenterNet2.train_net import setup
    from detectron2.engine import DefaultPredictor

    args_img = Args_img(img_config)
    cfg_img = setup(args_img)
    predictor = DefaultPredictor(cfg_img)
    return predictor

@DETECTORS.register_module
class VoxelNet_Fusion(SingleStageDetector):
    def __init__(
            self,
            reader,
            backbone,
            neck,
            bbox_head,
            train_cfg=None,
            test_cfg=None,
            pretrained=None,
            paint_img_features=False,
            image_config=None,
    ):
        super(VoxelNet_Fusion, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )
        # import argparse
        # parser = argparse.ArgumentParser(description="CenterPoint")
        # parser.add_argument('--config-file', type=str, default='c2_config/nuImages_CenterNet2_DLA_640_8x.yaml')
        # args = parser.parse_args()
        # args.opts.extend(["MODEL.WEIGHTS", "centernet2_checkpoint.pth"])

        self.img_predictor = init_detector(image_config)
        self.paint_img_features = paint_img_features

    def get_img_feat(self, pts_uvc, original_image, device, H=900, W=1600):
        with torch.no_grad():
            cams_num, batch_size = len(original_image),  original_image[0].shape[0]
            original_image = torch.cat(original_image, axis=0)
            original_image = original_image.cpu().numpy()
            images = []
            for i in range(original_image.shape[0]):
                image = self.img_predictor.aug.get_transform(original_image[i]).apply_image(original_image[i])
                image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1), device=device)

                images.append({'image':image, 'height':900, 'width':1600})
            # 12 64 128 232 -> (6 2) 64 128 232 -> 2 64 6 128 232
            img_feat = self.img_predictor.model(images)['dla2']
            img_feat = img_feat.view(
                cams_num, batch_size, img_feat.shape[1], img_feat.shape[-2], img_feat.shape[-1]).permute(1, 2, 0, 3, 4)

            feat_H, feat_W, feat_dim = img_feat.shape[-2], img_feat.shape[-1], img_feat.shape[1]
            points_img_feat_blist = []

            for i in range(batch_size):
                pts_img_feat = torch.ones((pts_uvc[i].shape[0], feat_dim)).to(device)
                ## max[224.9944, 126.4978, cam_id]
                pts_ft_uvc = torch.cat(
                    (pts_uvc[i][:, :2] * torch.as_tensor([900 / (W * 4), 506 / (H * 4)], device=pts_uvc[i].device), pts_uvc[i][:, -1:]),
                    axis=1).to(device)
                ## max[126.4978, 224.9944, cam_id], uv, align with featuremap
                pts_ft_uvc = pts_ft_uvc[:, [1, 0, 2]]

                # ##### for visualize
                # import cv2
                # import numpy as np
                # pick_camera = 3
                # pts_valid = pts_ft_uvc[:, -1] == pick_camera
                # pts_valid = pts_ft_uvc[pts_valid,]
                # pts_uv0 = pts_valid[:, :2]
                # pts_uv0 = (pts_uv0.cpu().numpy() * 6).astype(np.int64)
                # img_feat0 = img_feat[i, :, pick_camera, ...]
                # img_feat0_sum = img_feat0.sum(dim=0, keepdim=True)
                # img_feat0_rgb = img_feat0_sum / img_feat0_sum.max() * 255
                # img_feat0_rgb = img_feat0_rgb.cpu().numpy().astype(np.uint8)
                # img_feat0_rgb = np.tile(img_feat0_rgb, (3, 1, 1)).transpose(1, 2, 0)
                # img_feat0_rgb = cv2.resize(img_feat0_rgb, dsize=None, fx=6, fy=6)
                # ###  for visualize end


                ## normalize to [-1, 1]
                pts_ft_uvc = torch.cat((pts_ft_uvc[:, :2] / torch.as_tensor([feat_H-1, feat_W-1], device=device) * 2 -1, pts_ft_uvc[:, -1:]), axis=1)

                for cam_id in range(cams_num):
                    cam_mask = pts_ft_uvc[:, -1] == cam_id
                    pts_img_feat_cam = F.grid_sample(img_feat[i:i+1,:,cam_id,...], pts_ft_uvc[cam_mask][:, [1,0]][None,None,:,:], mode='bilinear', padding_mode='zeros')
                    pts_img_feat_cam = pts_img_feat_cam.squeeze().T
                    pts_img_feat[cam_mask] = pts_img_feat_cam

                    # ###  for visualize
                    # if cam_id == pick_camera and i == 1:
                    #     pts_f0 = (pts_img_feat_cam.sum(axis=1, keepdim=True) / pts_img_feat_cam.sum(axis=1).max() * 255).cpu().numpy().astype(np.int32)
                    #     pts_uvf0 = np.concatenate((pts_uv0, pts_f0), axis=1)
                    #     for coor in pts_uvf0:
                    #         cv2.circle(img_feat0_rgb, (int(coor[1]), int(coor[0])), radius=2, color=(0, 0, int(coor[2])))
                    #
                    #     image = original_image[2*pick_camera+1]
                    #     cv2.imshow('images', image)
                    #     cv2.imshow('features', img_feat0_rgb)
                    #     cv2.waitKey(-1)
                    #     print(f'show bathch_id{i} and camera_id{pick_camera} features and point clouds')
                    # ###  for visualize end

                points_img_feat_blist.append(pts_img_feat)
            return points_img_feat_blist


    def extract_feat(self, example):
        if self.paint_img_features:
            points_img_feat_list = self.get_img_feat(example['points_uvc'], example['images'], device=example['points'][0].device)
            for batch_id in range(len(points_img_feat_list)):
                points_img_fuse = torch.cat((example['points'][batch_id], points_img_feat_list[batch_id]), axis=1)
                example['points'][batch_id] = points_img_fuse

        if 'voxels' not in example:
            output = self.reader(example['points'])
            voxels, coors, shape = output

            data = dict(
                features=voxels,
                coors=coors,
                batch_size=len(example['points']),
                input_shape=shape,
                voxels=voxels
            )

        x, voxel_feature = self.backbone(
            data['voxels'], data["coors"], data["batch_size"], data["input_shape"]
        )

        if torch.isnan(x).any():
            print('backbone output is nan')
            print(x)

        if self.with_neck:
            x = self.neck(x, example)
        # print("&&&", len(x), type(x), x[0].keys())
        for task_id, ele in enumerate(x):
            for key, value in ele.items():
                if torch.is_tensor(value) and torch.isnan(value).any():
                    print(f'neck output taskid {task_id}, key {key} is nan')

        return x, voxel_feature

    def forward(self, example, return_loss=True, **kwargs):
        # if self.bbox_head.training:
        #     x, _ = self.extract_feat(example)
        # else:
        #     with autocast():
        #         x, _ = self.extract_feat(example)
        x, _ = self.extract_feat(example)
        preds = self.bbox_head(x)

        if return_loss:
            return self.bbox_head.loss(example, preds, self.test_cfg)
        else:
            return self.bbox_head.predict(example, preds, self.test_cfg)

    def forward_two_stage(self, example, return_loss=True, **kwargs):
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]

        batch_size = len(num_voxels)

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )

        x, _ = self.extract_feat(example, data)
        bev_feature = x['BEV_feat']
        preds = self.bbox_head(x)

        # manual deepcopy ...
        new_preds = []
        for pred in preds:
            new_pred = {}
            for k, v in pred.items():
                new_pred[k] = v.detach()

            new_preds.append(new_pred)

        boxes = self.bbox_head.predict(example, new_preds, self.test_cfg)

        if return_loss:
            return boxes, bev_feature, self.bbox_head.loss(example, preds, self.test_cfg)
        else:
            return boxes, bev_feature, None
