import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

import math
import matplotlib.pyplot as plt
## plot gt box in featuremap
from nuscenes import NuScenes
from nuscenes.eval.common.loaders import load_gt, add_center_dist, filter_eval_boxes
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.common.utils import boxes_to_sensor
from nuscenes.eval.common.config import config_factory

nusc_render = NuScenes(version='v1.0-mini', verbose=1, dataroot='data/nuScenes/')
gt_boxes = load_gt(nusc_render, 'mini_val', DetectionBox, verbose=1)
gt_boxes = add_center_dist(nusc_render, gt_boxes)
cfg_ = config_factory('detection_cvpr_2019')
gt_boxes = filter_eval_boxes(nusc_render, gt_boxes, cfg_.class_range, verbose=1)

def get_boxes_gt_from_sample_token(sample_token):
    boxes_gt_global = gt_boxes[sample_token]
    sample_rec = nusc_render.get('sample', sample_token)
    sd_record = nusc_render.get('sample_data', sample_rec['data']['LIDAR_TOP'])
    cs_record = nusc_render.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    pose_record = nusc_render.get('ego_pose', sd_record['ego_pose_token'])
    boxes_gt = boxes_to_sensor(boxes_gt_global, pose_record, cs_record)
    return boxes_gt

if __name__ == "__main__":
    token = '61a7bd24f88a46c2963280d8b13ac675'
    token = 'b22fa0b3c34f47b6a360b60f35d5d567'
    # token = 'fdc39b23ab4242eda6ec5e1e6574fe33'
    load_root = '/opt/sdatmp/lq/project/centerformer/vis/meshlab'
    load_dir = os.path.join(load_root, token)
    frame_n = 10
    # points_list = []
    # for i in range(frame_n):
    #     points_list.append(np.loadtxt(os.path.join(load_dir, f'frame_{i}.txt')))

    points_all = np.loadtxt(os.path.join(load_dir, 'frame_all.txt'))
    att_points = np.loadtxt(os.path.join(load_dir, 'trans_att_all.txt'))
    mask = att_points[:,-1] >= 0.3
    att_points = att_points[mask]
    # att_points = np.loadtxt(os.path.join(load_dir, 'trans_att_all.txt'))



    # for gt_obj in gt_objs:
    #     gt_boxs.append(Box(
    #         center=gt_obj[:3],
    #         size=gt_obj[3:6],
    #         orientation=Quaternion._from_axis_angle(np.array([0,0,1]), np.array(gt_obj[-1]*180/np.pi)))
    #     )

    ## generate box
    gt_objs = get_boxes_gt_from_sample_token(token)
    # 绘制点云的鸟瞰图
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # colors = ['#003499','#80abff']
    # for i in range(frame_n):
    #     points = points_list[i]
    #     ax.scatter(points[:,0], points[:,1], s=20, c=i, marker='.', cmap='Blues')

    mask = points_all[:,2] >= -1
    points_all = points_all[mask]

    ax.scatter(points_all[:,0], points_all[:,1], s=30, c=-points_all[:,-1], marker='.', cmap='Blues')
    # ax.scatter(att_points[:,0], att_points[:,1], s=att_points[:,-1]*10, c='r', marker='.', alpha=0.65)

    gt_objs_vel = np.loadtxt(os.path.join(load_dir, 'gt_bojs_vel_o3d.txt'))
    print(gt_objs_vel)
    ax.scatter(gt_objs_vel[0], gt_objs_vel[1], s=10, c='g', marker='o')


    # ax.scatter(np.array([13.5,14, 15.4, 13.5]), np.array([-15,-15.6, -18, -15.8]), s=[1000,900,1000,800], c='r', marker='.', alpha=0.65)
    # ax.scatter(np.array([36.8,36.6, 36.4, 38.5]), np.array([21.8,21, 22.5, 18.7]), s=[1000,800,1000,800], c='r', marker='.', alpha=0.65)

    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    for box in gt_objs:
        box.render(ax, box.corners(), view=np.eye(4), colors=('r', 'r', 'r'), linewidth=1)

    # ax.set_xlim(10,25)
    ax.set_xlim(2,6)
    ax.set_ylim(7,14)
    plt.show()