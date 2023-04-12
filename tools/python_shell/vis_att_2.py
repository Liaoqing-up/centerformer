import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

import math
import matplotlib.pyplot as plt
from matplotlib import colors
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
    # token = '61a7bd24f88a46c2963280d8b13ac675'
    token = 'b22fa0b3c34f47b6a360b60f35d5d567'
    load_root = '/opt/sdatmp/lq/project/centerformer/vis/meshlab'
    load_dir = os.path.join(load_root, token)
    # frame_n = 10
    # points_list = []
    # for i in range(frame_n):
    #     points_list.append(np.loadtxt(os.path.join(load_dir, f'frame_{i}.txt')))

    points_all = np.loadtxt(os.path.join(load_dir, 'frame_all.txt'))
    att_points = np.loadtxt(os.path.join(load_dir, 'trans_att_all.txt'))

    mask = points_all[:,2] >= -2
    points_all = points_all[mask]

    mask = att_points[:,-1] >= 0.03
    att_points = att_points[mask]

    # att_points = np.loadtxt(os.path.join(load_dir, 'trans_att_all.txt'))

    # gt_objs = np.loadtxt(os.path.join(load_dir, 'gt_bojs_o3d.txt'))
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

    changecolor = colors.Normalize(vmin=0.4, vmax=1)    # norm=changecolor,
    ax.scatter(points_all[:,0], points_all[:,1], s=200, c=-points_all[:,-1], marker='.', cmap='Blues')
    ax.scatter(att_points[:,0], att_points[:,1], s=att_points[:,-1]*2000, c='r', marker='.', alpha=0.65)

    # ax.scatter(np.array([13.5,14, 15.4, 13.5]), np.array([-15,-15.6, -18, -15.8]), s=[1000,900,1000,800], c='r', marker='.', alpha=0.65)
    # ax.scatter(np.array([36.8,36.6, 36.4, 38.5]), np.array([21.8,21, 22.5, 18.7]), s=[1000,800,1000,800], c='r', marker='.', alpha=0.65)
    # ax.scatter(np.array([-3, -2, -2.6, 0, -4, -3.2, -2.5, -2.1, -3.4]), np.array([12.5, 11.6, 12.8, 14.6, 11.5, 11, 12.1, 12.3, 11.4]), s=[1000,800,1000,800, 1000, 600, 800, 1500, 1300], c='r', marker='.', alpha=0.65)
    # xy = np.asarray([[-13,7,1000],[-12.5,7,1200],[-12.3,7.4,800],[-15.5,6.4,1500],
    #                  [-16,7,800],[-14.8,7.5,1500],[-14.5,6.5,1500], [-13.2,6.5,1000],
    #                  [-15.8,7.5,1200],[-14.1,6.5,800],[-15.5,7.7,1000]])
    xy = np.asarray([[7.9,-14.6,1500],[8,-14.5,800], [7.8,-15.15,600],[7.95,-14.82,600],[7.83,-14.99,400],[7.66,-14.96,400]])
    ax.scatter(xy[:,0], xy[:,1], s=xy[:,2], c='r', marker='.', alpha=0.65)


    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    for box in gt_objs:
        box.render(ax, box.corners(), view=np.eye(4), colors=('r', 'r', 'r'), linewidth=1)

    ax.set_xlim(7,8.5)
    ax.set_ylim(-15.3,-13.8)
    # ax.set_ylim(5,10)

    # plt.savefig('/home/lq/centerformerpng/attention_4.png', bbox_inches='tight', dpi=600)
    plt.show()