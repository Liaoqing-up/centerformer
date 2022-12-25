import os.path as osp
import warnings
import numpy as np
from functools import reduce
import cv2

# import pycocotools.mask as maskUtils

from pathlib import Path
from copy import deepcopy
from det3d import torchie
from det3d.core import box_np_ops
import pickle 
import os 
from ..registry import PIPELINES

def _dict_select(dict_, inds):
    for k, v in dict_.items():
        if isinstance(v, dict):
            _dict_select(v, inds)
        else:
            dict_[k] = v[inds]


def handle_painted(path, num_point_feature):
    painted_dir_name = '_VIRTUAL_DBSCAN_3D_10_FILTEDGE'
    dir_path = '/' + os.path.join(*path.split('/')[:-2], path.split('/')[-2]+painted_dir_name)
    painted_path = os.path.join(dir_path, path.split('/')[-1]+'.pkl.npy')
    points_dict =  np.load(painted_path, allow_pickle=True).item()
    fore_points =  points_dict['real_points']   ## N 15
    raw_points =  np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :num_point_feature]

    back_mask = np.ones(raw_points.shape[0], dtype=bool)
    back_mask[points_dict['real_points_indice'].astype(np.int32)] = 0
    back_points = raw_points[back_mask] ## N 4
    back_points = np.concatenate([back_points, np.ones([back_points.shape[0], 15-num_point_feature])], axis=1)
    points = np.concatenate([fore_points, back_points], axis=0).astype(np.float32)
    return points


def read_file(path, tries=2, num_point_feature=4, painted=False):
    if painted:
        #dir_path = os.path.join(*path.split('/')[:-2], 'painted_'+path.split('/')[-2])
        #painted_path = os.path.join(dir_path, path.split('/')[-1]+'.npy')
        #points =  np.load(painted_path)
        #points = points[:, [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]] # remove ring_index from features 
        points = handle_painted(path, num_point_feature)
    else:
        points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :num_point_feature]

    return points


def remove_close(points, radius: float) -> None:
    """
    Removes point too close within a certain radius from origin.
    :param radius: Radius below which points are removed.
    """
    x_filt = np.abs(points[0, :]) < radius
    y_filt = np.abs(points[1, :]) < radius
    not_close = np.logical_not(np.logical_and(x_filt, y_filt))
    points = points[:, not_close]
    return points


def read_sweep(sweep, painted=False):
    min_distance = 1.0
    points_sweep = read_file(str(sweep["lidar_path"]), painted=painted).T
    points_sweep = remove_close(points_sweep, min_distance)

    nbr_points = points_sweep.shape[1]
    if sweep["transform_matrix"] is not None:
        points_sweep[:3, :] = sweep["transform_matrix"].dot(
            np.vstack((points_sweep[:3, :], np.ones(nbr_points)))
        )[:3, :]
    curr_times = sweep["time_lag"] * np.ones((1, points_sweep.shape[1]))

    return points_sweep.T, curr_times.T

def read_single_waymo(obj):
    points_xyz = obj["lidars"]["points_xyz"]
    points_feature = obj["lidars"]["points_feature"]

    # normalize intensity 
    points_feature[:, 0] = np.tanh(points_feature[:, 0])

    points = np.concatenate([points_xyz, points_feature], axis=-1)
    
    return points 

def read_single_waymo_sweep(sweep):
    obj = get_obj(sweep['path'])

    points_xyz = obj["lidars"]["points_xyz"]
    points_feature = obj["lidars"]["points_feature"]

    # normalize intensity 
    points_feature[:, 0] = np.tanh(points_feature[:, 0])
    points_sweep = np.concatenate([points_xyz, points_feature], axis=-1).T # 5 x N

    nbr_points = points_sweep.shape[1]

    if sweep["transform_matrix"] is not None:
        points_sweep[:3, :] = sweep["transform_matrix"].dot( 
            np.vstack((points_sweep[:3, :], np.ones(nbr_points)))
        )[:3, :]

    curr_times = sweep["time_lag"] * np.ones((1, points_sweep.shape[1]))
    
    return points_sweep.T, curr_times.T


def get_obj(path):
    with open(path, 'rb') as f:
            obj = pickle.load(f)
    return obj 

# def batch_view_points(points, view, normalize, device='cuda:0'):
#     # points: batch x 3 x N
#     # view: batch x 3 x 3
#     batch_size, _, nbr_points = points.shape
#
#     viewpad = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
#     viewpad[:, :view.shape[1], :view.shape[2]] = view
#
#     points = torch.cat((points, torch.ones([batch_size, 1, nbr_points], device=device)), dim=1)
#
#     # (6 x 4 x 4) x (6 x 4 x N)   -> 6 x 4 x N
#     points = torch.bmm(viewpad, points)
#     # points = torch.einsum('abc,def->abd', viewpad, points)
#
#     points = points[:, :3]
#
#     if normalize:
#         # 6 x 1 x N
#         points = points / points[:, 2:3].repeat(1, 3, 1)
#
#     return points


def view_points(points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
    """
    This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
    orthographic projections. It first applies the dot product between the points and the view. By convention,
    the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
    normalization along the third dimension.
    For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
    For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
    For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
     all zeros) and normalize=False
    :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
    :param view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
        The projection should be such that the corners are projected onto the first 2 axis.
    :param normalize: Whether to normalize the remaining coordinate (along the third axis).
    :return: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.
    """

    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    return points


def get_points_image_pos(points, info, H=900, W=1600):
    all_cams_from_lidar = info['all_cams_from_lidar']
    all_cams_intrinsic = info['all_cams_intrinsic']
    num_camera = len(all_cams_from_lidar)

    # ## mvp, projected to all 6 cams
    # num_lidar_point = points.shape[0]
    # projected_points = torch.zeros((num_camera, points.shape[0], 4), device=device)
    #
    # # (N x 3) ->  (3 x N) -> (4 x N) for compute extrinsic
    # point_padded = torch.cat([
    #             points.transpose(1, 0)[:3, :],
    #             torch.ones(1, num_lidar_point, dtype=points.dtype, device=device)
    #         ], dim=0)
    #
    # # (6 x 4 x 4) x (4 x N) -> (6 x 4 x N) -> (6 x 3 x N)
    # transform_points = torch.einsum('abc,cd->abd', all_cams_from_lidar, point_padded)[:, :3, :]
    # depths = transform_points[:, 2]  # (6 x 1 x N)
    #
    # points_2d = batch_view_points(transform_points[:, :3], all_cams_intrinsic, normalize=True)[:, :2].transpose(2, 1)
    # valid_mask = (points_2d[...,0] > 0) & (points_2d[...,0] < W) & (points_2d[...,1] > 0) & (points_2d[...,1] < H) & (depths > 0)
    # valid_projected_points[:, :2] = points_2d[valid_mask]
    # valid_projected_points[:, 2] = depths[valid_mask]
    # valid_projected_points[:, 3] = 1 # indicate that there is a valid projection
    #
    # projected_points[valid_mask] = valid_projected_points

    ### pointaugmenting, projected reserve one cam
    pts_uv_all = np.ones([points.shape[0], 3]).astype(np.float32) * -100
    for cam_id in range(num_camera):
        cam_from_lidar, cam_intrinsic = all_cams_from_lidar[cam_id], all_cams_intrinsic[cam_id]
        pts_paddes = np.concatenate([points[:, :3], np.ones([points.shape[0], 1])], axis=1).T
        # (4 x 4) x (4 x N) -> (4 x N) -> (3 x N)
        pts_cam = cam_from_lidar.dot(pts_paddes)[:3, :]
        # pts_cam = torch.einsum('bc,cd->bd', cam_from_lidar, pts_paddes)[:3, :]
        pts_uv = view_points(pts_cam, cam_intrinsic, normalize=True)[:2, :].T    # N * 2
        mask = (pts_cam[2, :] > 0) & (pts_uv[:, 0] > 0) & (pts_uv[:, 0] < W) & (pts_uv[:, 1] > 0) & (pts_uv[:, 1] < H)
        pts_uv_all[mask, :2] = pts_uv[mask, :2]
        pts_uv_all[mask, 2] = float(cam_id)

    return pts_uv_all  # N * 3  (u,v,)


def get_images_from_paths(all_cams_path):
    images = []
    for path in all_cams_path:
        original_image = cv2.imread(path)

        # if self.predictor.input_format == "RGB":
            # whether the model expects BGR inputs or RGB
        original_image = original_image[:, :, ::-1]
        images.append(original_image)

    images = np.stack(images, axis=0)
    return images


@PIPELINES.register_module
class LoadPointCloudFromFile(object):
    def __init__(self, dataset="KittiDataset", **kwargs):
        self.type = dataset
        self.random_select = kwargs.get("random_select", False)
        self.npoints = kwargs.get("npoints", 16834)
        self.combine_frames = kwargs.get("combine", 1)

    def __call__(self, res, info):

        res["type"] = self.type

        if self.type == "NuScenesDataset":

            nsweeps = res["lidar"]["nsweeps"]

            lidar_path = Path(info["lidar_path"])
            points = read_file(str(lidar_path), painted=res["painted"])
            res["lidar"]["points_num"] = points.shape[0]

            sweep_points_list = [points]
            sweep_times_list = [np.zeros((points.shape[0], 1))]

            assert (nsweeps - 1) == len(
                info["sweeps"]
            ), "nsweeps {} should equal to list length {}.".format(
                nsweeps, len(info["sweeps"])
            )

            for i in np.random.choice(len(info["sweeps"]), nsweeps - 1, replace=False):
                sweep = info["sweeps"][i]
                points_sweep, times_sweep = read_sweep(sweep, painted=res["painted"])
                sweep_points_list.append(points_sweep)
                sweep_times_list.append(times_sweep)

            points = np.concatenate(sweep_points_list, axis=0)
            times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

            if res["painted_features"]:
                points_uvc = get_points_image_pos(points[:,:3], info)
                # res['metadata']["num_point_features"] += 3
                res["lidar"]["points_uvc"] = points_uvc
                images = get_images_from_paths(info['all_cams_path'])
                res["cam"]["images"] = images

            res["lidar"]["points"] = points
            res["lidar"]["times"] = times
            res["lidar"]["combined"] = np.hstack([points, times])
        elif self.type == "NuScenesDataset_multi_frame":

            nsweeps = res["lidar"]["nsweeps"]

            lidar_path = Path(info["lidar_path"])
            points = read_file(str(lidar_path), painted=res["painted"])

            sweep_points_list = [points]
            sweep_times_list = [np.zeros((points.shape[0], 1))]
            combine = self.combine_frames
            c_frame = nsweeps // combine

            assert (nsweeps - 1) == len(
                info["sweeps"]
            ), "nsweeps {} should equal to list length {}.".format(
                nsweeps, len(info["sweeps"])
            )

            if c_frame > 0:
                sweep_points_list = []
                sweep_times_list = []
                sweep_combined_list = []

                combine_points_list = [points]
                combine_times_list = [np.zeros((points.shape[0], 1))]
                for j in range(combine-1):
                    sweep = info["sweeps"][j]
                    points_sweep, times_sweep = read_sweep(sweep, painted=res["painted"])
                    combine_points_list.append(points_sweep)
                    combine_times_list.append(times_sweep)

                sweep_points_list.append(np.concatenate(combine_points_list, axis=0))
                sweep_times_list.append(np.concatenate(combine_times_list, axis=0).astype(points.dtype))
                sweep_combined_list.append(np.hstack([sweep_points_list[-1], sweep_times_list[-1]]))

                for i in range(c_frame - 1):
                    combine_points_list = []
                    combine_times_list = []
                    for j in range(combine):
                        sweep = info["sweeps"][(i + 1) * combine + j - 1]
                        points_sweep, times_sweep = read_sweep(sweep, painted=res["painted"])
                        combine_points_list.append(points_sweep)
                        combine_times_list.append(times_sweep)

                    sweep_points_list.append(np.concatenate(combine_points_list, axis=0))
                    sweep_times_list.append(np.concatenate(combine_times_list, axis=0).astype(points.dtype))
                    sweep_combined_list.append(np.hstack([sweep_points_list[-1], sweep_times_list[-1]]))

            res["lidar"]["points"] = sweep_points_list
            res["lidar"]["times"] = sweep_times_list
            res["lidar"]["combined"] = sweep_combined_list


        elif self.type == "WaymoDataset":
            path = info['path']
            nsweeps = res["lidar"]["nsweeps"]
            obj = get_obj(path)
            points = read_single_waymo(obj)
            res["lidar"]["points"] = points
            res["lidar"]["points_num"] = points.shape[0]

            if nsweeps > 1: 
                sweep_points_list = [points]
                sweep_times_list = [np.zeros((points.shape[0], 1))]

                assert (nsweeps - 1) == len(
                    info["sweeps"]
                ), "nsweeps {} should be equal to the list length {}.".format(
                    nsweeps, len(info["sweeps"])
                )

                for i in range(nsweeps - 1):
                    sweep = info["sweeps"][i]
                    points_sweep, times_sweep = read_single_waymo_sweep(sweep)
                    sweep_points_list.append(points_sweep)
                    sweep_times_list.append(times_sweep)

                points = np.concatenate(sweep_points_list, axis=0)
                times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

                res["lidar"]["points"] = points
                res["lidar"]["times"] = times
                res["lidar"]["combined"] = np.hstack([points, times])
        elif self.type == "WaymoDataset_multi_frame":
            path = info['path']
            nsweeps = res["lidar"]["nsweeps"]
            obj = get_obj(path)
            points = read_single_waymo(obj)
            res["lidar"]["points"] = points
            res["lidar"]["points_num"] = points.shape[0]
            combine = self.combine_frames

            c_frame = nsweeps//combine
            if c_frame > 0: 
                sweep_points_list = []
                sweep_times_list = []
                sweep_combined_list = []

                combine_points_list = [points]
                combine_times_list = [np.zeros((points.shape[0], 1))]
                for j in range(combine-1):
                    sweep = info["sweeps"][j]
                    points_sweep, times_sweep = read_single_waymo_sweep(sweep)
                    combine_points_list.append(points_sweep)
                    combine_times_list.append(times_sweep)

                sweep_points_list.append(np.concatenate(combine_points_list, axis=0))
                sweep_times_list.append(np.concatenate(combine_times_list, axis=0).astype(points.dtype))
                sweep_combined_list.append(np.hstack([sweep_points_list[-1], sweep_times_list[-1]]))
                
                for i in range(c_frame - 1):
                    combine_points_list = []
                    combine_times_list = []
                    for j in range(combine):
                        sweep = info["sweeps"][(i+1)*combine+j-1]
                        points_sweep, times_sweep = read_single_waymo_sweep(sweep)
                        combine_points_list.append(points_sweep)
                        combine_times_list.append(times_sweep)

                    sweep_points_list.append(np.concatenate(combine_points_list, axis=0))
                    sweep_times_list.append(np.concatenate(combine_times_list, axis=0).astype(points.dtype))
                    sweep_combined_list.append(np.hstack([sweep_points_list[-1], sweep_times_list[-1]]))

                res["lidar"]["points"] = sweep_points_list
                res["lidar"]["times"] = sweep_times_list
                res["lidar"]["combined"] = sweep_combined_list

        else:
            raise NotImplementedError

        return res, info


@PIPELINES.register_module
class LoadPointCloudAnnotations(object):
    def __init__(self, with_bbox=True, **kwargs):
        pass

    def __call__(self, res, info):

        if res["type"] in ["NuScenesDataset", "NuScenesDataset_multi_frame"] and "gt_boxes" in info:
            gt_boxes = info["gt_boxes"].astype(np.float32)
            gt_boxes[np.isnan(gt_boxes)] = 0
            res["lidar"]["annotations"] = {
                "boxes": gt_boxes,
                "names": info["gt_names"],
                "tokens": info["gt_boxes_token"],
                "velocities": info["gt_boxes_velocity"].astype(np.float32),
            }
        elif res["type"] in ['WaymoDataset','WaymoDataset_multi_frame'] and "gt_boxes" in info:
            res["lidar"]["annotations"] = {
                "boxes": info["gt_boxes"].astype(np.float32),
                "names": info["gt_names"],
            }
        else:
            pass 

        return res, info
