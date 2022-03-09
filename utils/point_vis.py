"""
Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.
"""
import open3d
import torch
import matplotlib
import numpy as np

box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]

def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num+1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba


def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True):
    if isinstance(points, np.ndarray):
        points = points
    if isinstance(gt_boxes, np.ndarray):
        gt_boxes = gt_boxes
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    # axis_pcd = open3d.create_mesh_coordinate_frame(size=0.1, origin=[0, 0, 0])
    vis = open3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    vis.add_geometry(axis_pcd)
    if point_colors is None:
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (0, 0, 1))

    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)

    vis.run()
    vis.destroy_window()

def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d

def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])

        vis.add_geometry(line_set)

        # if score is not None:
        #     corners = box3d.get_box_points()
        #     vis.add_3d_label(corners[5], '%.2f' % score[i])
    return vis

def rotation_points_single_angle(points, angle, axis=0):
    rot_sin = np.sin(angle)
    rot_cos = np.cos(angle)
    if axis == 1:
        rot_mat_T = np.array(
            [rot_cos, 0, -rot_sin, 0, 1, 0, rot_sin, 0, rot_cos],
            dtype=points.dtype).reshape(3, 3)
    elif axis == 2:
        rot_mat_T = np.array(
            [rot_cos, -rot_sin, 0, rot_sin, rot_cos, 0, 0, 0, 1],
            dtype=points.dtype
        ).reshape(3, 3)
    else:
        rot_mat_T = np.array(
            [1, 0, 0, 0, rot_cos, -rot_sin, 0, rot_sin, rot_cos],
            dtype=points.dtype
        ).reshape(3, 3)

    return points @ rot_mat_T

def points_select(points, npoint=16384, far_filter=40.0):
    if points.shape[0] < npoint:
        # 当前的样本点是少的，进行填充
        choice = np.arange(0, points.shape[0], dtype=np.int32)
        extra_choice = np.random.choice(choice, npoint-points.shape[0], replace=False)      # 是否进行重复采样
        choice = np.concatenate((choice, extra_choice), axis=0)
        np.random.shuffle(choice)
    elif points.shape[0] > npoint:
        # 当前的样本点数较多，需要进行减少
        # 根据距离进行筛选
        pts_depth = points[:, 2]
        pts_near_flag = pts_depth < far_filter
        # far_idxs_choice = np.where(pts_near_flat==0)[0]
        far_idxs = np.where(pts_near_flag==0)[0]
        near_idxs = np.where(pts_near_flag==1)[0]

        if len(near_idxs) < npoint:
            # 近处的点不够
            near_idxs_choice = near_idxs

            far_idxs_choice = np.random.choice(far_idxs, npoint-len(near_idxs_choice), replace=False)
            choice = np.concatenate((near_idxs, far_idxs_choice), aixs=0)
        else:
            choice = np.random.choice(near_idxs, npoint, replace=False)
        
        np.random.shuffle(choice)
    else:
        choice = np.arange(0, points.shape[0], dtype=np.int32)

    points = points[choice, :]
    return points

def pc_normalize(points):
    centroid = np.mean(points, axis=0)
    
    points = points - centroid
    m = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
    points = points / m
    import pdb; pdb.set_trace()
    return points

if __name__ == '__main__':
    #points = np.load('points.npy')[:, :3]
    points = np.load('point.npy')[:, :3]
#    points = rotation_points_single_angle(points, -0.5*np.pi, 2)
    print('before select ', points.shape[0])
    # select point cloud
#    points = points_select(points, 60266)

    # points normalization
#    points = pc_normalize(points)
    print('after select', points.shape[0])
    # points[:, -1] = points[:, -1] 
    # pred_box = np.load('pred_boxes.npy')
    
    OPEN3D_FLAG = True
    draw_scenes(
                points=points, ref_boxes=None
            )
