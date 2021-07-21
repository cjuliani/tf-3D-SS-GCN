import numpy as np
import matplotlib.pyplot as plt
import object_extraction as ext
import matplotlib.patches as patches
from utils.dataset import convert_angle_to_class
import config
from shapely.geometry import Polygon
from shapely import affinity
import tensorflow as tf


def get_boxes_2D(vertices, object_ids, kernel=1, display=True):
    """Get 2D bounding boxes parameters by projecting points over a map"""
    unique_ids = np.unique(object_ids)[1:]  # get unique object ids (except 0, which is background)

    # build map
    coord_max = np.ceil(vertices.max(axis=0))
    row, clmn = np.arange(0., coord_max[1], 1.), np.arange(0., coord_max[0], 1.)
    map = np.zeros(shape=(row.shape[0] + 1, clmn.shape[0] + 1))

    if display is True:
        fig, ax = plt.subplots()

    # look for contours per object
    contours, maps = [], []
    for i in unique_ids:
        idx = np.where(object_ids == i)[0]
        obj_xy = vertices[idx]
        for j, coords in enumerate(obj_xy):
            x, y, _ = np.ceil(coords).astype(np.int32)
            map[y - kernel:y + kernel, x - kernel:x + kernel] = 1.

        # get object contours
        con = ext.get_contours(map, (3, 3), 120)
        contours.append(con)
        maps.append(map)

        # reset map
        map = np.zeros(shape=(row.shape[0] + 1, clmn.shape[0] + 1))

    # get map with all objects by taking maximum over 1st axis
    if display is True:
        map = np.max(np.array(maps), axis=0)
        ax.imshow(map)

    # then plot the contours on this aggregated map
    centers_xy, box_params, box_vertices = [], [], []
    for i, con in enumerate(contours):
        if len(con[0]) > 1:
            sizes = [con[0][j].size for j in range(len(con))]
            idx = np.argmax(sizes)
        else:
            idx = 0
        box, box_coords = ext.get_bounding_box_properties(con[0][idx], config.ROTATED_BOXES)

        # find major axis given reference point <pref>
        p1, p2, p3, p4 = box

        w = np.linalg.norm(p1 - p2)
        l = np.linalg.norm(p1 - p4)
        angle = np.abs(box_coords[2])
        xc, yc = box_coords[0]

        centers_xy.append([xc, yc])
        box_params.append([l, w, angle])
        box_vertices.append(box)

        if display is True:
            # display polygons
            rect = patches.Polygon(box, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

            # display object number
            plt.text(xc, yc, str(i+1), fontsize=12, color='red')
            plt.plot(xc, yc, 'ro', markersize=3)

    if display is True:
        plt.gca().invert_yaxis()
        plt.show()

    return np.array(centers_xy), np.array(box_params), np.array(box_vertices)


def get_boxes_3D(centers_xy, box_vertices, object_ids, vertices, box_params):
    # calculate center Z coord. and height of boxes
    unique_ids = np.unique(object_ids)[1:]

    centers_xyz = np.pad(centers_xy, ((0, 0), (0, 1)))
    box_vertices_3D = np.zeros(shape=(box_vertices.shape[0], 8, 3))
    new_box_params = []
    for cnt, i in enumerate(unique_ids):
        idx = np.where(object_ids == i)[0]
        obj_xyz = vertices[idx]
        zmax, zmin = np.max(obj_xyz, axis=0)[-1], np.min(obj_xyz, axis=0)[-1]

        h = zmax - zmin
        centers_xyz[cnt][-1] = zmin + (h / 2)

        new_params = box_params[cnt].tolist()
        new_params.insert(2, h)  # add z coord to 2nd position, s.t. (w, l, h, angle)
        new_box_params.append(new_params)

        # add 3rd dimension of boxes given 2D coords and calculate height (h)
        box_vertices_3D[cnt][:4, :2] = box_vertices[cnt]
        box_vertices_3D[cnt][4:, :2] = box_vertices[cnt]

        box_vertices_3D[cnt][:4, -1] = zmax
        box_vertices_3D[cnt][4:, -1] = zmin

    # separate (l,w,h) sizes from angles (radian scalars)
    box_sizes, box_angles = np.array(new_box_params)[:, :-1], np.array(new_box_params)[:, -1]

    # get angle classes and residuals
    box_angles_cls, box_angles_res = convert_angle_to_class(box_angles)

    # make first row as 0s, which correspond to reference (unknown center)
    centers_xyz = np.pad(centers_xyz, ((1, 0), (0, 0)))
    box_vertices_3D = np.pad(box_vertices_3D, ((1, 0), (0, 0), (0, 0)))
    box_sizes = np.pad(box_sizes, ((1, 0), (0, 0)))
    box_angles_cls = np.pad(box_angles_cls, (1, 0))
    box_angles_res = np.pad(box_angles_res, (1, 0))

    return centers_xyz, box_vertices_3D, box_sizes, box_angles_cls, box_angles_res


def roty(angle):
    """Rotation about the y-axis give input angle (in degrees) concerted in radian"""
    radian_fact = np.pi / 180
    c = np.cos(angle * radian_fact)
    s = np.sin(angle * radian_fact)
    return np.array([[c, s, 0],
                     [-s, c, 0],
                     [0, 0, 1]])


def get_3D_box_corners_via_numpy(box_size, angle_cls, angle_res, center):
    """Get 3D box coordinates given the 2D map frames. """
    # retrieve angle in degrees from class and residual
    angle_increment = config.MAX_ANGLE / config.ANGLE_BINS
    # counter-clockwise angles between horiz. and [p1,p4]
    heading_angles = (angle_cls * angle_increment) + angle_res

    box_corners = np.zeros(shape=(box_size.shape[0], 8,3))
    for i, _ in enumerate(box_size):
        R = roty(heading_angles[i])
        l, w, h = box_size[i]
        x_corners = [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2]
        y_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
        z_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
        corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        corners_3d[0, :] = corners_3d[0, :] + center[i][0]  # x
        corners_3d[1, :] = corners_3d[1, :] + center[i][1]  # y
        corners_3d[2, :] = corners_3d[2, :] + center[i][2]  # z
        corners_3d = np.transpose(corners_3d)
        box_corners[i] = corners_3d
    return box_corners


def get_element(A, indices):
    """Outputs (ith element of indices) from (ith row of a)"""
    one_hot_mask = tf.one_hot(indices, A.shape[1], on_value=True, off_value=False, dtype=tf.bool)
    return tf.boolean_mask(A, one_hot_mask)


def get_3D_box_corners(box_size, angle_cls, angle_res, center):
    """Get 3D box coordinates given the 2D map frames. """
    pos_size = tf.shape(box_size)[0]
    angle_increment = config.MAX_ANGLE / config.ANGLE_BINS
    heading_angles = (angle_cls * angle_increment) + angle_res

    radian_fact = np.pi / 180
    c = tf.cos(heading_angles * radian_fact)
    s = tf.sin(heading_angles * radian_fact)
    zeros = tf.zeros_like(c)
    ones = tf.ones_like(c)

    rotation = tf.reshape(tf.stack([c, s, zeros,
                                    -s, c, zeros,
                                    zeros, zeros, ones], -1),
                          tf.stack([pos_size, 3, 3]))
    l, w, h = box_size[..., 0], box_size[..., 1], box_size[..., 2]

    corners = tf.reshape(tf.stack([-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2,
                                   w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2,
                                   h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2], -1),
                         shape=tf.stack([pos_size,3,8]))

    rotated = tf.einsum('ijk,ikm->ijm', rotation, corners) + tf.expand_dims(center, axis=2)
    return tf.transpose(rotated, perm=[0, 2, 1])


def get_mean_iou_3d(gt_bbox, pred_bbox, return_mean=True):
    '''Return iou of 3D box'''
    mean_iou, ious = 0., []
    for b1, b2 in zip(gt_bbox, pred_bbox):
        gt_poly_xz = Polygon(np.stack([b1[:4, 0], b1[:4, 1]], -1))
        pred_poly_xz = Polygon(np.stack([b2[:4, 0], b2[:4, 1]], -1))

        tmp = []
        for angle in [0, 90, 180]:
            rot_poly = affinity.rotate(pred_poly_xz, angle, 'center')
            inter_area = gt_poly_xz.intersection(rot_poly).area

            # calculate volume of intersecting plane from height (h)
            zmin = max(min(b1[:8, 2]), min(b2[:8, 2]))
            zmax = min(max(b1[:8, 2]), max(b2[:8, 2]))
            inter_h = max(0.0, zmax - zmin)
            inter_vol = inter_area * inter_h
            iou = inter_vol / (gt_poly_xz.area * (b1[0, 2] - b1[4, 2])
                               + pred_poly_xz.area * (b2[0, 2] - b2[4, 2]) - inter_vol)
            tmp.append(iou)
        # add best iou for current bbox analyzed
        mean_iou += max(tmp) / gt_bbox.shape[0]
        ious.append(max(tmp))

    if return_mean:
        return mean_iou
    else:
        return ious


def get_boxes_volumes(bbox):
    '''Calculate volume per box'''
    volumes = []
    for box in bbox:
        poly_xz = Polygon(np.stack([box[:4, 0], box[:4, 1]], -1))

        # calculate volume of intersecting plane from height (h)
        zmin, zmax = min(box[:8, 2]), max(box[:8, 2])
        h = max(0.0, zmax - zmin)
        vol = poly_xz.area * h
        volumes.append(vol)
    return volumes
