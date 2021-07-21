import sys
import time
import os

import geopandas as gpd
import numpy as np
import tensorflow as tf
import config
import copy


def get_vertices(folder, file_name, save=True, has_objects=True):
    """Return geometries in array form"""
    print('\n∎ Collecting vertices')
    print('\t▹ Reading file <{}.shp> from <{}>'.format(file_name, folder))
    folder_path = config.DATA_ROOT + folder

    # extract file given <file_name>
    mask = np.array([(file_name in word) and ('.shp' in word) and (not '.shp.' in word) for word in
                     os.walk(folder_path).__next__()[-1]])
    files = os.walk(folder_path).__next__()[-1]
    file = files[np.where(mask == True)[0][0]]

    # read file
    data = gpd.read_file(os.path.join(folder_path, file))

    print('\t▹ Organizing as array')
    # setup toolbar
    toolbar_width = 10
    sys.stdout.write("\t[%s]" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width + 1))  # return to start of line, after '['
    tot_it = 1 if data.shape[0] < toolbar_width else data.shape[0] // 10

    vertx = np.zeros([data.shape[0], 3])  # define array of XYZ point data
    labels, objects_ids = [], []

    cnt = 0
    for index, geom in data.iterrows():
        if cnt % tot_it == 0:
            time.sleep(0.1)
            sys.stdout.write("∙")
            sys.stdout.flush()
        cnt += 1

        vertx[index] = np.array([geom.geometry.x, geom.geometry.y, geom.geometry.z])

        if has_objects:
            # make sure that a vertex with obj_id has a label (>0, indicating an obj., not background)
            if (geom.objects < 0) and (geom.labels > 0):
                # obj_id not known, but label is known, so consider the vertex as background (0)
                labels.append(0)
                objects_ids.append(geom.objects)
            else:
                labels.append(geom.labels)
                objects_ids.append(geom.objects)

    labels = np.array(labels)
    objects_ids = np.array(objects_ids)

    sys.stdout.write("]\n")  # this ends the progress bar
    print('○ {} vertices collected.'.format(data.shape[0]))

    if has_objects:
        # reformat labels to make it starts from 1 (not 0) and
        # to replace empty values (-9999) by 0
        objects_ids[objects_ids > -1] = objects_ids[objects_ids > -1] + 1
        objects_ids[objects_ids < 0] = 0.

    if save is True:
        np.save(os.path.join(folder_path, 'vertices'), vertx)
        np.save(os.path.join(folder_path, 'object_labels'), labels)
        np.save(os.path.join(folder_path, 'object_ids'), objects_ids)

    return vertx, labels, objects_ids


def get_edges(folder, file_name, save=True):
    """Return geometries in array form"""
    print('\n∎ Collecting edges')
    print('\t▹ Reading file <{}.shp> from <{}>'.format(file_name, folder))
    folder_path = config.DATA_ROOT + folder

    # extract file given <file_name>
    mask = np.array([(file_name in word) and ('.shp' in word) and (not '.shp.' in word) for word in
                     os.walk(folder_path).__next__()[-1]])
    files = os.walk(folder_path).__next__()[-1]
    file = files[np.where(mask == True)[0][0]]

    # read file
    data = gpd.read_file(os.path.join(folder_path, file))

    print('\t▹ Organizing as array')
    # setup toolbar
    toolbar_width = 10
    sys.stdout.write("\t[%s]" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width + 1))  # return to start of line, after '['
    tot_it = 1 if data.shape[0] < toolbar_width else data.shape[0] // 10

    array = np.zeros([data.shape[0], 4])  # define array of XYZ point data

    cnt = 0
    for index, geom in data.iterrows():
        if cnt % tot_it == 0:
            time.sleep(0.1)
            sys.stdout.write("∙")
            sys.stdout.flush()
        cnt += 1

        tmp = geom.geometry.coords.xy
        tmp = np.array([tmp[0][0], tmp[1][0], tmp[0][1], tmp[1][1]])
        array[index] = tmp

    sys.stdout.write("]\n")  # this ends the progress bar
    print('○ {} edges collected.'.format(data.shape[0]))

    if save is True:
        path = os.path.join(folder_path, 'edges')
        np.save(path, array)
    return array


def reload_data(folder, data):
    root_path = config.DATA_ROOT + folder
    return [np.load(os.path.join(root_path, name + '.npy'), allow_pickle=True) for name in data]


def format_edges(vertices, edges, name, folder, save=True):
    """Return geometries in array form"""
    assert edges.shape[1] == 4

    # convert data as <int32> type
    vertices_f = vertices.astype(np.float32)
    edges_f = edges.astype(np.float32)

    print('\n∎ Collecting edge vertices')
    # setup toolbar
    toolbar_width = 10
    sys.stdout.write("\t[%s]" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width + 1))  # return to start of line, after '['
    tot_it = 1 if vertices_f.shape[0] < toolbar_width else vertices_f.shape[0] // 10

    # define new edges with corresponding vertices
    array = np.zeros(shape=(edges_f.shape[0], 2), dtype=np.int32)

    for vindex, vcoords in enumerate(vertices_f):
        # vertex id <vindex> must start from 0 to process sparse adjacency matrix
        if vindex % tot_it == 0:
            time.sleep(0.1)
            sys.stdout.write("∙")
            sys.stdout.flush()
        x, y, _ = vcoords

        # increment over edge columns
        for cindex, i in enumerate(range(0, 3, 2)):
            # check and replace edge tips whose coordinates match given vertex' coordinates
            xmatch = np.equal(x, edges_f[:, i]).astype(np.int8)
            ymatch = np.equal(y, edges_f[:, i + 1]).astype(np.int8)
            match = (xmatch * ymatch) * vindex
            array[:, cindex] = array[:, cindex] + match

    sys.stdout.write("]\n")  # this ends the progress bar
    print('○ {} formated edges.'.format(array.shape[0]))

    # make sure no edge contains same vertex twice
    if np.sum(np.equal(array[:, 0], array[:, 1]).astype(np.int8)) > 0:
        print("!!! Some edges have same vertex twice.")

    if save is True:
        path = os.path.join(config.DATA_ROOT + folder, name)
        np.save(path, array)
    return array


def normalize_geom(points):
    """Normalize points coordinates"""
    return np.array([(points[:, i] - points[:, i].min()) / (points[:, i].max() - points[:, i].min())
                     for i in range(points.shape[1])]).T


def transform_geom(points, coord_reference, name, folder, save=True):
    """Transform points coordinates given <coord_reference>"""
    # get point cloud reference origin
    coord_min = np.amin(coord_reference, axis=0).tolist()
    coord_offset = np.asarray(coord_min[:points.shape[1]])

    # re-define points coords. given their distance to origin, and normalize by voxel size
    coord_trans = (points - coord_offset)
    if save is True:
        path = os.path.join(config.DATA_ROOT + folder, name)
        np.save(path, coord_trans)

    text = ', '.join([str(i) for i in np.amax(coord_trans, axis=0)])
    print('\n○ Max dimensions of transformed vertices (xyz):', text)
    return coord_trans


def get_centers(labels, vertices, folder, file_name, save=True):
    """Return geometries in array form"""
    path = os.path.join(config.DATA_ROOT + folder, file_name + '.shp')
    data = gpd.read_file(path)
    array = np.zeros([data.shape[0], 2])

    for index, geom in data.iterrows():
        tmp = np.array([geom.geometry.coords.xy[0][0], geom.geometry.coords.xy[1][0]])
        array[index] = tmp
    print('○ {} centers collected.'.format(data.shape[0]))

    # add 1 more column
    padded = np.pad(array, ((0, 0), (0, 1)))
    for row, lbl in enumerate(np.unique(labels)[1:]):
        # get z mean
        idx = np.where(labels == lbl)[0]
        obj_points = vertices[idx]
        padded[row][-1] = obj_points.mean(axis=0)[-1]

    if save is True:
        path = os.path.join(config.DATA_ROOT + folder, 'centers')
        np.save(path, padded)
    return padded


def squared_dist(A, B):
    expanded_a = tf.expand_dims(A, 1)
    expanded_b = tf.expand_dims(B, 0)
    distances = tf.reduce_sum(tf.compat.v1.squared_difference(expanded_a, expanded_b), 2)
    return tf.math.sqrt(distances)


def squared_dist_np(A, B):
    expanded_a = np.expand_dims(A, 1)
    expanded_b = np.expand_dims(B, 0)
    distances = np.sum(np.square(expanded_a - expanded_b), axis=2)
    return np.sqrt(distances)


def convert_angle_to_class(angle, eps=1e-10):
    """Convert angles to discrete class and residuals."""
    angle_increment = config.MAX_ANGLE / float(config.ANGLE_BINS)
    angle_classes = (angle - eps) / angle_increment
    angle_classes = angle_classes.astype(np.int32)
    residual_angles = angle - (angle_classes * angle_increment)
    return angle_classes, residual_angles


def rotate_point_cloud(points, angles):
    """Rotate in-place around Z axis"""
    rotation_angle = np.random.choice(angles) * (np.pi / 180)
    sinval, cosval = np.sin(rotation_angle), np.cos(rotation_angle)
    rotation_matrix = np.array([[cosval, sinval, 0],
                                [-sinval, cosval, 0],
                                [0, 0, 1]])
    ctr = points.mean(axis=0)
    rotated_data = np.dot(points - ctr, rotation_matrix) + ctr
    return rotated_data


def flip_around_y(points):
    """Apply flip transform on xy directions given maximum coordinates"""
    new_points = copy.deepcopy(points)

    coord_max = np.ceil(points.max(axis=0))
    xmax, ymax = coord_max[0], coord_max[1]

    def flip(i, imax):
        new_points[:, i] = imax - new_points[:, i]
        return new_points

    def double_flip(i, imax):
        new_points[:, i[0]] = imax[0] - new_points[:, i[0]]
        new_points[:, i[1]] = imax[1] - new_points[:, i[1]]
        return new_points

    func = np.random.choice([flip, double_flip])
    try:
        rand_n = np.random.choice([0, 1, None])
        if rand_n is None:
            return new_points  # no flipping
        elif rand_n == 0:
            return func(0, xmax)
        else:
            return func(1, ymax)
    except:
        return func([0, 1], [xmax, ymax])


def flip_xy_axes(points):
    """Flip x and y axis"""

    def flip(data):
        new_points = copy.deepcopy(data)
        x, y = copy.deepcopy(new_points[:, 0]), copy.deepcopy(new_points[:, 1])
        new_points[:, 0] = y
        new_points[:, 1] = x
        return new_points

    def no_flip(data):
        return data

    func = np.random.choice([flip, no_flip])
    return func(points)


def correlation_matrix(matrix):
    avg = tf.reduce_mean(matrix, axis=1, keepdims=True)
    diff = matrix - avg
    std = tf.reduce_sum(tf.math.square(diff), axis=1, keepdims=True)
    cov = tf.math.multiply(tf.expand_dims(diff, 1), tf.expand_dims(diff, 0))
    cov = tf.math.reduce_sum(cov, axis=2, keepdims=True)
    std = tf.math.multiply(tf.expand_dims(std, 1), tf.expand_dims(std, 0))
    std = tf.math.sqrt(std)
    corr = cov[:, :, 0] / std[:, :, 0]
    return tf.matmul(corr, matrix)
