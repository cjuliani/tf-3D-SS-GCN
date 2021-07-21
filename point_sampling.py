import numpy as np
import random
from scipy.spatial import distance_matrix


def get_batch_pointset(vertices, point_sets, batch_keys, batch_size):
    """Format point batch as input to the network in the format [B,H,W,C]"""
    point_set = [vertices[point_sets[key]] for key in batch_keys]
    non_zero_limits = [pset.shape[0] for pset in point_set]
    npoints_max = max(non_zero_limits)

    sparse_point_set = np.zeros(shape=(batch_size, 1, npoints_max, 3))
    for i, pset in enumerate(point_set):
        sparse_point_set[i, 0, :pset.shape[0], :] = pset
    return sparse_point_set


def get_patch_keys_from_centers(obj_centers, voxel_xy, voxel_ids, obj_n, random_selection, batch_size):
    """Batching of voxels keys from object centers."""
    outs = []
    for i in range(obj_n):
        # Randomize choice of a voxel key for training phase.
        obj_i = np.random.choice(obj_centers, size=1) if (random_selection is True) else np.array([obj_centers[i]])
        obj_center = voxel_xy[obj_i]
        # calculate distance between selected voxel-object center and other voxels
        dist_mat = distance_matrix(obj_center, voxel_xy)[0]
        closest = np.argsort(dist_mat)  # sort indices
        outs.append(voxel_ids[closest][:batch_size])
    return np.array([e for sub in outs for e in sub])


def get_batch_from_pattern(voxel_ids, voxel_xy, voxel_size, objects):
    """Get batch of voxel keys neighboring a randomly sampled positive key"""
    BPG = np.array([[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                    [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]])
    while True:
        try:
            # select positive key randomly
            voxel_xy_bt = (voxel_xy // voxel_size).astype(np.int32)
            # build map at <VOXEL_SIZE> resolution
            coord_max = np.ceil(voxel_xy_bt.max(axis=0))
            row, clmn = np.arange(0., coord_max[1], 1.), np.arange(0., coord_max[0], 1.)
            map = np.zeros(shape=(row.shape[0] + 1, clmn.shape[0] + 1))
            map[voxel_xy_bt[:, 1], voxel_xy_bt[:, 0]] = voxel_ids

            # get random voxel id assigned to an object
            pos_voxel_idxs = np.where(objects[voxel_ids] > 0)[0]
            pos_voxel_i = np.random.choice(voxel_ids[pos_voxel_idxs], size=1)
            coords = np.where(map==pos_voxel_i[0])
            origin_map_x, origin_map_y = coords[0][0], coords[1][0]

            # find voxel index corresponding to a center shifted by <x,y>
            find_neighbors = lambda dx, dy: map[origin_map_x+dx, origin_map_y+dy]

            # define offsets given pattern
            pidx_x, pidx_y = np.where(BPG == 1)
            origin = BPG.shape[0] // 2
            offsets_x, offsets_y = pidx_x - origin, pidx_y - origin

            # get voxel neighborhoods given offsets
            neighborhood = np.array([find_neighbors(x, y) for x, y in zip(offsets_x, offsets_y)]).astype(np.int32)
            break
        except IndexError:
            continue

    return neighborhood


def voxelize(points, voxel_size):
    """Voxelize point cloud over XY dimensions"""
    xyz = points // voxel_size
    xy = xyz[:, :2].astype(np.int32)

    voxels, voxel_keys = {}, []
    for pidx in range(len(xy)):
        key = ''.join([str(j) for j in xy[pidx]])
        if key in voxels:
            voxels[key].append(pidx)
        else:
            voxels[key] = [pidx]
            voxel_keys.append(key)

    # get representative xyz center of voxel
    voxel_centers = []
    for key in voxels:
        center_idx = random.choice(voxels[key])
        voxel_centers.append(center_idx)
    return np.array(voxel_centers)


def get_graph_sets_by_squared_zoning(voxel_keys, vertices_xy, spl_radius, win_radius, center_xy=None):
    """Return pointset per given voxel given a squared zone"""
    voxel_xy = vertices_xy[voxel_keys]

    # (1) Get new batch keys
    win_min_xy = center_xy - win_radius
    win_max_xy = center_xy + win_radius

    origin, max_xy = np.squeeze(win_min_xy), np.squeeze(win_max_xy)

    # assign color to vertices within box predicted
    new_batch_keys = []
    for i, xy in enumerate(voxel_xy):
        if (all(xy <= max_xy) is True) and (all(xy >= origin) is True):
            new_batch_keys += [voxel_keys[i]]
    new_batch_keys = np.array(new_batch_keys)

    # (2) Get pointsets by radius per batch key
    voxel_sets = {}
    # calculate distance between a voxel coords and those of other nodes
    voxel_xy = vertices_xy[new_batch_keys]
    distances = distance_matrix(voxel_xy,voxel_xy)
    # extract nodes withing a given distance from the voxel (neighborhood)
    mask = (distances < spl_radius).astype('int32')
    mask[mask==0] = -1
    indices = (mask * new_batch_keys) + mask

    for i, row in enumerate(indices):
        pointset = row[row >= 0] - 1
        voxel_sets[new_batch_keys[i]] = pointset

    return new_batch_keys, voxel_sets
