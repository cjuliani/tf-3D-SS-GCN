from utils.dataset import *
from utils.box import *
from utils.visualization import plot_point_cloud
from point_sampling import *
import config
import os


class Sampler(object):
    def __init__(self):
        print('\n∎ Collecting data')
        folders = os.walk(config.DATA_ROOT).__next__()[1]  # training, validation, testing

        self.folders = {}
        for folder in folders:
            datasets = os.walk(config.DATA_ROOT + folder).__next__()[1]

            print('\t- folder: {}'.format(folder))
            self.folders[folder] = []
            for dir in datasets:
                path = os.path.join(config.DATA_ROOT, folder)
                path = os.path.join(path, dir)
                # look for node file with .shp format
                mask = np.array([('nodes' in word) and ('.shp' in word) and (not '.shp.' in word) for word in
                                 os.walk(path).__next__()[-1]])
                files = os.walk(path).__next__()[-1]
                file = files[np.where(mask == True)[0][0]]
                # then, get related path
                path = os.path.join(path, file)
                if os.path.exists(path) is False:
                    print('\t\t✖ No spatial data in', dir)
                    continue
                path = os.path.join(config.DATA_ROOT, folder)
                path = os.path.join(path, dir)
                path = os.path.join(path, 'vertices_norm.npy')
                if os.path.exists(path) is False:
                    print('\t\t✖ Spatial data in {} is not processed.'.format(dir))
                    continue
                self.folders[folder].append(dir)
            print('\t\t▹ usable folders: ', ', '.join(datasets))

            if folder == 'training':
                if (not config.FOLDERS_TO_USE) is False:
                    print('\t\t▹ used folders: ', ', '.join(config.FOLDERS_TO_USE))
                    self.training_folders = config.FOLDERS_TO_USE
                else:
                    self.training_folders = self.folders['training']
        print('\n')

    def __call__(self, phase, folder):
        self.current_phase, self.current_folder = phase, folder

        # make sure data has been pre-processed
        root = os.path.join(config.DATA_ROOT + phase, folder)
        path = os.path.join(root, 'vertices_norm.npy')
        assert os.path.exists(path) is True, \
            '✖ File {} does not exist in {}. You must pre-process data from {}.'.format(path, folder, folder)
        # import pre-processed data
        self.vertices, self.vertices_form = reload_data(os.path.join(phase, folder), ['vertices', 'vertices_norm'])

        if phase not in ['detect', 'to_detect']:
            obj_ids = reload_data(os.path.join(phase, folder), ['object_ids'])[0]
            # make <obj_ids> ranging from 0...n count (instead of using the GIS/data source numbering)
            # because objects from current folder have e.g. numbers 45, 48, 65 from GIS (when applying Extract To Point)
            # but we need ids in increasing order, with increment of 1, starting from 0 (0 being the background)
            self.gis_obj_ids = np.asarray(obj_ids).astype(np.int32)  # gis numbering
            self.obj_ids = copy.deepcopy(self.gis_obj_ids)
            sorted = np.sort(np.unique(self.gis_obj_ids))
            for i, id in enumerate(sorted):
                mask = self.obj_ids == id
                self.obj_ids[mask] = i

    def generate_batch(self, augment):
        if augment:
            # apply data augmentation
            new_points = flip_xy_axes(self.vertices_form)
            self.vertices_aug = rotate_point_cloud(new_points, angles=[0, 90, -90, 180])
        else:
            self.vertices_aug = copy.deepcopy(self.vertices_form)

        # get 3D boxes given 2D boxes determined via opencv
        centers_xy, box_params, box_vertices = get_boxes_2D(self.vertices_aug, self.obj_ids, kernel=2, display=False)
        self.centers_form, self.box_vertices, self.box_sizes, \
        self.box_angles_cls, self.box_angles_res = get_boxes_3D(centers_xy, box_vertices,
                                                                self.obj_ids, self.vertices_aug, box_params)

        # get indices of voxels centers
        v_size = config.VOXEL_SIZE()
        self.voxel_idx = voxelize(points=self.vertices_aug, voxel_size=v_size)
        voxel_xy = self.vertices_aug[self.voxel_idx][:, :2]
        voxel_obj_ids = self.obj_ids[self.voxel_idx]

        # get unique obj IDs and related <counts>
        self.obj_ids_unique, self.obj_ids_counts = np.unique(voxel_obj_ids, return_counts=True)

        # Get indices of objects' centers to select voxel indices next such centers.
        obj_mask = np.greater(self.obj_ids, 0)
        points_on_obj_xy = self.vertices_aug[:, :2][obj_mask]
        obj_centers_idx = np.argmin(distance_matrix(centers_xy, points_on_obj_xy), axis=-1)
        obj_centers_xy = points_on_obj_xy[obj_centers_idx]
        self.obj_voxel_centers_idx = np.argmin(distance_matrix(obj_centers_xy, voxel_xy), axis=-1)

        # select a random object/negative index
        obj_idx = np.random.choice(self.obj_voxel_centers_idx, size=1)[0]
        neg_idx = np.random.choice(np.arange(voxel_xy.shape[0]), size=1)[0]

        if self.current_phase == 'training':
            pos_xy = voxel_xy[obj_idx]
            pos_keys, pos_sets = get_graph_sets_by_squared_zoning(voxel_keys=self.voxel_idx,
                                                                  vertices_xy=self.vertices_aug[:, :2],
                                                                  spl_radius=config.POINTSET_RADIUS(),
                                                                  win_radius=config.BATCH_WINDOW_RADIUS(),
                                                                  center_xy=pos_xy)
            neg_xy = voxel_xy[neg_idx]
            neg_keys, neg_sets = get_graph_sets_by_squared_zoning(voxel_keys=self.voxel_idx,
                                                                  vertices_xy=self.vertices_aug[:, :2],
                                                                  spl_radius=config.POINTSET_RADIUS(),
                                                                  win_radius=config.BATCH_WINDOW_RADIUS(),
                                                                  center_xy=neg_xy)
            self.voxel_sets = {**pos_sets, **neg_sets}
            batch_keys = np.hstack([pos_keys, neg_keys])
        else:
            # locate sampling window at an object center for validation
            origin_xy = voxel_xy[obj_idx]

            batch_keys, self.voxel_sets = get_graph_sets_by_squared_zoning(voxel_keys=self.voxel_idx,
                                                                           vertices_xy=self.vertices_aug[:, :2],
                                                                           spl_radius=config.POINTSET_RADIUS(),
                                                                           win_radius=config.BATCH_WINDOW_RADIUS(),
                                                                           center_xy=origin_xy)

        # format shape
        shp = batch_keys.shape[0]
        if (shp % 2) != 0:
            batch_keys = batch_keys[:shp - 1]

        # get point sets and make it 1 sparse matrix (batch_size, 1, max_npoints, 3) for feature learning (0.0010)
        batch = get_batch_pointset(self.vertices_aug, self.voxel_sets, batch_keys, batch_size=batch_keys.shape[0])

        return batch_keys, batch


    def visualize_cloud_points(self, boxes=None, semantics=None, **kwargs):
        # visualize point cloud, voxel centers and boxes
        if hasattr(self, 'vertices_aug') is True:
            cloud_points = self.vertices_aug
            self.voxel_points = self.vertices_aug[self.voxel_idx]
        else:
            cloud_points = self.vertices_form

        boxs = self.box_vertices[1:, :, :] if boxes is not None else None
        centers = self.centers_form[1:, :].astype(np.float32) if boxes is not None else None

        if semantics:
            vertices_sem = np.zeros(shape=(self.vertices_aug.shape[0],), dtype='int32')
            for box_i in range(len(boxs)):
                min, max = boxs[box_i].min(axis=0), boxs[box_i].max(axis=0)
                for i, xyz in enumerate(self.vertices_form):
                    if (all(xyz <= max) is True) and (all(xyz >= min) is True):
                        vertices_sem[i] = 1
            func = lambda x: '#000000' if x < 1 else '#d99748'
            semantics = [func(i) for i in vertices_sem]

        plot_point_cloud(cloud_points, boxes=boxs, box_centers=centers, semantics=semantics, **kwargs)


    def generate_batch_by_sliding_window(self):
        # get vertices
        self.vertices_aug = copy.deepcopy(self.vertices_form)

        # get indices of voxels centers (0.333)
        v_size = config.VOXEL_SIZE()
        voxel_idx = voxelize(points=self.vertices_aug, voxel_size=v_size)
        voxel_xy = self.vertices_aug[voxel_idx][:, :2]

        # get map limits
        vmin, vmax = voxel_xy.min(axis=0), voxel_xy.max(axis=0)
        diff = vmax - vmin
        dx, dy = diff[1], diff[0]   # row, column

        step_incr = config.DETECT_WINDOW_RADIUS
        xsteps, ysteps = [int((d // step_incr)+1) for d in [dx, dy]]

        for clm_i in range(xsteps):
            step_dx = step_incr * clm_i
            for row_i in range(ysteps):
                step_dy = step_incr * row_i
                tmp_origin = np.array([vmin[0] + step_dy, vmin[1] + step_dx])

                try:
                    batch_keys, self.voxel_sets = get_graph_sets_by_squared_zoning(voxel_keys=voxel_idx,
                                                                                   vertices_xy=self.vertices_aug[:, :2],
                                                                                   spl_radius=config.POINTSET_RADIUS(),
                                                                                   win_radius=config.DETECT_WINDOW_RADIUS,
                                                                                   center_xy=tmp_origin)
                except IndexError:
                    continue
                # make sure the batch size is even (not odd) because the up-sampling is
                # done via kernel of 2, hence final tensor shape will be even
                shp = batch_keys.shape[0]
                if (shp % 2) != 0:
                    batch_keys = batch_keys[:shp - 1]

                # get point sets for learning
                batch = get_batch_pointset(self.vertices_aug, self.voxel_sets, batch_keys,
                                           batch_size=batch_keys.shape[0])
                yield batch_keys, batch
