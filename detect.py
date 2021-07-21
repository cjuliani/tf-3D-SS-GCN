# coding: utf-8
from solver import Solver
from utils.dataset import *
from utils.visualization import plot_point_cloud
from utils.box import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpu_memory", default=0.15, help='Memory fraction of GPU used for training.')
parser.add_argument("--gpu_allow_growth", default=False)
parser.add_argument("--soft_placement", default=True)
parser.add_argument("--model", type=str, default='train_1')
parser.add_argument("--task", type=str, default='detect')
parser.add_argument("--folder", type=str)
parser.add_argument("--radius", default=10)
parser.add_argument("--NMS", default=False)
parser.add_argument("--overlap_ratio", default=0.1)
parser.add_argument("--iou_thresh", default=0.5)
parser.add_argument("--semantics_only", default=False)
ARGS, unknown = parser.parse_known_args()

tf.compat.v1.disable_eager_execution()


class Detector(object):
    def __init__(self):
        self.solver = Solver()

    def show_ground_truth(self, folder):
        """Display ground truth elements in scene"""
        batch_keys, batch = self.solver.generate_batch('to_detect', folder)

        # define object vertices vector
        shp = self.solver.sampler.vertices_aug.shape[0]
        vertices_sem = np.zeros(shape=(shp,), dtype='int32')

        # Display boxes
        obj_ids = reload_data(os.path.join('to_detect', folder), ['object_ids'])[0]
        # make <obj_ids> ranging from 0...n count (instead of using the GIS/data source numbering)
        # because objects from current folder have e.g. numbers 45, 48, 65 from GIS (when applying Extract To Point)
        # but we need ids in increasing order, with increment of 1, starting from 0 (0 being the background)
        gis_obj_ids = np.asarray(obj_ids).astype(np.int32)  # gis numbering
        obj_ids = copy.deepcopy(gis_obj_ids)
        sorted = np.sort(np.unique(gis_obj_ids))  # sort gis numbering
        for i, id in enumerate(sorted):
            mask = obj_ids == id
            obj_ids[mask] = i

        # get 3D boxes given 2D boxes determined via opencv (0.2187)
        centers_xy, box_params, box_vertices = get_boxes_2D(self.solver.sampler.vertices_aug, obj_ids, kernel=2,
                                                            display=False)
        centers_form, box_vertices, _, _, _ = get_boxes_3D(centers_xy, box_vertices,
                                                           obj_ids, self.solver.sampler.vertices_aug,
                                                           box_params)  # (0.0059)

        # Display semantics
        indicator = np.greater(obj_ids, 0).astype(np.int32)
        pos_keys = np.where(indicator > 0)[0]
        pos_keys = np.array([i for i in pos_keys if i in batch_keys])
        vertices_sem[pos_keys] = 1

        # build array of colors to display per cloud point
        print('\t▹ plotting')
        func = lambda x: '#000000' if x < 1 else '#d99748'
        colors = [func(i) for i in vertices_sem]
        plot_point_cloud(self.solver.sampler.vertices_form, boxes=box_vertices, semantics=colors)

    def non_maximum_suppression(self, boxes, scores, dist_mat, iou_thresh=0.5):
        """Calculate NMS"""
        # copy boxes
        tboxes = copy.deepcopy(boxes)
        tscores = copy.deepcopy(scores)

        boxes_idxs_to_return = []
        tmp_indices = list(np.arange(len(tscores)))  # fixed indices or 'fidx' (do not change in loop)
        while len(tmp_indices) != 0:
            # collect idx given minimum distance between points
            dist = np.sum(dist_mat[np.asarray(tmp_indices)], axis=1)
            idx = np.argmin(dist)

            idxm = tmp_indices[idx]  # get fidx given local index
            bm = np.expand_dims(tboxes[idxm], axis=0)  # get box given fidx
            boxes_idxs_to_return.append(idxm)  # keep current box

            # redefine list of indices without fidx of current box
            tmp_indices = [i for i in tmp_indices if i != idxm]

            idx_to_remove = []
            for j in tmp_indices:
                # calculate iou between current box (bm) and boxes for remaining fidx
                bj = np.expand_dims(tboxes[j], axis=0)
                iou = get_mean_iou_3d(bm, bj)
                if iou > iou_thresh:
                    idx_to_remove.append(j)
            # redefine fixed indices
            tmp_indices = [i for i in tmp_indices if i not in idx_to_remove]

        # closest to others
        tmp = np.sum(dist_mat, axis=1)[np.array(boxes_idxs_to_return)]
        best = np.argmin(tmp)

        return np.array(boxes_idxs_to_return[best])  # return indices as array

    def detect(self, model, sess_config, folder, cluster_radius, NMS, semantics_only=False):
        with tf.compat.v1.Session(config=sess_config) as sess:
            # initialize variables
            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(tf.compat.v1.local_variables_initializer())

            print('\n∎ Detection with {}'.format(model))
            # restore model
            self.solver.saver.restore(sess, os.path.join('./checkpoint/', model))
            print('\t▹ {} restored.'.format(model))

            # (1) Get positive semantics in dataset by patching steps
            # get dataset keys, poinsets and coords.
            print('\t▹ processing {}'.format(folder))
            batch_maker = self.solver.generate_batch('to_detect', folder)

            # define object vertices vector
            shp = self.solver.sampler.vertices_form.shape[0]
            vertices_sem = np.zeros(shape=(shp,), dtype='int32')

            cnt = 0
            dataset_pos_keys, dataset_boxes, dataset_centers, dataset_scores, dataset_xyz = [], [], [], [], []
            for batch_keys, batch in batch_maker:
                batch_xyz = self.solver.sampler.vertices_aug[batch_keys]

                print('\t\tbatch {} done'.format(cnt))
                cnt += 1

                feed_dict = {self.solver.placeholders['point_set']: batch,
                             self.solver.placeholders['batch_xyz']: batch_xyz,
                             self.solver.placeholders['batch_keys']: np.expand_dims(batch_keys, -1),
                             self.solver.placeholders['training']: False}

                # get dataset keys for which semantics are positive
                pred_sem = sess.run(self.solver.pred_sem, feed_dict=feed_dict)
                if 1 in pred_sem:
                    # get indices of positive segmentation within batch
                    batch_pos_indices = np.where(pred_sem > 0)[0]
                    dataset_xyz += [batch_xyz[batch_pos_indices]]

                    pos_keys = np.reshape(batch_keys[batch_pos_indices], (-1,))
                    dataset_pos_keys += list(pos_keys)

                    if not semantics_only:
                        batch_pos_indices = np.reshape(batch_pos_indices, (-1, 1))
                        # predictions
                        start, end = 2, 2 + config.ANGLE_BINS
                        out = self.solver.proposals[..., start:end]
                        angle_cls = tf.argmax(tf.nn.softmax(out, axis=-1), axis=-1)
                        angle_cls = tf.gather_nd(angle_cls, batch_pos_indices)

                        start, end = 2 + config.ANGLE_BINS, 2 + (config.ANGLE_BINS * 2)
                        angle_res = self.solver.proposals[..., start:end]
                        angle_res = tf.gather_nd(angle_res, batch_pos_indices)

                        start, end = 2 + (config.ANGLE_BINS * 2), 2 + (config.ANGLE_BINS * 2) + 3
                        sizes = self.solver.proposals[..., start:end]
                        sizes = tf.gather_nd(sizes, batch_pos_indices)

                        pred_angle_res = get_element(angle_res, angle_cls)
                        obj_centers = tf.gather_nd(self.solver.pred_centers_xyz, batch_pos_indices)
                        corners = get_3D_box_corners(sizes, tf.cast(angle_cls, tf.float32), pred_angle_res, obj_centers)

                        # process
                        obj_boxes = sess.run(corners, feed_dict=feed_dict)
                        obj_scores = sess.run(tf.gather_nd(
                            tf.reduce_max(tf.nn.softmax(self.solver.segmentation, axis=1), axis=1), batch_pos_indices),
                            feed_dict=feed_dict)

                        dataset_boxes += [obj_boxes]
                        dataset_centers += [sess.run(obj_centers, feed_dict=feed_dict)]
                        dataset_scores += [obj_scores]

            # convert as array and rule out same keys
            unique_pos_keys, unique_pos_idxs = np.unique(np.array(dataset_pos_keys), return_index=True)

            if not list(unique_pos_keys):
                # no objects found
                print('No objects found in dataset.')
                return

            if semantics_only:
                # DISPLAY SEMANTICS
                vertices_sem[unique_pos_keys] = 1

                # build array of colors to display per cloud point
                print('\t▹ plotting')
                func = lambda x: '#000000' if x < 1 else '#d99748'
                colors = [func(i) for i in vertices_sem]
                plot_point_cloud(self.solver.sampler.vertices_form, semantics=colors)
            else:
                # FIND CLUSTERS
                dataset_boxes = np.reshape(np.array(dataset_boxes), (-1, 8, 3))[unique_pos_idxs]
                dataset_centers = np.reshape(np.array(dataset_centers), (-1, 3))[unique_pos_idxs]
                dataset_scores = np.reshape(np.array(dataset_scores), (-1,))[unique_pos_idxs]
                obj_surface = np.reshape(np.array(dataset_xyz), (-1, 3))[unique_pos_idxs]

                # calculate distance matrix between
                # object points and cluster closest points (within ball radius)
                # given that their center fall within the given radius.
                dist_matrix = squared_dist_np(dataset_centers, dataset_centers)
                mask = np.less_equal(dist_matrix, cluster_radius).astype(np.int32)

                obj_indices, to_skip = [], []
                for row in mask:
                    if 1 in row:
                        indices = np.where(row > 0)[0]
                        if any(i in to_skip for i in indices):
                            # it is possible that 2 close objects have their ball region intersecting
                            # hence we make sure that this intersection does not reach
                            ratios = []
                            for inds in obj_indices:
                                intersect = list(set(indices) & set(inds))
                                ratios += [len(intersect) / len(inds)]  # overlap ratio
                            if any(i > float(ARGS.overlap_ratio) for i in ratios):
                                # skip current row if related object overlaps too much
                                # with known objects
                                continue
                        to_skip += list(indices)
                        obj_indices.append(indices)  # all objects collected from a cluster
                print('\t▹ number of boxes:', len(obj_indices))

                # FIND BEST BOX(ES) FROM CLUSTER(S)
                best_boxes, best_centers, cluster_centers = [], [], []
                for cluster_idxs in obj_indices:
                    # collect boxes/scores of objects whose semantic is 1 in current cluster
                    tmp_boxes = dataset_boxes[cluster_idxs]
                    tmp_scores = dataset_scores[cluster_idxs]
                    tmp_centers = dataset_centers[cluster_idxs]

                    if NMS:
                        # get the best box for current cluster using NMS
                        dist_matrix = squared_dist_np(tmp_centers, tmp_centers)
                        nms_i = self.non_maximum_suppression(tmp_boxes, tmp_scores, dist_matrix,
                                                             iou_thresh=float(ARGS.iou_thresh))
                    else:
                        # get the box whose volume aggregate most of the vertices segmented as 1
                        counts = []
                        for bbox in tmp_boxes:
                            # take bounding box limits
                            bbox_min, bbox_max = bbox.min(axis=0), bbox.max(axis=0)
                            cnt = 0
                            for i, xyz in enumerate(obj_surface):
                                if (all(xyz <= bbox_max) is True) and (all(xyz >= bbox_min) is True):
                                    cnt += 1
                            counts.append(cnt)
                        nms_i = np.argmax(counts)

                    # collect box and center of current cluster
                    best_box = tmp_boxes[nms_i]
                    best_cent = tmp_centers[nms_i]

                    # assign color to vertices within box predicted
                    min, max = best_box.min(axis=0), best_box.max(axis=0)
                    for i, xyz in enumerate(self.solver.sampler.vertices_form):
                        if (all(xyz <= max) is True) and (all(xyz >= min) is True):
                            vertices_sem[i] = 1

                    # -------------------------
                    best_boxes += [best_box]
                    best_centers += [best_cent]
                    cluster_centers += [tmp_centers]

                # build array of colors to display per cloud point
                print('\t▹ plotting')
                func = lambda x: '#000000' if x < 1 else '#d99748'
                colors = [func(i) for i in vertices_sem]
                plot_point_cloud(self.solver.sampler.vertices_form,
                                 boxes=best_boxes,
                                 pred_centers=np.vstack(cluster_centers),
                                 semantics=colors)

if __name__ == '__main__':
    detector = Detector()
    # soft_placement:  parts of your network (which didn't fit in the GPU's memory) might be placed at the CPU
    tf_config = tf.compat.v1.ConfigProto(allow_soft_placement=bool(ARGS.soft_placement))
    tf_config.gpu_options.allow_growth = bool(ARGS.gpu_allow_growth)
    tf_config.gpu_options.per_process_gpu_memory_fraction = float(ARGS.gpu_memory)
    if ARGS.task == 'show_ground_truth':
        detector.show_ground_truth(folder=ARGS.folder)
    else:
        detector.detect(model=ARGS.model,
                        sess_config=tf_config,
                        folder=ARGS.folder,
                        cluster_radius=float(ARGS.radius),
                        NMS=bool(ARGS.NMS),
                        semantics_only=bool(ARGS.semantics_only))
