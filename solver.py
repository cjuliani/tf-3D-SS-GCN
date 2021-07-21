from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sampler import Sampler
from utils.dataset import *
from utils.box import *
from models import *
from utils.network import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpu_memory", default=0.15, help='Memory fraction of GPU used for training.')
parser.add_argument("--gpu_allow_growth", default=False)
parser.add_argument("--soft_placement", default=True)
parser.add_argument("--restore_model", default=False)
parser.add_argument("--save_model", default=False)
parser.add_argument("--model", type=str, default='train_1')
parser.add_argument("--task", type=str, default='train')
ARGS, unknown = parser.parse_known_args()

tf.compat.v1.disable_eager_execution()


class Solver(object):
    def __init__(self):
        # Get summary folders, create them if not existing
        self.summary_dir = os.path.join(config.SUMMARY_DIR, config.MODEL_DIR)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

        self.train_summary_dir = os.path.join(self.summary_dir, 'train')
        if not os.path.exists(self.train_summary_dir):
            os.makedirs(self.train_summary_dir)

        self.val_summary_dir = os.path.join(self.summary_dir, 'validation')
        if not os.path.exists(self.val_summary_dir):
            os.makedirs(self.val_summary_dir)

        # Get checkpoint folder, create it if not existing
        if not os.path.exists(config.CHECKPOINT_DIR):
            os.makedirs(config.CHECKPOINT_DIR)

        # get data folder
        self.sampler = Sampler()
        _, _ = self.generate_batch(phase='training')  # to initialize sampler and build network

        # define placeholders, models and related metrics
        self.placeholders = {
            'point_set': tf.compat.v1.placeholder(tf.float32, [None, 1, None, 3], 'point_set'),
            'batch_xyz': tf.compat.v1.placeholder(tf.float32, [None, 3], name='batch_xyz'),
            'batch_keys': tf.compat.v1.placeholder(tf.int32, [None, 1], name='batch_keys'),
            'training': tf.compat.v1.placeholder(tf.bool, shape=[], name='training'),
            'learning_rate': tf.compat.v1.placeholder(tf.float32, name='learning_rate'),
            'object_ids': tf.compat.v1.placeholder(tf.int32, [None], 'object_ids'),
            'box_sizes': tf.compat.v1.placeholder(tf.float64, [None, 3], 'box_sizes'),
            'obj_ids_counts': tf.compat.v1.placeholder(tf.int64, [None], 'obj_ids_counts'),
            'centers': tf.compat.v1.placeholder(tf.float64, [None, 3], 'centers'),
            'box_angles_cls': tf.compat.v1.placeholder(tf.int32, [None], 'box_angles_cls'),
            'box_angles_res': tf.compat.v1.placeholder(tf.float64, [None], 'box_angles_res'),
            'box_vertices': tf.compat.v1.placeholder(tf.float64, [None, 8, 3], 'box_vertices')
        }
        self.build_network()
        self.saver = tf.compat.v1.train.Saver()

    def build_network(self):
        """Define network sequence"""
        features, self.pointset_xyz, self.xyz_factor = AverageAggregation(self.placeholders['point_set'])
        self.pointset_norm_xyz = features[:, :3]
        self.pointset_feat = features[:, 3:]

        self.pointset_norm_xyz = self.pointset_xyz / self.xyz_factor
        features = tf.concat([self.pointset_norm_xyz, self.pointset_feat], axis=-1)

        ptnet1 = pointnet([16, 16, 32, 32])
        ptnet2 = pointnet([64, 64, 128, 128])
        ptnet3 = pointnet([128, 128, 256, 256])

        points_feat = ptnet1(features=features, istraining=self.placeholders['training'])

        points_xyz1, points_feat1 = sample_group_by_radius(self.pointset_xyz, points_feat,
                                                           radius=10,
                                                           filters=32)

        points_feat1 = ptnet2(features=points_feat1, istraining=self.placeholders['training'])

        points_xyz2, points_feat2 = sample_group_by_radius(points_xyz1, points_feat1,
                                                           radius=15,
                                                           filters=128)

        points_feat2 = ptnet3(features=points_feat2, istraining=self.placeholders['training'])

        points_xyz3, points_feat3 = sample_group_by_radius(points_xyz2, points_feat2,
                                                           radius=20,
                                                           filters=256)

        points_feat5 = upsampling_from_group(ref_feat=points_feat2, feat=points_feat3, filters=256,
                                             istraining=self.placeholders['training'])
        points_feat6 = upsampling_from_group(ref_feat=points_feat1, feat=tf.nn.relu(points_feat5),
                                             filters=128, istraining=self.placeholders['training'])
        self.seeds = upsampling_from_group(ref_feat=points_feat, feat=tf.nn.relu(points_feat6),
                                           filters=67, istraining=self.placeholders['training'])

        self.seeds_xyz = self.seeds[:, -3:]  # (16, 3)
        self.seeds_feat = self.seeds[:, :-3]

        seg_net = FullyConnectedSeq(config.SEG_UNITS, bias=config.BIAS, batchnorm=config.BATCH_NORM)
        self.segmentation = seg_net(tf.nn.relu(self.seeds), training=self.placeholders['training'])

        # offset calculation
        voting = VotingModule(config.VOTING_UNITS, bias=config.BIAS, batchnorm=config.BATCH_NORM)
        self.votes = voting(tf.nn.relu(tf.expand_dims(self.seeds, axis=0)),
                            training=self.placeholders['training'])
        self.votes_xyz = self.votes[:, -3:]

        # get center predictions
        self.pred_delta = self.votes_xyz - (self.pointset_norm_xyz * self.seeds_xyz)
        self.pred_centers_xyz = self.pointset_xyz + self.pred_delta

        # get ground truth
        indicator = tf.cast(tf.math.greater(self.placeholders['object_ids'], 0), tf.int32)
        self.gt_sem = tf.gather_nd(indicator, self.placeholders['batch_keys'])  # (batch, )
        gt_obj_ids = tf.gather_nd(self.placeholders['object_ids'], self.placeholders['batch_keys'])  # (batch, )
        self.gt_obj_ids = tf.reshape(gt_obj_ids, (-1, 1))

        # get proposals
        proposal_gen = FullyConnectedSeq(config.PROPOSAL_UNITS, bias=config.BIAS, batchnorm=config.BATCH_NORM)
        self.proposals = proposal_gen(tf.nn.relu(self.votes),
                                      training=self.placeholders['training'])

        # define losses and metrics
        self.define_losses()
        self.build_metrics()

    def build_metrics(self):
        """Concatenate metrics to be displayed while training"""
        # semantic accuracy
        epsilon = tf.constant(value=1e-10)
        self.pred_sem = tf.cast(tf.argmax(tf.nn.softmax(self.segmentation, axis=1), axis=1), tf.int32)
        TP = tf.compat.v1.count_nonzero(self.pred_sem * self.gt_sem, dtype=tf.float32, axis=0)
        FP = tf.compat.v1.count_nonzero(self.pred_sem * (self.gt_sem - 1), dtype=tf.float32, axis=0)
        FN = tf.compat.v1.count_nonzero((self.pred_sem - 1) * self.gt_sem, dtype=tf.float32, axis=0)
        # divide_no_NAN in case no TP exists in sample
        rec = tf.math.divide_no_nan(TP, (TP + FN))
        prec = tf.math.divide_no_nan(TP, (TP + FP))
        self.recall, self.precision = rec, prec
        self.sem_f1 = 2 * prec * rec / (prec + rec + epsilon)
        self.sem_acc = tf.reduce_mean(tf.cast(tf.equal(self.pred_sem, self.gt_sem), tf.float32))

        # intersection over union
        self.iou = tf.py_function(get_mean_iou_3d,
                                  inp=[tf.random.shuffle(self.positive_box_corners)[:config.IOU_NUMBER],
                                       tf.random.shuffle(self.pred_box_corners)[:config.IOU_NUMBER]],
                                  Tout=tf.float32)

        # center accuracy
        self.cent_acc = tf.reduce_mean(tf.norm(self.pred_centers - self.positive_centers, axis=1))

        # semantics ground truth
        self.sem_gt_ratio = tf.reduce_mean(tf.cast(self.gt_sem, tf.float32))

        # define summaries
        self.metrics = [self.loss, self.loss_sem, self.loss_center, self.loss_angle_cls, self.loss_angle_res,
                        self.loss_sizes, self.loss_corners, self.iou, self.sem_f1, self.sem_acc,
                        self.recall, self.precision, self.loss_mean_centroid]
        self.metric_names = ['loss', 'loss_sem', 'loss_center', 'loss_angle_cls', 'loss_angle_res',
                             'loss_sizes', 'loss_corners', 'iou', 'sem_f1', 'sem_acc', 'sem_rec',
                             'sem_prec', 'centroid']

        for tensor, name in zip(self.metrics, self.metric_names):
            tf.compat.v1.summary.scalar('summary/' + name, tensor)

        tf.compat.v1.summary.histogram('summary/' + 'segmentation', self.segmentation)
        tf.compat.v1.summary.histogram('summary/' + 'pred_centers', self.pred_centers)
        tf.compat.v1.summary.histogram('summary/' + 'delta_gt', self.delta_gt)
        tf.compat.v1.summary.histogram('summary/' + 'delta_pred', self.delta_pred)

    def weighted_binary_crossentropy(self, y_true, y_pred, w0, w1):
        """Return cross entropy loss weighted for segmentation task."""
        gt_sem_enc = tf.one_hot(self.gt_sem, depth=2)
        bce = tf.losses.binary_crossentropy(y_pred=tf.nn.softmax(y_pred, axis=1), y_true=gt_sem_enc)
        # Apply the weights
        weight_vector = y_true * w1 + (1. - y_true) * w0
        weighted_bce = weight_vector * bce

        return tf.reduce_mean(weighted_bce)

    def select_object_of_interest_from_batch(self, obj_n):
        """Select a random object to learn the box properties from."""
        objects_idx = tf.where(self.gt_sem > 0)
        objects = tf.gather_nd(self.gt_obj_ids, objects_idx)  # get object ids
        # get unique object ids and counts
        object_id, _, count = tf.unique_with_counts(tf.reshape(objects, (-1,)), out_idx=tf.dtypes.int32)

        # collect objects whose ratios are above a given threshold and learn from them (bbox)
        original_obj_counts = tf.gather(self.placeholders['obj_ids_counts'], object_id)
        ratios = tf.reshape(count, (-1,)) / tf.cast(tf.reshape(original_obj_counts, (-1,)), tf.int32)
        best_idx = tf.where(tf.math.greater_equal(ratios, config.OBJ_RATIO))  # (?,1)

        # if ratio not satisfied (best_idx is empty or 0), then take index of best ratio
        # otherwise, keep object ids whose ratio is satisfied
        obj_id = tf.cond(tf.equal(tf.size(best_idx), 0),
                         lambda: tf.reshape([object_id[tf.argmax(ratios)]], (-1,)),
                         lambda: tf.gather(tf.reshape(object_id, (-1,)), best_idx))

        batch_obj_ids = tf.reshape(self.gt_obj_ids, (-1,))
        mask_per_obj = tf.equal(batch_obj_ids, tf.reshape(tf.random.shuffle(obj_id)[:obj_n], (-1, 1)))
        mask = tf.reduce_sum(tf.cast(mask_per_obj, tf.int32), axis=0)

        return tf.where(tf.not_equal(mask, 0)), tf.where(tf.equal(mask, 0)), mask_per_obj

    def define_losses(self):
        self.loss_weights = tf.constant(config.LOSS_WEIGHTS)

        # Semantic loss
        self.loss_sem = self.weighted_binary_crossentropy(tf.cast(self.gt_sem, 'float32'),
                                                          self.segmentation,
                                                          config.SEG_WEIGHT_1s,
                                                          config.SEG_WEIGHT_0s) * self.loss_weights[0]

        # get indices of positive semantics (seeds on objects)
        positives_idx, negatives_idx, mask_per_obj = self.select_object_of_interest_from_batch(config.NUMB_OBJ_TO_LEARN)

        # Size loss
        gt_batch_sizes = tf.gather_nd(self.placeholders['box_sizes'], self.gt_obj_ids)
        positive_sizes = tf.gather_nd(gt_batch_sizes, positives_idx)

        start, end = 2 + (config.ANGLE_BINS * 2), 2 + (config.ANGLE_BINS * 2) + 3
        out_sizes = self.proposals[..., start:end]
        self.pred_sizes = tf.gather_nd(out_sizes, positives_idx)
        self.loss_sizes = tf.reduce_mean(tf.reduce_sum(
            tf.compat.v1.losses.huber_loss(labels=positive_sizes,
                                           predictions=self.pred_sizes,
                                           reduction=tf.losses.Reduction.NONE), axis=-1)) * self.loss_weights[4]

        # Center loss
        gt_batch_centers = tf.gather_nd(self.placeholders['centers'], self.gt_obj_ids)
        self.positive_centers = tf.cast(tf.gather_nd(gt_batch_centers, positives_idx), tf.float32)
        self.positive_xyz = tf.gather_nd(self.pointset_norm_xyz, positives_idx)

        self.delta_gt = self.positive_centers - tf.gather_nd(self.pointset_xyz, positives_idx)
        self.delta_pred = tf.gather_nd(self.pred_delta, positives_idx)

        self.pred_centers = tf.gather_nd(self.pred_centers_xyz, positives_idx)
        self.loss_center = tf.reduce_mean(tf.reduce_sum(
            tf.compat.v1.losses.huber_loss(labels=self.delta_gt,
                                           predictions=self.delta_pred,
                                           reduction=tf.losses.Reduction.NONE), axis=-1)) * self.loss_weights[1]

        # Centroid loss: force the voted centers to aggregate closer to each other
        # collect center xyz per object in scene
        tmp = tf.cast(tf.transpose(tf.gather_nd(tf.transpose(mask_per_obj), positives_idx)), tf.float32)
        centers_per_obj = tf.math.multiply(tf.expand_dims(tmp, -1), tf.expand_dims(self.pred_centers, 0))
        # calculate the mean centers xyz per object
        nonzero_mask = tf.math.reduce_any(tf.not_equal(centers_per_obj, 0.0), axis=-1, keepdims=True)
        n = tf.reduce_sum(tf.cast(nonzero_mask, 'float32'), axis=1, keepdims=True)
        centers_mean = tf.reduce_sum(centers_per_obj, axis=1, keepdims=True) / n
        # calculate distance between these mean xyz and the centers voted per object
        tiled_mean = tf.tile(centers_mean, multiples=[1, tf.shape(tmp)[-1], 1])
        nonzero_float = tf.cast(nonzero_mask, tf.float32)
        nonzero_mean = tf.math.multiply(nonzero_float, tiled_mean)
        distances = tf.math.sqrt(tf.reduce_sum(tf.math.square(centers_per_obj - nonzero_mean),
                                               axis=-1, keepdims=False))
        self.loss_mean_centroid = tf.math.reduce_sum(tf.math.reduce_mean(distances, axis=1), axis=0) * 0.1

        # Heading class loss
        gt_batch_angle_cls = tf.gather_nd(self.placeholders['box_angles_cls'], self.gt_obj_ids)
        positive_angle_cls = tf.gather_nd(gt_batch_angle_cls, positives_idx)
        positive_angle_cls_enc = tf.one_hot(positive_angle_cls, depth=config.ANGLE_BINS)

        start, end = 2, 2 + config.ANGLE_BINS
        out_angles_cls = self.proposals[..., start:end]
        pred_angle_cls_enc = tf.gather_nd(out_angles_cls, positives_idx)
        self.loss_angle_cls = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=pred_angle_cls_enc,
                                                    labels=positive_angle_cls_enc)) * self.loss_weights[2]

        # Heading residual loss
        one_hot_targets = tf.gather_nd(tf.eye(config.ANGLE_BINS),
                                       tf.expand_dims(self.placeholders['box_angles_cls'], 1))
        one_hot_res = tf.cast(tf.expand_dims(self.placeholders['box_angles_res'], 1), tf.float32) * one_hot_targets
        positive_angle_res = tf.gather_nd(one_hot_res, positives_idx)

        start, end = 2 + config.ANGLE_BINS, 2 + (config.ANGLE_BINS * 2)
        out_angles_res = self.proposals[..., start:end]
        self.pred_angle_res_enc = tf.gather_nd(out_angles_res, positives_idx)

        self.loss_angle_res = tf.reduce_mean(tf.reduce_sum(
            tf.compat.v1.losses.huber_loss(labels=positive_angle_res,
                                           predictions=self.pred_angle_res_enc,
                                           reduction=tf.losses.Reduction.NONE), axis=-1)) * self.loss_weights[3]

        # Box corners (regularization)
        gt_batch_corners = tf.gather_nd(self.placeholders['box_vertices'], self.gt_obj_ids)
        self.positive_box_corners = tf.cast(tf.gather_nd(gt_batch_corners, positives_idx), tf.float32)

        # get angle class number take angle residual per row given class number (indices)
        self.pred_angle_cls = tf.argmax(tf.nn.softmax(pred_angle_cls_enc, axis=-1), axis=-1)
        self.pred_angle_res = get_element(self.pred_angle_res_enc, self.pred_angle_cls)
        self.pred_box_corners = get_3D_box_corners(self.pred_sizes,
                                                   tf.cast(self.pred_angle_cls, tf.float32),
                                                   self.pred_angle_res,
                                                   self.pred_centers)

        # calculate eucl. dist, between corners <xyz>, then take the mean dist. per box (8 corners), then sum
        self.loss_corners = tf.stop_gradient(tf.reduce_sum(tf.reduce_mean(
            tf.norm(self.positive_box_corners - self.pred_box_corners, ord='euclidean', axis=2), axis=1))) \
                            * self.loss_weights[5]

        # Total loss
        self.loss = self.loss_sem + self.loss_center + self.loss_angle_cls + self.loss_angle_res + self.loss_sizes \
                    + self.loss_mean_centroid

    def get_progress(self, metrics):
        val = ['{:.4f}'.format(v) for v in metrics]
        nm = [[n, v] for n, v in zip(self.metric_names, val)]
        return [" --- {0}: {1}".format(*lst) for lst in nm]

    def generate_batch(self, phase, folder=None):
        """
            Randomize data set selection from usable folders (with processed data), then generate batch
            from a randomly selected folder.
            This allows to work on different 3D meshes (from different folders)
            during training when generating batches from those meshes. If <FOLDERS_TO_USE>
            is specified, only the specified folder will be used for training (or testing).
        """
        if (phase == 'training') and (config.FOLDERS_TO_USE is not None) and ((not config.FOLDERS_TO_USE) is False):
            assert all(fld in self.sampler.folders[phase] for fld in config.FOLDERS_TO_USE) is True, \
                "\n✖ All provided folders <FOLDERS_TO_USE> must be usable folders with pre-processed data."
            self.current_folder = folder if (folder is not None) else np.random.choice(config.FOLDERS_TO_USE)
            self.sampler(phase, self.current_folder)  # load processed data from randomly chosen data folder
        else:
            self.current_folder = folder if (folder is not None) else np.random.choice(self.sampler.folders[phase])
            self.sampler(phase, self.current_folder)

        augment = config.AUGMENT if phase == 'training' else False  # augment only when training
        if (phase in ['to_detect', 'detect']) is False:
            return self.sampler.generate_batch(augment=augment)
        else:
            return self.sampler.generate_batch_by_sliding_window()

    @tf.function
    def write_summary(self, step):
        for tensor, name in zip(self.metrics, self.metric_names):
            tf.summary.scalar(name, tensor, step=step)
        tf.summary.scalar('sem_accuracy', self.sem_acc, step=step)

    def train(self, sess_config):
        # train ops
        lr = tf.Variable(config.LR, trainable=False)
        optimizer = tf.compat.v1.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(self.loss)
        self.change_lr = tf.compat.v1.assign(lr, self.placeholders['learning_rate'])

        # Summary merger
        self.merged = tf.compat.v1.summary.merge_all()

        with tf.compat.v1.Session(config=sess_config) as sess:
            # write summary protocol buffers to event files.
            try:
                # delete previous event file
                existing_summary_files_train = os.walk(self.train_summary_dir).__next__()[-1]
                existing_summary_files_valid = os.walk(self.val_summary_dir).__next__()[-1]
                if existing_summary_files_train:
                    for file in existing_summary_files_train:
                        os.remove(os.path.join(self.train_summary_dir, file))
                if existing_summary_files_valid:
                    for file in existing_summary_files_valid:
                        os.remove(os.path.join(self.val_summary_dir, file))
            except PermissionError:
                pass

            self.train_writer = tf.compat.v1.summary.FileWriter(logdir=self.train_summary_dir, graph=sess.graph)
            self.validation_writer = tf.compat.v1.summary.FileWriter(logdir=self.val_summary_dir, graph=sess.graph)

            # initialize variables
            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(tf.compat.v1.local_variables_initializer())

            # Get checkpoint folder, create it if not existing
            if not os.path.exists(config.CHECKPOINT_DIR):
                os.mkdir(config.CHECKPOINT_DIR)

            # restore model
            if bool(ARGS.restore_model):
                # restore existing model
                self.saver.restore(sess, os.path.join(config.CHECKPOINT_DIR, config.MODEL_NAME))
            else:
                # or create new saving files otherwise
                _ = open(os.path.join(config.CHECKPOINT_DIR, 'training.txt'), 'w')
                _ = open(os.path.join(config.CHECKPOINT_DIR, 'validation.txt'), 'w')

            # get training folders
            tfld = self.sampler.training_folders

            print('\n∎ Training')
            train_metrics, global_step = 0., 0
            for epoch in range(config.MAX_ITER + 1):

                # change learning rate
                if (epoch % config.LR_CHANGE_EPOCH == 0) and (epoch != 0):
                    value = sess.run(lr) / config.LR_CHANGE_VAL
                    if value >= config.LR_MINIMUM:
                        sess.run(self.change_lr, feed_dict={self.placeholders['learning_rate']: value})

                # sequentially learn through training folders
                for step in range(len(tfld)):
                    # randomize data set selection from available folders, then, get batch from data of current folder
                    batch_keys, batch = self.generate_batch(phase='training', folder=tfld[step])
                    batch_xyz = self.sampler.vertices_aug[batch_keys]
                    batch_sem = np.greater(self.sampler.obj_ids[batch_keys], 0).astype(np.int32)

                    feed_dict = {self.placeholders['point_set']: batch,
                                 self.placeholders['batch_xyz']: batch_xyz,
                                 self.placeholders['batch_keys']: np.expand_dims(batch_keys, -1),
                                 self.placeholders['training']: config.IS_TRAINING,
                                 self.placeholders['object_ids']: self.sampler.obj_ids,
                                 self.placeholders['box_sizes']: self.sampler.box_sizes,
                                 self.placeholders['obj_ids_counts']: self.sampler.obj_ids_counts,
                                 self.placeholders['centers']: self.sampler.centers_form,
                                 self.placeholders['box_angles_cls']: self.sampler.box_angles_cls,
                                 self.placeholders['box_angles_res']: self.sampler.box_angles_res,
                                 self.placeholders['box_vertices']: self.sampler.box_vertices}

                    # Optimize
                    sess.run(train_op, feed_dict=feed_dict)

                    # Calculate metrics only for positives
                    if 1 in batch_sem:
                        # Get metrics
                        metrics = np.array([sess.run(m, feed_dict=feed_dict) for m in self.metrics])

                        # Calculate positive ratio to compare with semantics
                        sem_ratio = sess.run(self.sem_gt_ratio, feed_dict=feed_dict)

                        train_metrics += metrics / config.SUMMARY_ITER

                        if global_step % config.SUMMARY_ITER == 0:
                            # display results
                            text = self.get_progress(train_metrics)
                            text = 'Epoch {} '.format(epoch) + 'folder: {}'.format(self.current_folder) \
                                   + ''.join(text) + " --- {0}: {1:.5f}".format('sem_ratio', sem_ratio)
                            print(text)

                            # write results ie. append new lines to opened text file
                            with open(os.path.join(config.CHECKPOINT_DIR, 'training.txt'), 'a') as file:
                                file.write(text + "\n")

                            summary = sess.run(self.merged, feed_dict=feed_dict)
                            self.train_writer.add_summary(summary, global_step=global_step)

                            # reset metrics
                            train_metrics = 0.

                    if (global_step % config.VALIDATION_ITER == 0) and (global_step != 0):
                        # Validation
                        tmp_keys, tmp_batch = self.generate_batch(phase='validation')
                        tmp_xyz = self.sampler.vertices_aug[tmp_keys]

                        feed_dict = {self.placeholders['point_set']: tmp_batch,
                                     self.placeholders['batch_xyz']: tmp_xyz,
                                     self.placeholders['batch_keys']: np.expand_dims(tmp_keys, -1),
                                     self.placeholders['training']: False,
                                     self.placeholders['object_ids']: self.sampler.obj_ids,
                                     self.placeholders['box_sizes']: self.sampler.box_sizes,
                                     self.placeholders['obj_ids_counts']: self.sampler.obj_ids_counts,
                                     self.placeholders['centers']: self.sampler.centers_form,
                                     self.placeholders['box_angles_cls']: self.sampler.box_angles_cls,
                                     self.placeholders['box_angles_res']: self.sampler.box_angles_res,
                                     self.placeholders['box_vertices']: self.sampler.box_vertices}

                        # get metrics
                        metrics = np.array([sess.run(m, feed_dict=feed_dict) for m in self.metrics])

                        summary = sess.run(self.merged, feed_dict=feed_dict)
                        self.validation_writer.add_summary(summary, global_step=global_step)

                        text = self.get_progress(metrics)
                        text = 'Validation ' + 'folder: {}'.format(self.current_folder) + ''.join(text)
                        print(text + '\n')

                        with open(os.path.join(config.CHECKPOINT_DIR, 'validation.txt'), 'a') as file:
                            file.write(text + "\n")

                    global_step += 1

                # Save model
                if (bool(ARGS.save_model) is True) and (epoch % config.SAVE_ITER == 0):
                    self.saver.save(sess, os.path.join(config.CHECKPOINT_DIR, config.MODEL_NAME), global_step=epoch)

    def test(self, model, sess_config):
        print('\n∎ Testing with model {}'.format(model))

        with tf.compat.v1.Session(config=sess_config) as sess:
            # initialize variables
            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(tf.compat.v1.local_variables_initializer())

            # restore model
            self.saver.restore(sess, os.path.join('./checkpoint/', model))
            print('\n○ {} restored.\n'.format(model))

            n_folder = len(self.sampler.folders['test'])

            tot_metrics, tot_prec, tot_rec, obj_n = 0., 0., 0., 0.
            for folder in self.sampler.folders['test']:
                # get batch and predict
                batch_keys, batch = self.generate_batch(phase='test', folder=folder)
                batch_xyz = self.sampler.vertices_aug[batch_keys]

                feed_dict = {self.placeholders['point_set']: batch,
                             self.placeholders['batch_xyz']: batch_xyz,
                             self.placeholders['batch_keys']: np.expand_dims(batch_keys, -1),
                             self.placeholders['training']: False,
                             self.placeholders['object_ids']: self.sampler.obj_ids,
                             self.placeholders['box_sizes']: self.sampler.box_sizes,
                             self.placeholders['obj_ids_counts']: self.sampler.obj_ids_counts,
                             self.placeholders['centers']: self.sampler.centers_form,
                             self.placeholders['box_angles_cls']: self.sampler.box_angles_cls,
                             self.placeholders['box_angles_res']: self.sampler.box_angles_res,
                             self.placeholders['box_vertices']: self.sampler.box_vertices}

                metrics = np.array([sess.run(m, feed_dict=feed_dict) for m in self.metrics])

                # ---- increment metrics
                obj_n += len(self.sampler.obj_voxel_centers_idx)
                tot_metrics += metrics

                # Calculate metrics per folder, average by the number of ground truth object in that folder
                obj = len(self.sampler.obj_voxel_centers_idx)
                text = self.get_progress(metrics)
                text = 'Folder {}: {} objects'.format(self.current_folder, obj) + ''.join(text)
                print(text)

            print('\nTotal objects: ', obj_n)
            text = self.get_progress(tot_metrics / n_folder)
            text = 'Total average: ' + ''.join(text)
            print(text)


if __name__ == '__main__':
    solver = Solver()
    # soft_placement:  parts of your network (which didn't fit in the GPU's memory) might be placed at the CPU
    tf_config = tf.compat.v1.ConfigProto(allow_soft_placement=bool(ARGS.soft_placement))
    tf_config.gpu_options.allow_growth = bool(ARGS.gpu_allow_growth)
    tf_config.gpu_options.per_process_gpu_memory_fraction = float(ARGS.gpu_memory)
    if ARGS.task == 'test':
        solver.test(model=ARGS.model, sess_config=tf_config)
    else:
        solver.train(sess_config=tf_config)
