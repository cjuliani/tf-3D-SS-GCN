from utils.dataset import squared_dist
import tensorflow as tf
import config


def sample_group_by_radius(pointset_xyz, pointset_feat, radius, filters):
    """Return pointset per given voxel given a distance radius on xy coords."""
    assert radius > 0

    dist_matrix = squared_dist(pointset_xyz, pointset_xyz)
    mask = tf.cast((dist_matrix <= radius), tf.int32)

    # make sure the shape is even (not odd)
    shp = tf.math.floordiv(tf.shape(mask)[0], 2)
    shp = tf.cond(tf.not_equal(tf.math.floormod(shp,2),0), true_fn=lambda:shp+1, false_fn=lambda:shp)
    batch_indices = tf.random.shuffle(tf.range(tf.shape(mask)[0]))
    random_indices = tf.gather_nd(batch_indices, tf.expand_dims(tf.range(shp), axis=1))

    # cluster centers
    cluster_xyz = tf.gather(pointset_xyz, tf.expand_dims(random_indices, axis=1))
    cluster_feat = tf.gather(pointset_feat, tf.expand_dims(random_indices, axis=1))[:,0,:]
    cluster_mask = tf.gather(mask, tf.expand_dims(random_indices, axis=1))[:,0,:]

    # get masked features
    cluster_mask = tf.tile(tf.expand_dims(tf.cast(cluster_mask, 'float32'), axis=-1),
                           multiples=[1, 1, tf.shape(pointset_feat)[-1]])
    masked_hidden = tf.math.multiply(cluster_mask, tf.expand_dims(pointset_feat, axis=0))

    # combine points at different scales
    maxpool = tf.math.reduce_max(masked_hidden, axis=1, keepdims=True)
    hidden = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='valid')(maxpool)
    hidden = tf.keras.layers.ReLU()(hidden)
    hidden = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='valid')(hidden)

    return cluster_xyz[:,0,:], tf.concat([hidden[:,0,:], cluster_feat], axis=-1)


def upsampling_from_group(ref_feat, feat, filters, istraining, batchnorm=config.BATCH_NORM):
    ref = tf.expand_dims(ref_feat, axis=0)
    deconv = tf.keras.layers.UpSampling1D(size=2)(tf.expand_dims(feat, axis=0))

    shp1, shp2 = tf.shape(ref)[1], tf.shape(deconv)[1]
    deconv = tf.cond(tf.not_equal(shp1, shp2), lambda : deconv[:,:-2,:], lambda : deconv)

    concat = tf.concat([ref, deconv], axis=-1)
    output = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='valid')(concat)
    if batchnorm: output = tf.keras.layers.BatchNormalization()(output, training=istraining)
    output = tf.keras.layers.ReLU()(output)
    output = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='valid')(output)
    return output[0,:,:]
