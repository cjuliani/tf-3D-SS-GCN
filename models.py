from layers import *


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        self.vars = {}
        self.layers = []
        self.sequence = []

        self.inputs = None
        self.outputs = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        with tf.compat.v1.variable_scope(self.name):
            self._build()

        # Sequential layer model
        self.sequence.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.sequence[-1])  # call last layer of sequence by <layer>
            self.sequence.append(hidden)
        self.outputs = self.sequence[-1]

        # Store model variables
        variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}


def gather_axis(params, indices, axis=0):
    return tf.stack(tf.unstack(tf.gather(tf.unstack(params, axis=axis), indices)), axis=axis)


def AverageAggregation(inputs):
    # average coordinates over non-zero entries
    nonzero = tf.math.reduce_any(tf.not_equal(inputs, 0.0), axis=-1, keepdims=True)
    n = tf.reduce_sum(tf.cast(nonzero, 'float32'), axis=2, keepdims=True)
    xyz = tf.reduce_sum(inputs, axis=2, keepdims=True) / n

    # normalization of xyz
    xyz_max = tf.reduce_max(xyz, axis=0, keepdims=True)
    xyz_min = tf.reduce_min(xyz, axis=0, keepdims=True)
    xyz_factor = (xyz_max - xyz_min)
    xyz_norm = xyz / xyz_factor

    # offsets from centroid
    xi = tf.expand_dims(xyz[0], axis=0)
    offset = xi - xyz
    dist = tf.norm(offset, ord='euclidean', keepdims=True, axis=-1)

    # H delta
    h = xyz[:, :, :, -1:]
    hmin = tf.reduce_min(h, axis=0, keepdims=True)
    hmax = tf.reduce_max(h, axis=0, keepdims=True)
    hnorm = h / (hmax - hmin)

    hidden = tf.concat([xyz_norm, hnorm, offset, dist], axis=-1)

    return hidden[:, 0, 0, :], xyz[:, 0, 0, :], xyz_factor[:, 0, 0, :]


class pointnet(Model):
    def __init__(self, units, **kwargs):
        super(pointnet, self).__init__(**kwargs)

        self.units = units

    def __call__(self, features, istraining, batchnorm=config.BATCH_NORM):
        hidden = tf.expand_dims(features, axis=0)
        conv1 = tf.keras.layers.Conv1D(filters=self.units[0], kernel_size=1, strides=1, padding='same')(hidden)
        if batchnorm: conv1 = tf.keras.layers.BatchNormalization()(conv1, training=istraining)
        conv1 = tf.keras.layers.ReLU()(conv1)

        conv1 = tf.keras.layers.Conv1D(filters=self.units[1], kernel_size=1, strides=1, padding='same')(conv1)
        if batchnorm: conv1 = tf.keras.layers.BatchNormalization()(conv1, training=istraining)
        conv1 = tf.keras.layers.ReLU()(conv1)

        conv1 = tf.keras.layers.Conv1D(filters=self.units[2], kernel_size=1, strides=1, padding='same')(conv1)
        if batchnorm: conv1 = tf.keras.layers.BatchNormalization()(conv1, training=istraining)
        conv1 = tf.keras.layers.ReLU()(conv1)

        conv1 = tf.keras.layers.Conv1D(filters=self.units[3], kernel_size=1, strides=1, padding='same')(conv1)
        if batchnorm: conv1 = tf.keras.layers.BatchNormalization()(conv1, training=istraining)
        conv1 = tf.keras.layers.ReLU()(conv1)

        maxpool = tf.math.reduce_max(conv1, axis=1, keepdims=True)

        # ------------
        conv2 = tf.keras.layers.Conv1D(filters=self.units[2], kernel_size=1, strides=1, padding='same')(maxpool)
        if batchnorm: conv2 = tf.keras.layers.BatchNormalization()(conv2, training=istraining)
        conv2 = tf.keras.layers.ReLU()(conv2)

        conv2 = tf.keras.layers.Conv1D(filters=self.units[1], kernel_size=1, strides=1, padding='same')(conv2)
        if batchnorm: conv2 = tf.keras.layers.BatchNormalization()(conv2, training=istraining)
        conv2 = tf.keras.layers.ReLU()(conv2)

        tiled = tf.tile(conv2, multiples=[1, tf.shape(conv1)[1], 1])
        merge = tf.concat([conv1, tiled], axis=-1)

        # -------------
        conv3 = tf.keras.layers.Conv1D(filters=self.units[2], kernel_size=1, strides=1, padding='same')(merge)
        if batchnorm: conv3 = tf.keras.layers.BatchNormalization()(conv3, training=istraining)
        conv3 = tf.keras.layers.ReLU()(conv3)
        conv3 = tf.keras.layers.Conv1D(filters=self.units[1], kernel_size=1, strides=1, padding='same')(conv3)

        return conv3[0, :, :]


class VotingModule(Model):
    def __init__(self, units, activation=tf.nn.relu, bias=True, batchnorm=False, **kwargs):
        super(VotingModule, self).__init__(**kwargs)

        self.layers = []
        self.activation = activation
        self.batchnorm = batchnorm

        # build ops sequentially
        for i, unit in enumerate(units):
            if i == len(units) - 1:
                tmp_act = lambda x: x  # does nothing, no activation
                tmp_bn = lambda x: x
            else:
                tmp_act = self.activation
                tmp_bn = self.batchnorm
            offset = SharedConv(unit, activation=tmp_act, bias=bias,
                                batchnorm=tmp_bn, name='voting{}'.format(i))
            self.layers.append(offset)

    def __call__(self, inputs, training):
        # cascading layer calls
        offset = inputs
        for layer in self.layers:
            offset = layer(offset, training)
        return offset[0, :, :]


class FullyConnectedSeq(Model):
    def __init__(self, units, bias=True, activation=tf.nn.relu, batchnorm=False, **kwargs):
        super(FullyConnectedSeq, self).__init__(**kwargs)

        self.activation = activation
        self.batchnorm = batchnorm
        self.bias = bias
        self.layers = []

        # build ops sequentially
        for i, unit in enumerate(units):
            if i == (len(units) - 1):
                tmp_act = lambda x: x  # does nothing, no activation
                tmp_bn = lambda x: x
            else:
                tmp_act = self.activation
                tmp_bn = self.batchnorm
            output = FullyConnected(unit, bias=self.bias, batchnorm=tmp_bn, activation=tmp_act,
                                    name='fc{}'.format(i))
            self.layers.append(output)

    def __call__(self, inputs, training):
        output = inputs
        for layer in self.layers:
            output = layer(output, training)
        return output
