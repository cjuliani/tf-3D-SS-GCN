import tensorflow as tf
from tensorflow.keras.layers import Layer, BatchNormalization
from keras import regularizers
import config


class Dense(Layer):
    def __init__(self, out_dim, bias=True, activation=tf.nn.relu, batchnorm=False,
                 regularizer=config.WEIGHTS_REG, **kwargs):
        super(Dense, self).__init__(**kwargs)

        self.output_dim = out_dim
        self.activation = activation
        self.bias = bias
        self.batchnorm = batchnorm
        self.regularizer = regularizers.get(regularizer)

    def build(self, input_shape):
        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.w = self.add_weight(shape=(input_shape[-1], self.output_dim),
                                     initializer='glorot_uniform',
                                     regularizer=self.regularizer,
                                     trainable=True, name='weights')
            self.b = self.add_weight(shape=(self.output_dim,),
                                     initializer='random_normal',
                                     regularizer=self.regularizer,
                                     trainable=True, name='bias')
        self.normalizer = BatchNormalization()

    def call(self, inputs, training=True):
        output = tf.compat.v1.matmul(inputs, self.w)
        if self.bias: output = tf.add(output, self.b)
        if self.batchnorm is not False: output = self.normalizer(output, training=training)
        if self.activation is not False: output = self.activation(output)
        return output


class SharedConv(Layer):
    def __init__(self, filters, strides=None, bias=True, activation=tf.nn.relu, padding='VALID', batchnorm=False,
                 regularizer=config.WEIGHTS_REG, **kwargs):
        super(SharedConv, self).__init__(**kwargs)

        if strides is None: strides = [1]
        self.filters = filters
        self.strides = strides
        self.bias = bias
        self.activation = activation
        self.padding = padding
        self.batchnorm = batchnorm
        self.regularizer = regularizers.get(regularizer)

    def build(self, input_shape):
        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.w = self.add_weight(shape=(1, input_shape[-1], self.filters),
                                     initializer='glorot_uniform',
                                     regularizer=self.regularizer,
                                     trainable=True, name='weights')
            self.b = self.add_weight(shape=(self.filters,),
                                     initializer='random_normal',
                                     regularizer=self.regularizer,
                                     trainable=True, name='bias')
        self.normalizer = BatchNormalization()

    def call(self, inputs, training=True):
        output = tf.nn.conv1d(inputs, filters=self.w, stride=self.strides, padding=self.padding, data_format='NWC')
        if self.bias: output = tf.add(output, self.b)
        if self.batchnorm is not False: output = self.normalizer(output, training=training)
        if self.activation is not False: output = self.activation(output)
        return output


class FullyConnected(Layer):
    def __init__(self, filters, strides=None, bias=True, activation=tf.nn.relu, padding='VALID', batchnorm=False,
                 regularizer=config.WEIGHTS_REG, **kwargs):
        super(FullyConnected, self).__init__(**kwargs)

        if strides is None: strides = 1
        self.filters = filters
        self.bias = bias
        self.strides = strides
        self.activation = activation
        self.padding = padding
        self.batchnorm = batchnorm
        self.regularizer = regularizers.get(regularizer)

    def build(self, input_shape):
        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.w = self.add_weight(shape=(input_shape[-1], self.filters),
                                     initializer='glorot_uniform',
                                     regularizer=self.regularizer,
                                     trainable=True, name='weights')
            self.b = self.add_weight(shape=(self.filters,),
                                     initializer='random_normal',
                                     regularizer=self.regularizer,
                                     trainable=True, name='bias')
            self.normalizer = BatchNormalization()
            super(FullyConnected, self).build(input_shape)

    def call(self, inputs, training=True):
        output = tf.matmul(inputs, self.w)
        if self.bias: output = tf.add(output, self.b)
        if self.batchnorm is not False: output = self.normalizer(output, training=training)
        if self.activation is not False: output = self.activation(output)
        return output
