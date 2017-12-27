import numpy as np

from tv_graph_cnn.layers import fir_tv_filtering_conv1d

import tensorflow as tf


def _weight_variable(shape):
    """_weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=1 / np.prod(shape[:-1]), dtype=tf.float32)
    return tf.Variable(initial)


def _bias_variable(shape):
    """_bias_variable generates a bias variable of a given shape."""
    initial = 0.1 * tf.ones(shape=shape, dtype=tf.float32)
    return tf.Variable(initial)


def _variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        tf.summary.histogram('histogram', var)


def _batch_normalization(input, is_training=True, scope=None):
    # Note: is_training is tf.placeholder(tf.bool) type
    return tf.cond(is_training,
                   lambda: tf.contrib.layers.batch_norm(input, is_training=True, decay=0.8,
                                                        center=False, updates_collections=None, scope=scope),
                   lambda: tf.contrib.layers.batch_norm(input, is_training=False, decay=0.8,
                                                        updates_collections=None, center=False, scope=scope,
                                                        reuse=True))


def _fir_tv_layer(x, L, time_filter_order, vertex_filter_order, num_filters):
    _, _, _, num_channels = x.get_shape()
    num_channels = int(num_channels)
    with tf.name_scope("fir_tv"):
        with tf.name_scope("weights"):
            hfir = _weight_variable([vertex_filter_order, time_filter_order, num_channels, num_filters])
            _variable_summaries(hfir)
        graph_conv = fir_tv_filtering_conv1d(x, L, hfir, None, "chebyshev")
    return graph_conv


def deep_fir_tv_fc_fn(x, L, num_classes, time_filter_orders, vertex_filter_orders, num_filters, time_poolings,
                      vertex_poolings):
    assert len(time_filter_orders) == len(vertex_filter_orders) == len(num_filters) == len(time_poolings), \
        "Filter parameters should all be of the same length"

    n_layers = len(time_filter_orders)
    phase = tf.placeholder(tf.bool, name="phase")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    # Convolutional layers
    drop = x
    for n in range(n_layers):
        with tf.name_scope("conv%d" % n):
            conv = _fir_tv_layer(drop, L[n], time_filter_orders[n], vertex_filter_orders[n], num_filters[n])
            conv = _batch_normalization(conv, is_training=phase, scope="conv%d" % n)
            conv = tf.nn.elu(conv, name="conv%d" % n)

        with tf.name_scope("subsampling%d" % n):
            tpool = tf.layers.max_pooling2d(
                inputs=conv,
                pool_size=(1, time_poolings[n]),
                padding="same",
                strides=(1, time_poolings[n])
            )
        with tf.name_scope("vertex_pooling%d" % n):
            if vertex_poolings[n] > 1:
                vpool = tf.layers.max_pooling2d(
                    inputs=tpool,
                    pool_size=(vertex_poolings[n], 1),
                    padding="same",
                    strides=(vertex_poolings[n], 1)
                )
            else:
                vpool = tpool

        with tf.name_scope("drop%d" % n):
            drop = tf.nn.dropout(vpool, keep_prob=keep_prob)

    # Last fully connected layer
    with tf.name_scope("fc"):
        fc_input = tf.layers.flatten(drop)
        fc = tf.layers.dense(
            inputs=fc_input,
            units=num_classes,
            activation=None,
            use_bias=True
        )
        fc = tf.identity(fc, name="fc")
    return fc, phase, keep_prob

