import numpy as np
import tensorflow as tf

from tv_graph_cnn.layers import fir_tv_filtering_einsum, chebyshev_convolution, jtv_chebyshev_convolution


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=1 / np.prod(shape[:-1]), dtype=tf.float32)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = 0.1 * tf.ones(shape=shape, dtype=tf.float32)
    return tf.Variable(initial)


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def fc_fn(x, output_units):
    """
    Neural network consisting on 1 convolutional chebyshev layer and one dense layer
    :param x: input signal
    :param L: graph laplacian
    :param output_units: number of output units
    :param filter_order: order of convolution
    :param num_filters: number of parallel filters
    :return: computational graph
    """
    keep_prob = tf.placeholder(tf.float32)

    with tf.name_scope("fc"):
        fc_input = tf.layers.flatten(x)
        fc = tf.layers.dense(
            inputs=fc_input,
            units=output_units,
            activation=None,
            use_bias=True
        )

    return fc, keep_prob


def _batch_normalization(input, is_training=True, scope=None):
    # Note: is_training is tf.placeholder(tf.bool) type
    return tf.cond(is_training,
                   lambda: tf.contrib.layers.batch_norm(input, is_training=True, decay=0.9,
                                                        center=False, updates_collections=None, scope=scope),
                   lambda: tf.contrib.layers.batch_norm(input, is_training=False, decay=0.9,
                                                        updates_collections=None, center=False, scope=scope,
                                                        reuse=True))


def deep_fir_tv_fc_fn(x, L, num_classes, time_filter_orders, vertex_filter_orders, num_filters, poolings):
    assert len(time_filter_orders) == len(vertex_filter_orders) == len(num_filters), \
        "Filter parameters should all be of the same length"

    n_layers = len(time_filter_orders)
    keep_prob = tf.placeholder(tf.float32)
    phase = tf.placeholder(tf.bool)
    # Convolutional layers
    pool = x
    for n in range(n_layers):
        with tf.name_scope("conv%d" % n):
            conv = _fir_tv_layer(pool, L, time_filter_orders[n], vertex_filter_orders[n], num_filters[n])
            conv = _batch_normalization(conv, is_training=phase, scope="conv%d" % n)
            conv = tf.nn.relu(conv)
        with tf.name_scope("drop%d" % n):
            drop = tf.nn.dropout(conv, keep_prob=keep_prob)
        with tf.name_scope("subsampling%d" % n):
            pool = tf.layers.max_pooling2d(
                inputs=drop,
                pool_size=(1, poolings[n]),
                padding="same",
                strides=(1, poolings[n])
            )

    # Last fully connected layer
    with tf.name_scope("fc"):
        fc_input = tf.layers.flatten(pool)
        fc = tf.layers.dense(
            inputs=fc_input,
            units=num_classes,
            activation=None,
            use_bias=True,
        )
    return fc, keep_prob, phase


def _fir_tv_layer(x, L, time_filter_order, vertex_filter_order, num_filters):
    _, _, _, num_channels = x.get_shape()
    num_channels = int(num_channels)
    with tf.name_scope("fir_tv"):
        with tf.name_scope("weights"):
            hfir = weight_variable([vertex_filter_order, time_filter_order, num_channels, num_filters])
            variable_summaries(hfir)
        graph_conv = fir_tv_filtering_einsum(x, L, hfir, None, "chebyshev")
    return graph_conv


def fir_tv_fc_fn(x, L, output_units, time_filter_order, vertex_filter_order, num_filters):
    """
    Neural network consisting on 1 FIR-TV layer and one dense layer
    :param x: input signal
    :param L: graph laplacian
    :param output_units: number of output units
    :param time_filter_order: order of time convolution
    :param vertex_filter_order: order of vertex convolution
    :param num_filters: number of parallel filters
    :return: computational graph
    """
    graph_conv = _fir_tv_layer(x, L, time_filter_order, vertex_filter_order, num_filters)
    graph_conv = tf.nn.relu(graph_conv)
    with tf.name_scope("dropout"):
        keep_prob = tf.placeholder(tf.float32)
        dropout = tf.nn.dropout(graph_conv, keep_prob=keep_prob)

    with tf.name_scope("fc"):
        fc_input = tf.layers.flatten(dropout)
        fc = tf.layers.dense(
            inputs=fc_input,
            units=output_units,
            activation=None,
            use_bias=True,
        )
    return fc, keep_prob


def cheb_fc_fn(x, L, output_units, filter_order, num_filters):
    """
    Neural network consisting on 1 convolutional chebyshev layer and one dense layer
    :param x: input signal
    :param L: graph laplacian
    :param output_units: number of output units
    :param filter_order: order of convolution
    :param num_filters: number of parallel filters
    :return: computational graph
    """
    with tf.name_scope("chebyshev_conv"):
        with tf.name_scope("weights"):
            Wcheb = weight_variable([filter_order, num_filters])
            variable_summaries(Wcheb)
        with tf.name_scope("biases"):
            bcheb = bias_variable([num_filters])
            variable_summaries(bcheb)
        graph_conv = chebyshev_convolution(x, L, Wcheb, bcheb)
        graph_conv = tf.nn.relu(graph_conv)
        with tf.name_scope("dropout"):
            keep_prob = tf.placeholder(tf.float32)
            dropout = tf.nn.dropout(graph_conv, keep_prob=keep_prob)
    with tf.name_scope("fc"):
        fc_input = tf.layers.flatten(dropout)
        fc = tf.layers.dense(
            inputs=fc_input,
            units=output_units,
            activation=None,
            use_bias=True
        )

    return fc, keep_prob


def jtv_cheb_fc_fn(x, L, output_units, filter_order, num_filters):
    """
    Neural network consisting on 1 joint time-vertex convolutional chebyshev layer and one dense layer
    :param x: input signal
    :param L: graph laplacian
    :param output_units: number of output units
    :param filter_order: order of convolution
    :param num_filters: number of parallel filters
    :return: computational graph
    """
    with tf.name_scope("jtv_chebyshev_conv"):
        with tf.name_scope("weights"):
            Wcheb = weight_variable([filter_order, num_filters])
            variable_summaries(Wcheb)
        with tf.name_scope("biases"):
            bcheb = bias_variable([num_filters])
            variable_summaries(bcheb)
        graph_conv = jtv_chebyshev_convolution(x, L, Wcheb, bcheb)
        graph_conv = tf.nn.relu(graph_conv)
        with tf.name_scope("dropout"):
            keep_prob = tf.placeholder(tf.float32)
            dropout = tf.nn.dropout(graph_conv, keep_prob=keep_prob)
    with tf.name_scope("fc"):
        fc_input = tf.layers.flatten(dropout)
        fc = tf.layers.dense(
            inputs=fc_input,
            units=output_units,
            activation=None,
            use_bias=True
        )

    return fc, keep_prob
