import tensorflow as tf

from tv_graph_cnn.layers import fir_tv_filtering_matmul, chebyshev_convolution, jtv_chebyshev_convolution


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.5, dtype=tf.float32)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.zeros(shape=shape, dtype=tf.float32)
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
            activation=tf.nn.relu,
            use_bias=True
        )

    return fc, keep_prob


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
    with tf.name_scope("FIR-TV"):
        with tf.name_scope("weights"):
            hfir = weight_variable([vertex_filter_order, time_filter_order, num_filters])
            variable_summaries(hfir)
        with tf.name_scope("biases"):
            bfir = bias_variable([num_filters])
            variable_summaries(bfir)

        graph_conv = fir_tv_filtering_matmul(x, L, hfir, bfir, "chebyshev")
        graph_conv = tf.nn.relu(graph_conv)
        with tf.name_scope("dropout"):
            keep_prob = tf.placeholder(tf.float32)
            dropout = tf.nn.dropout(graph_conv, keep_prob=keep_prob)
    with tf.name_scope("fc"):
        fc_input = tf.layers.flatten(dropout)
        fc = tf.layers.dense(
            inputs=fc_input,
            units=output_units,
            activation=tf.nn.relu,
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
            activation=tf.nn.relu,
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
            activation=tf.nn.relu,
            use_bias=True
        )

    return fc, keep_prob