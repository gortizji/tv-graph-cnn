import time
import os
import argparse
import sys
import numpy as np
import scipy

from scipy.sparse.linalg import eigsh
from pygsp import graphs

import tensorflow as tf

from tv_graph_cnn.layers import chebyshev_convolution, jtv_chebyshev_convolution, fir_tv_filtering_conv1d, fir_tv_filtering_matmul
from source_localization.data_generation import generate_diffusion_samples, generate_wave_samples

FLAGS = None
TEMPDIR = "/users/gortizjimenez/tmp/"


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


class TemporalGraphBatchSource:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.indices = np.random.permutation(np.arange(self.dataset_length))
        self.idx = 0
        self.is_finished = False

    @property
    def dataset_length(self):
        return self.data.shape[0]

    def next_batch(self, batch_size):
        if self.is_finished:
            self.indices = np.random.permutation(np.arange(self.dataset_length))
            self.is_finished = False

        end_idx = self.idx + batch_size

        if end_idx < self.dataset_length:
            batch = [self.data[self.idx:end_idx], self.labels[self.idx:end_idx]]
        else:
            batch = [self.data[self.idx:], self.labels[self.idx:]]
            end_idx = end_idx % self.dataset_length
            batch[0] = np.concatenate((batch[0], self.data[:end_idx]), axis=0)
            batch[1] = np.concatenate((batch[1], self.labels[:end_idx]), axis=0)
        self.idx = end_idx

        return batch


def laplacian(W, normalized=True):
    """Return the Laplacian of the weigth matrix."""

    W = W.astype(np.float32)
    # Degree matrix.
    d = W.sum(axis=0)

    # Laplacian matrix.
    if not normalized:
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        L = D - W
    else:
        d += np.spacing(np.array(0, W.dtype))
        d = 1 / np.sqrt(d)
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        I = scipy.sparse.identity(d.size, dtype=W.dtype)
        L = I - D * W * D

    # assert np.abs(L - L.T).mean() < 1e-9
    assert type(L) is scipy.sparse.csr.csr_matrix
    return L


def rescale_L(L, lmax=2):
    """Rescale the Laplacian eigenvalues in [-1,1]."""
    M, M = L.shape
    I = scipy.sparse.identity(M, format='csr', dtype=L.dtype)
    L /= lmax / 2
    L -= I
    return L


def _initialize_graph_laplacian(G):
    L = laplacian(G.W, normalized=True)
    l, _ = eigsh(L, k=1)
    print(l)
    L = rescale_L(L)
    L = L.tocoo()
    data = L.data.astype(np.float32)
    indices = np.empty((L.nnz, 2))
    indices[:, 0] = L.row
    indices[:, 1] = L.col
    L = tf.SparseTensor(indices, data, L.shape)
    L = tf.sparse_reorder(L)
    return L


def _fill_feed_dict(mb_source, x, y, dropout):
    data, labels = mb_source.next_batch(FLAGS.batch_size)
    labels_one_hot = tf.one_hot(labels, FLAGS.num_vertices).eval()
    feed_dict = {x: data, y: labels_one_hot, dropout: 0.5}
    return feed_dict


def run_training(mb_source, L, test_data, test_labels):
    """Performs training and evaluation."""

    # Create data placeholders
    x = tf.placeholder(tf.float32, [None, FLAGS.num_vertices, FLAGS.num_frames])
    y_ = tf.placeholder(tf.float32, [None, FLAGS.num_vertices])

    # Initialize model
    logits, dropout = fir_tv_fc_fn(x, L, FLAGS.num_vertices, FLAGS.time_filter_order, FLAGS.filter_order, FLAGS.num_filters)
    # logits, dropout = cheb_fc_fn(x, L, FLAGS.num_vertices, FLAGS.filter_order, FLAGS.num_filters)
    # logits, dropout = fc_fn(x, FLAGS.num_vertices)

    # Define loss
    with tf.name_scope("loss"):
        cross_entropy = tf.losses.softmax_cross_entropy(y_, logits=logits)
        loss = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('xentropy', loss)

    # Define metric
    with tf.name_scope("metric"):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)
        tf.summary.scalar('accuracy', accuracy)

    # Select optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    opt_train = optimizer.minimize(loss, global_step=global_step)

    # Build the summary Tensor based on the TF collection of Summaries.
    summary = tf.summary.merge_all()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Run session
    with tf.Session() as sess:

        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

        sess.run(tf.global_variables_initializer())

        MAX_STEPS = FLAGS.num_epochs * FLAGS.num_train // FLAGS.batch_size

        test_labels_one_hot = tf.one_hot(test_labels, FLAGS.num_vertices).eval()
        test_feed_dict = {x: test_data, y_: test_labels_one_hot, dropout: 1}

        # Start training loop
        for step in range(MAX_STEPS):

            start_time = time.time()

            feed_dict = _fill_feed_dict(mb_source, x, y_, dropout)

            # Perform one training iteration
            _, loss_value = sess.run([opt_train, loss],
                                     feed_dict=feed_dict)

            duration = time.time() - start_time

            # Write the summaries and print an overview fairly often.
            if step % 10 == 0:
                # Print status to stdout.
                accuracy_value = sess.run(accuracy, feed_dict=feed_dict)
                print('Step %d: loss = %.2f accuracy = %.2f (%.3f sec)' % (step, loss_value, accuracy_value, duration))
                # Update the events file.
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) % 50 == 0 or (step + 1) == MAX_STEPS:
                checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)

                accuracy_value = sess.run(accuracy, feed_dict=test_feed_dict)
                print('Test accuracy = %.2f' % accuracy_value)


def main(_):
    # Initialize tempdir
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)

    # Initialize data
    G = graphs.ErdosRenyi(FLAGS.num_vertices, 0.4, seed=42)
    G.compute_laplacian("normalized")
    L = _initialize_graph_laplacian(G)
    W = (G.W).astype(np.float32)

    train_data, train_labels = generate_wave_samples(FLAGS.num_train, W, T=FLAGS.num_frames, sigma=FLAGS.sigma)
    train_mb_source = TemporalGraphBatchSource(train_data, train_labels)

    test_data, test_labels = generate_wave_samples(FLAGS.num_test, W, T=FLAGS.num_frames, sigma=FLAGS.sigma)

    # Run training and evaluation loop
    run_training(train_mb_source, L, test_data, test_labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-5,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=100,
        help='Number of steps to run trainer.'
    )
    parser.add_argument(
        '--num_train',
        type=int,
        default=5000,
        help='Number of training samples.'
    )
    parser.add_argument(
        '--num_test',
        type=int,
        default=200,
        help='Number of test samples.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='Minibatch size in samples.'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default=os.path.join(TEMPDIR, "tensorflow/tv_graph_cnn/logs/cheb_net"),
        help='Logging directory'
    )
    parser.add_argument(
        '--num_vertices',
        type=int,
        default=100,
        help='Number of graph vertices.'
    )
    parser.add_argument(
        '--num_frames',
        type=int,
        default=24,
        help='Number of temporal frames.'
    )
    parser.add_argument(
        '--filter_order',
        type=int,
        default=5,
        help='Convolution vertex order.'
    )
    parser.add_argument(
        '--time_filter_order',
        type=int,
        default=5,
        help='Convolution time order.'
    )
    parser.add_argument(
        '--num_filters',
        type=int,
        default=32,
        help='Number of parallel convolutional filters.'
    )
    parser.add_argument(
        '--sigma',
        type=float,
        default=0.01,
        help='Noise typical deviation.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
