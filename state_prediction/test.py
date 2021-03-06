import time
import os
import argparse
import sys
import numpy as np

from pygsp import graphs

import tensorflow as tf

from graph_utils.laplacian import initialize_laplacian_tensor
from source_localization.models import fir_tv_fc_fn, cheb_fc_fn, fc_fn
from synthetic_data.data_generation import generate_wave_samples

FLAGS = None
TEMPDIR = "/users/gortizjimenez/tmp/"


class TemporalGraphBatchSource:
    def __init__(self, data):
        self.data = data
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
            data_batch = self.data[self.idx:end_idx]
        else:
            data_batch = self.data[self.idx:]
            end_idx = end_idx % self.dataset_length
            data_batch = np.concatenate([data_batch, self.data[:end_idx]], axis=0)
        batch = [data_batch[:, :, :-1], data_batch[:, :, -1]]
        self.idx = end_idx

        return batch


def _fill_feed_dict(mb_source, x, y, dropout):
    past, future = mb_source.next_batch(FLAGS.batch_size)
    feed_dict = {x: past, y: future, dropout: 0.5}
    return feed_dict


def run_training(mb_source, L, test_data):
    """Performs training and evaluation."""

    # Create data placeholders
    x = tf.placeholder(tf.float32, [None, FLAGS.num_vertices, FLAGS.num_frames-1])
    y_ = tf.placeholder(tf.float32, [None, FLAGS.num_vertices])

    # Initialize model
    prediction, dropout = fir_tv_fc_fn(x, L, FLAGS.num_vertices, FLAGS.time_filter_order, FLAGS.filter_order, FLAGS.num_filters)
    # prediction, dropout = cheb_fc_fn(x, L, FLAGS.num_vertices, FLAGS.filter_order, FLAGS.num_filters)
    # prediction, dropout = fc_fn(x, FLAGS.num_vertices)

    # Define loss
    with tf.name_scope("loss"):
        mse = tf.losses.mean_squared_error(y_, prediction)
        loss = tf.reduce_mean(mse)
        tf.summary.scalar('mse', loss)
    with tf.name_scope("metric"):
        metric, metric_opt = tf.metrics.root_mean_squared_error(y_, prediction)

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

        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        MAX_STEPS = FLAGS.num_epochs * FLAGS.num_train // FLAGS.batch_size

        test_feed_dict = {x: test_data[:, :, :-1], y_: test_data[:, :, -1], dropout: 1}

        # Start training loop
        epoch_count = 0
        for step in range(MAX_STEPS):

            start_time = time.time()

            feed_dict = _fill_feed_dict(mb_source, x, y_, dropout)

            # Perform one training iteration
            sess.run([opt_train, metric_opt], feed_dict=feed_dict)

            duration = time.time() - start_time

            if step % (FLAGS.num_train // FLAGS.batch_size) == 0:
                print("Epoch %d" % epoch_count)
                print("--------------------")
                epoch_count += 1

            # Write the summaries and print an overview fairly often.
            if step % 10 == 0:
                # Print status to stdout.
                print('Step %d: rmse = %.2f (%.3f sec)' % (step, metric.eval(), duration))
                # Update the events file.
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) % (FLAGS.num_train // FLAGS.batch_size) == 0 or (step + 1) == MAX_STEPS:
                checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)

                sess.run(metric_opt, feed_dict=test_feed_dict)
                print("--------------------")
                print('Test rmse = %.2f' % metric.eval())
                print("====================")


def main(_):
    # Initialize tempdir
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)

    # Initialize data
    G = graphs.ErdosRenyi(FLAGS.num_vertices, 0.1, seed=42)
    G.compute_laplacian("normalized")
    L = initialize_laplacian_tensor(G.W)
    W = (G.W).astype(np.float32)

    train_data, _ = generate_wave_samples(FLAGS.num_train, W, T=FLAGS.num_frames, sigma=FLAGS.sigma, signal="cosine")
    train_mb_source = TemporalGraphBatchSource(train_data)

    test_data, _ = generate_wave_samples(FLAGS.num_test, W, T=FLAGS.num_frames, sigma=FLAGS.sigma, signal="cosine")

    # Run training and evaluation loop
    run_training(train_mb_source, L, test_data)


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
        default=20,
        help='Number of epochs to run trainer.'
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
        default=25,
        help='Number of graph vertices.'
    )
    parser.add_argument(
        '--num_frames',
        type=int,
        default=32,
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
        default=0.1,
        help='Noise typical deviation.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
