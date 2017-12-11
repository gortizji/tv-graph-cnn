import time
import os
import argparse
import sys
import numpy as np

from pygsp import graphs

import tensorflow as tf

from graph_utils import initialize_laplacian_tensor
from signal_classification.models import fir_tv_fc_fn, cheb_fc_fn, jtv_cheb_fc_fn, deep_fir_tv_fc_fn, fc_fn, deep_cheb_fc_fn
from synthetic_data.data_generation import generate_spectral_samples

FLAGS = None
TEMPDIR = "/users/gortizjimenez/tmp/"


class TemporalGraphBatchSource:
    def __init__(self, data, labels, repeat=False):
        self.data = data
        self.labels = labels
        self.indices = np.random.permutation(self.dataset_length)
        self.idx = 0
        self.repeat = repeat
        self.end = False

    @property
    def dataset_length(self):
        return self.data.shape[0]

    def next_batch(self, batch_size):
        if self.end:
            if not self.repeat:
                return None, self.end
            else:
                self.end = False

        end_idx = self.idx + batch_size

        if end_idx < self.dataset_length:
            batch = [self.data[self.indices[self.idx:end_idx]], self.labels[self.indices[self.idx:end_idx]]]
        else:
            self.end = True
            batch = [self.data[self.indices[self.idx:]], self.labels[self.indices[self.idx:]]]
            if self.repeat:
                end_idx = end_idx % self.dataset_length
                self.indices = np.random.permutation(self.dataset_length)
                batch[0] = np.concatenate((batch[0], self.data[self.indices[:end_idx]]), axis=0)
                batch[1] = np.concatenate((batch[1], self.labels[self.indices[:end_idx]]), axis=0)
        self.idx = end_idx

        return batch, self.end

    def restart(self):
        self.end = False
        self.idx = 0


def _fill_feed_dict(mb_source, x, y, dropout, phase, is_training):
    (data, labels), is_end = mb_source.next_batch(FLAGS.batch_size)
    labels_one_hot = tf.one_hot(labels, FLAGS.num_classes).eval()
    feed_dict = {x: data, y: labels_one_hot, dropout: 0.5 if is_training else 1, phase: is_training}
    still_data = not is_end
    return feed_dict, still_data


def run_training(L, train_mb_source, test_mb_source):
    """Performs training and evaluation."""

    # Create data placeholders
    x = tf.placeholder(tf.float32, [None, FLAGS.num_vertices, FLAGS.num_frames, 1])
    y_ = tf.placeholder(tf.float32, [None, FLAGS.num_classes])

    # Initialize model
    if FLAGS.model_type == "deep_fir":
        print("Training deep FIR-TV model...")
        logits, phase = deep_fir_tv_fc_fn(x, L, FLAGS.num_classes, FLAGS.time_filter_orders,
                                                   FLAGS.vertex_filter_orders, FLAGS.num_filters, FLAGS.poolings)
        dropout = tf.placeholder(tf.float32)
    elif FLAGS.model_type == "deep_cheb":
        print("Training deep Chebyshev time invariant model...")
        xt = tf.transpose(x, perm=[0, 1, 3, 2])
        logits, phase = deep_cheb_fc_fn(xt, L, FLAGS.num_classes, FLAGS.vertex_filter_orders, FLAGS.num_filters)
        dropout = tf.placeholder(tf.float32)
    elif FLAGS.model_type == "fc":
        print("Training linear classifier model...")
        logits = fc_fn(x, FLAGS.num_classes)
        dropout = tf.placeholder(tf.float32)
        phase = tf.placeholder(tf.bool)
    else:
        raise ValueError("model_type not valid.")

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

    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        # Select optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        opt_train = optimizer.minimize(loss, global_step=global_step)

    # Build the summary Tensor based on the TF collection of Summaries.
    summary = tf.summary.merge_all()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    print("Number of training parameters:", _number_of_trainable_params())

    # Run session
    with tf.Session() as sess:

        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

        sess.run(tf.global_variables_initializer())

        MAX_STEPS = FLAGS.num_epochs * FLAGS.num_train // FLAGS.batch_size

        # Start training loop
        epoch_count = 0
        for step in range(MAX_STEPS):

            start_time = time.time()

            feed_dict, _ = _fill_feed_dict(train_mb_source, x, y_, dropout, phase, True)

            # Perform one training iteration
            _, loss_value = sess.run([opt_train, loss],
                                     feed_dict=feed_dict)

            duration = time.time() - start_time

            if step % (FLAGS.num_train // FLAGS.batch_size) == 0:
                print("Epoch %d" % epoch_count)
                print("--------------------")
                epoch_count += 1

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
            if (step + 1) % (FLAGS.num_train // FLAGS.batch_size) == 0 or (step + 1) == MAX_STEPS or (step + 1) % 30 == 0:
                checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)

                still_data = True
                test_accuracies = []
                test_mb_source.restart()
                while still_data:
                    test_feed_dict, still_data = _fill_feed_dict(test_mb_source, x, y_, dropout, phase, False)
                    test_accuracies.append(sess.run(accuracy, feed_dict=test_feed_dict))

                print("--------------------")
                print('Test accuracy = %.2f' % np.mean(test_accuracies))
                if (step + 1) % (FLAGS.num_train // FLAGS.batch_size) == 0:
                    print("====================")
                else:
                    print("--------------------")


def main(_):
    # Initialize tempdir
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)

    # Initialize data
    G = graphs.ErdosRenyi(FLAGS.num_vertices, 0.1)
    G.compute_laplacian("normalized")
    L = initialize_laplacian_tensor(G.W)

    train_data, train_labels = generate_spectral_samples(
        N=FLAGS.num_train // 4,
        G=G,
        T=FLAGS.num_frames,
        f_h=FLAGS.f_h,
        f_l=FLAGS.f_h,
        lambda_h=FLAGS.lambda_h,
        lambda_l=FLAGS.lambda_l,
        sigma=FLAGS.sigma,
        sigma_n=FLAGS.sigma_n
    )
    train_mb_source = TemporalGraphBatchSource(train_data, train_labels, repeat=True)

    test_data, test_labels = generate_spectral_samples(
        N=FLAGS.num_test // 4,
        G=G,
        T=FLAGS.num_frames,
        f_h=FLAGS.f_h,
        f_l=FLAGS.f_h,
        lambda_h=FLAGS.lambda_h,
        lambda_l=FLAGS.lambda_l,
        sigma=FLAGS.sigma,
        sigma_n=FLAGS.sigma_n
    )

    test_mb_source = TemporalGraphBatchSource(test_data, test_labels, repeat=False)

    # Run training and evaluation loop
    run_training(L, train_mb_source,  test_mb_source)


def _number_of_trainable_params():
    return np.sum([np.product(x.shape) for x in tf.trainable_variables()])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        default="deep_fir",
        help="Model type"
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-4,
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
        default=10000,
        help='Number of training samples.'
    )
    parser.add_argument(
        '--num_test',
        type=int,
        default=1000,
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
        default=128,
        help='Number of temporal frames.'
    )
    parser.add_argument(
        '--vertex_filter_orders',
        type=list,
        default=[3, 3, 3],
        help='Convolution vertex order.'
    )
    parser.add_argument(
        '--time_filter_orders',
        type=list,
        default=[3, 3, 3],
        help='Convolution time order.'
    )
    parser.add_argument(
        '--num_filters',
        type=list,
        default=[8, 16, 32],
        help='Number of parallel convolutional filters.'
    )
    parser.add_argument(
        '--poolings',
        type=list,
        default=[4, 4, 4],
        help='Pooling sizes.'
    )
    parser.add_argument(
        '--f_h',
        type=int,
        default=40,
        help='High pass cut frequency (time)'
    )
    parser.add_argument(
        '--f_l',
        type=int,
        default=60,
        help='Low pass cut frequency (time)'
    )
    parser.add_argument(
        '--lambda_h',
        type=int,
        default=40,
        help='High pass cut frequency (graph)'
    )
    parser.add_argument(
        '--lambda_l',
        type=int,
        default=60,
        help='low pass cut frequency (graph)'
    )
    parser.add_argument(
        '--sigma',
        type=float,
        default=2,
        help='Source standard deviation.'
    )
    parser.add_argument(
        '--sigma_n',
        type=float,
        default=2,
        help='Noise standard deviation.'
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        default=4,
        help='Number of classes to separate.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
