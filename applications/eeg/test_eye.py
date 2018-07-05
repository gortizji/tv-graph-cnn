import time
import os
import argparse
import sys
import numpy as np
import json
import matplotlib

matplotlib.use("Agg")

from pygsp import graphs

import tensorflow as tf

from applications.eeg.eye_dataset import EyeMinibatchSource, MONTAGE, load_data, train_validation_test_split

from graph_utils.laplacian import initialize_laplacian_tensor
from graph_utils.coarsening import coarsen, perm_data, keep_pooling_laplacians
from applications.eeg.data_utils import create_spatial_eeg_graph, create_data_eeg_graph
from applications.eeg.models import deep_fir_tv_fc_fn, deep_cheb_fc_fn

from graph_utils.visualization import plot_tf_fir_filter

FLAGS = None
FILEDIR = os.path.dirname(os.path.realpath(__file__))
TEMPDIR = os.path.realpath(os.path.join(FILEDIR, "../experiments/eye"))

EPOCH_SIZE = 0
NUM_CLASSES = 2


def _fill_feed_dict(mb_source, x, y, dropout, phase, is_training):
    (data, labels), is_end = mb_source.next_batch(FLAGS.batch_size)
    feed_dict = {x: data, y: labels, dropout: FLAGS.dropout if is_training else 1, phase: is_training}
    still_data = not is_end
    return feed_dict, still_data


def run_training(L, train_mb_source, test_mb_source):
    """Performs training and evaluation."""

    # Create data placeholders
    num_vertices, _ = L[0].get_shape()
    x = tf.placeholder(tf.float32, [None, num_vertices, 2 * FLAGS.margin], name="x")
    x_ = tf.expand_dims(x, axis=-1)
    y = tf.placeholder(tf.uint8, name="labels")

    # Initialize model
    if FLAGS.model_type == "deep_fir":
        print("Training deep FIR-TV model...")
        out, phase, dropout = deep_fir_tv_fc_fn(x=x_,
                                                L=L,
                                                time_filter_orders=FLAGS.time_filter_orders,
                                                vertex_filter_orders=FLAGS.vertex_filter_orders,
                                                num_filters=FLAGS.num_filters,
                                                time_poolings=FLAGS.time_poolings,
                                                vertex_poolings=FLAGS.vertex_poolings)
        out = tf.squeeze(out)
    elif FLAGS.model_type == "deep_cheb":
        print("Training deep FIR-TV model...")
        out, phase, dropout = deep_cheb_fc_fn(x=x_,
                                              L=L,
                                              vertex_filter_orders=FLAGS.vertex_filter_orders,
                                              num_filters=FLAGS.num_filters,
                                              vertex_poolings=FLAGS.vertex_poolings)
        out = tf.squeeze(out)
    else:
        raise ValueError("model_type not valid.")

    # Define loss
    with tf.name_scope("loss"):
        cross_entropy = tf.losses.log_loss(y, out, reduction=tf.losses.Reduction.NONE)
        loss = tf.reduce_mean(cross_entropy)
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        l1_loss = tf.add_n([tf.norm(v, ord=1) for v in tf.trainable_variables()])
        loss += FLAGS.weight_decay * l2_loss + FLAGS.l1_reg * l1_loss
        tf.summary.scalar('xentropy', loss)

        # Define metric
    with tf.name_scope("metric"):
        prediction = tf.cast(tf.greater_equal(out, 0.5), tf.uint8)
        correct_prediction = tf.equal(prediction, y)
        correct_prediction = tf.cast(correct_prediction, tf.float32, name="correct_prediction")
        accuracy = tf.reduce_mean(correct_prediction, name="accuracy")
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
        train_writer = tf.summary.FileWriter(FLAGS.log_dir + "/train", sess.graph)
        test_writer = tf.summary.FileWriter(FLAGS.log_dir + "/test", sess.graph)

        sess.run(tf.global_variables_initializer())

        MAX_STEPS = FLAGS.num_epochs * EPOCH_SIZE // FLAGS.batch_size

        # Start training loop
        epoch_count = 0
        for step in range(MAX_STEPS):

            start_time = time.time()

            feed_dict, _ = _fill_feed_dict(train_mb_source, x, y, dropout, phase, True)

            # Perform one training iteration
            _, loss_value = sess.run([opt_train, loss],
                                     feed_dict=feed_dict)

            duration = time.time() - start_time

            if step % (EPOCH_SIZE // FLAGS.batch_size) == 0:
                print("Epoch %d" % epoch_count)
                print("--------------------")
                epoch_count += 1

            # Write the summaries and print an overview fairly often.
            if step % 10 == 0:
                # Print status to stdout.
                accuracy_value = sess.run(accuracy, feed_dict=feed_dict)
                # print(sess.run(prediction, feed_dict=feed_dict))
                print('Step %d: loss = %.2f accuracy = %.2f (%.3f sec)' % (step, loss_value, accuracy_value, duration))
                # Update the events file.
                summary_str = sess.run(summary, feed_dict=feed_dict)
                train_writer.add_summary(summary_str, step)
                train_writer.flush()

            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) % (EPOCH_SIZE // FLAGS.batch_size) == 0 or (step + 1) == MAX_STEPS:
                checkpoint_file = os.path.join(FLAGS.log_dir, 'model')
                saver.save(sess, checkpoint_file, global_step=step)

                test_accuracy = _eval_metric(sess, correct_prediction, dropout, phase, x, y, test_mb_source)

                test_summary = tf.Summary(value=[tf.Summary.Value(tag="test_accuracy", simple_value=test_accuracy)])
                test_writer.add_summary(test_summary, step)
                test_writer.flush()

                print("--------------------")
                print('Test accuracy = %.2f' % test_accuracy)
                if (step + 1) % (EPOCH_SIZE // FLAGS.batch_size) == 0:
                    print("====================")
                else:
                    print("--------------------")


def _eval_metric(sess, correct_prediction, dropout, phase, x, y, test_mb_source):
    still_data = True
    test_correct_predictions = []
    test_mb_source.restart()
    while still_data:
        test_feed_dict, still_data = _fill_feed_dict(test_mb_source, x, y, dropout, phase, False)
        if still_data is False:
            break
        test_correct_predictions.append(sess.run(correct_prediction, feed_dict=test_feed_dict))

    test_correct_predictions = np.concatenate(test_correct_predictions, axis=0)

    return np.mean(test_correct_predictions)


def run_eval(test_mb_source):
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(
            os.path.join(FLAGS.log_dir, "model-" + str(_last_checkpoint(FLAGS.log_dir)) + ".meta"))
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.log_dir))
        graph = tf.get_default_graph()

        # Get inputs
        x = graph.get_tensor_by_name("x:0")
        y = graph.get_tensor_by_name("labels:0")
        keep_prob = graph.get_tensor_by_name("keep_prob:0")
        phase = graph.get_tensor_by_name("phase:0")

        # Get output
        correct_prediction = graph.get_tensor_by_name("metric/correct_prediction:0")

        print("Evaluation accuracy: %.2f" % _eval_metric(sess, correct_prediction, keep_prob, phase, x, y,
                                                         test_mb_source))

        for idx, v in enumerate([v for v in tf.trainable_variables() if "conv" in v.name]):
            plot_tf_fir_filter(sess, v, os.path.join(FLAGS.log_dir, "conv_%d" % idx))


def main(_):
    global EPOCH_SIZE
    # Initialize tempdir
    if FLAGS.action == "eval" and FLAGS.read_dir is not None:
        FLAGS.log_dir = FLAGS.read_dir
    else:
        FLAGS.log_dir = os.path.join(FLAGS.log_dir, FLAGS.model_type)
        exp_n = _last_exp(FLAGS.log_dir) + 1 if FLAGS.action == "train" else _last_exp(FLAGS.log_dir)
        FLAGS.log_dir = os.path.join(FLAGS.log_dir, "exp_" + str(exp_n))

    print(FLAGS.log_dir)

    # Initialize data
    X, y = load_data()
    G = create_spatial_eeg_graph(MONTAGE, q=FLAGS.q, k=FLAGS.k)
    #G = create_data_eeg_graph(MONTAGE, X)
    G.compute_laplacian("normalized")

    if FLAGS.action == "train":
        if tf.gfile.Exists(FLAGS.log_dir):
            tf.gfile.DeleteRecursively(FLAGS.log_dir)
        tf.gfile.MakeDirs(FLAGS.log_dir)

        # Prepare pooling
        num_levels = _number_of_pooling_levels(FLAGS.vertex_poolings)
        error = True
        while error:
            try:
                adjacencies, perm = coarsen(G.A, levels=num_levels)  # Coarsens in powers of 2
                error = False
            except IndexError:
                error = True
                continue

        np.save(os.path.join(FLAGS.log_dir, "ordering"), perm)
        L = [initialize_laplacian_tensor(A) for A in adjacencies]
        L = keep_pooling_laplacians(L, FLAGS.vertex_poolings)

    elif FLAGS.action == "eval":
        perm = np.load(os.path.join(FLAGS.log_dir, "ordering.npy"))

    X = perm_data(X, perm)
    train_samples, test_samples = train_validation_test_split(len(y), FLAGS.margin, FLAGS.test_size)

    if FLAGS.action == "train":
        EPOCH_SIZE = train_samples.shape[0]
        print(EPOCH_SIZE)
        train_mb_source = EyeMinibatchSource(X, y, train_samples, margin=FLAGS.margin, repeat=True)
        EPOCH_SIZE = train_mb_source.dataset_length

    test_mb_source = EyeMinibatchSource(X, y, test_samples, margin=FLAGS.margin, repeat=False)

    if FLAGS.action == "train":
        params = vars(FLAGS)
        with open(os.path.join(FLAGS.log_dir, "params.json"), "w") as f:
            json.dump(params, f)

        # Run training and evaluation loop
        print("Training model...")
        run_training(L, train_mb_source, test_mb_source)
    elif FLAGS.action == "eval":
        print("Evaluating model...")
        run_eval(test_mb_source)
    else:
        raise ValueError("No valid action selected")


def _number_of_pooling_levels(vertex_poolings):
    return np.log2(np.prod(vertex_poolings)).astype(int)


def _number_of_trainable_params():
    return np.sum([np.product(x.shape) for x in tf.trainable_variables()])


def _last_checkpoint(log_dir):
    checkpoints = []
    if not os.path.exists(log_dir):
        raise IOError("No such file or directory:", log_dir)

    for file in os.listdir(log_dir):
        if ".meta" not in file:
            continue
        else:
            checkpoints.append(int(file.split("-")[1].split(".")[0]))

    return max(checkpoints)


def _last_exp(log_dir):
    exp_numbers = []
    if not os.path.exists(log_dir):
        return 0
    for file in os.listdir(log_dir):
        if "exp" not in file:
            continue
        else:
            exp_numbers.append(int(file.split("_")[1]))
    return max(exp_numbers) if len(exp_numbers) > 0 else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        default="deep_fir",
        help="Model type"
    )
    parser.add_argument(
        "--action",
        type=str,
        default="train",
        help="Action to perform on the model"
    )
    parser.add_argument(
        "--read_dir",
        type=str,
        default=None
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-3,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=250,
        help='Number of epochs to run trainer.'
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=1,
        help="Dropout keep_rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0,
        help="Weight decay strength"
    )
    parser.add_argument(
        "--l1_reg",
        type=float,
        default=0,
        help="l1 regularization strength"
    )
    parser.add_argument(
        "--q",
        type=float,
        default=0.1,
        help="RBF kernel"
    )
    parser.add_argument(
        "--k",
        type=float,
        default=0.1,
        help="Distance threshold"
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=512,
        help='Minibatch size in samples.'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default=os.path.join(TEMPDIR, "test"),
        help='Logging directory'
    )
    parser.add_argument(
        '--vertex_filter_orders',
        type=int,
        default=[3, 3, 2, 2],
        nargs="+",
        help='Convolution vertex order.'
    )
    parser.add_argument(
        '--time_filter_orders',
        type=int,
        nargs="+",
        default=[3, 3, 2, 2],
        help='Convolution time order.'
    )
    parser.add_argument(
        '--num_filters',
        type=int,
        nargs="+",
        default=[8, 16, 32, 64],
        help='Number of parallel convolutional filters.'
    )
    parser.add_argument(
        '--time_poolings',
        type=int,
        nargs="+",
        default=[2, 2, 2, 2],
        help='Time pooling sizes.'
    )
    parser.add_argument(
        "--vertex_poolings",
        type=int,
        nargs="+",
        default=[1, 1, 2, 2],
        help="Vertex pooling sizes"
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=32,
        help="Right and left margin"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.25,
        help="Percentage left for test"
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
