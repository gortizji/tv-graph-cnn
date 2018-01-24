import time
import os
import argparse
import sys
import numpy as np
import json
import matplotlib

matplotlib.use("Agg")

from applications.seirs.data_utils import create_train_test_mb_sources, DATASET_FILE, create_graph

from graph_utils.laplacian import initialize_laplacian_tensor
from graph_utils.coarsening import coarsen, keep_pooling_laplacians
from applications.seirs.models import deep_fir_tv_fc_fn

from graph_utils.visualization import plot_tf_fir_filter

import tensorflow as tf

FLAGS = None
FILEDIR = os.path.dirname(os.path.realpath(__file__))
TEMPDIR = os.path.realpath(os.path.join(FILEDIR, "../experiments/seirs"))

EPOCH_SIZE = 0
NUM_CLASSES = 2


def _fill_feed_dict(mb_source, x, y_cp, y_tim, dropout, phase, is_training):
    (data, cp, tim), is_end = mb_source.next_batch(FLAGS.batch_size)
    feed_dict = {x: data, y_cp: cp, y_tim: tim, dropout: FLAGS.dropout if is_training else 1, phase: is_training}
    still_data = not is_end
    return feed_dict, still_data


def run_training(L, train_mb_source, test_mb_source):
    """Performs training and evaluation."""

    # Create data placeholders
    num_vertices, _ = L[0].get_shape()
    time_length = train_mb_source.time_length
    x = tf.placeholder(tf.float32, [None, num_vertices, time_length], name="x")
    x_ = tf.expand_dims(x, axis=-1)
    y_cp = tf.placeholder(tf.float32, name="cp")
    y_tim = tf.placeholder(tf.float32, name="tim")

    # Initialize model
    if FLAGS.model_type == "deep_fir":
        print("Training deep FIR-TV model...")
        out_cp, out_tim, phase, dropout = deep_fir_tv_fc_fn(x=x_,
                                                            L=L,
                                                            time_filter_orders=FLAGS.time_filter_orders,
                                                            vertex_filter_orders=FLAGS.vertex_filter_orders,
                                                            num_filters=FLAGS.num_filters,
                                                            time_poolings=FLAGS.time_poolings,
                                                            vertex_poolings=FLAGS.vertex_poolings)
    else:
        raise ValueError("model_type not valid.")

    # Define loss
    with tf.name_scope("loss"):
        mse_tim = (y_tim - out_tim) ** 2 / out_tim
        mse_cp = (y_cp - out_cp) ** 2 / out_cp
        loss = tf.reduce_mean(mse_tim + mse_cp)
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        l1_loss = tf.add_n([tf.norm(v, ord=1) for v in tf.trainable_variables()])
        loss += FLAGS.weight_decay * l2_loss + FLAGS.l1_reg * l1_loss
        tf.summary.scalar('mse', loss)

        # Define metric
    with tf.name_scope("metric"):
        rmse_tim = tf.sqrt(mse_tim)
        rmse_cp = tf.sqrt(mse_cp)
        metric = rmse_cp + rmse_tim
        mean_metric = tf.reduce_mean(metric)
        tf.summary.scalar('metric', metric)

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

            feed_dict, _ = _fill_feed_dict(train_mb_source, x, y_cp, y_tim, dropout, phase, True)

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
                metric_value = sess.run(mean_metric, feed_dict=feed_dict)
                # print(sess.run(prediction, feed_dict=feed_dict))
                print('Step %d: loss = %.2f metric = %.2f (%.3f sec)' % (step, loss_value, metric_value, duration))
                # Update the events file.
                summary_str = sess.run(summary, feed_dict=feed_dict)
                train_writer.add_summary(summary_str, step)
                train_writer.flush()

                checkpoint_file = os.path.join(FLAGS.log_dir, 'model')
                saver.save(sess, checkpoint_file, global_step=step)

                test_metric = _eval_metric(sess, metric, dropout, phase, x, y_cp, y_tim, test_mb_source)

                test_summary = tf.Summary(value=[tf.Summary.Value(tag="test_metric", simple_value=test_metric)])
                test_writer.add_summary(test_summary, step)
                test_writer.flush()

                print("--------------------")
                print('Test accuracy = %.2f' % test_metric)
                if (step + 1) % (EPOCH_SIZE // FLAGS.batch_size) == 0:
                    print("====================")
                else:
                    print("--------------------")

            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) % (EPOCH_SIZE // FLAGS.batch_size) == 0 or (step + 1) == MAX_STEPS:
                checkpoint_file = os.path.join(FLAGS.log_dir, 'model')
                saver.save(sess, checkpoint_file, global_step=step)

                test_metric = _eval_metric(sess, metric, dropout, phase, x, y_cp, y_tim, test_mb_source)

                test_summary = tf.Summary(value=[tf.Summary.Value(tag="test_metric", simple_value=test_metric)])
                test_writer.add_summary(test_summary, step)
                test_writer.flush()

                print("--------------------")
                print('Test accuracy = %.2f' % test_metric)
                if (step + 1) % (EPOCH_SIZE // FLAGS.batch_size) == 0:
                    print("====================")
                else:
                    print("--------------------")


def _eval_metric(sess, rmse, dropout, phase, x, y_cp, y_tim, test_mb_source):
    still_data = True
    metrics = []
    test_mb_source.restart()
    while still_data:
        test_feed_dict, still_data = _fill_feed_dict(test_mb_source, x, y_cp, y_tim, dropout, phase, False)
        if still_data is False:
            break
        metrics.append(sess.run(rmse, feed_dict=test_feed_dict))

    # print(metrics)

    metrics = np.concatenate(metrics, axis=0)

    return np.mean(metrics)


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
    G = create_graph(DATASET_FILE)
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

    train_mb_source, test_mb_source = create_train_test_mb_sources(DATASET_FILE, FLAGS.test_size, perm=perm)
    EPOCH_SIZE = train_mb_source.dataset_length

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
        default=5e-3,
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
        '--batch_size',
        type=int,
        default=20,
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
        default=[3, 3, 3, 3],
        nargs="+",
        help='Convolution vertex order.'
    )
    parser.add_argument(
        '--time_filter_orders',
        type=int,
        nargs="+",
        default=[5, 5, 3, 2],
        help='Convolution time order.'
    )
    parser.add_argument(
        '--num_filters',
        type=int,
        nargs="+",
        default=[10, 20, 30, 40],
        help='Number of parallel convolutional filters.'
    )
    parser.add_argument(
        '--time_poolings',
        type=int,
        nargs="+",
        default=[4, 4, 3, 2],
        help='Time pooling sizes.'
    )
    parser.add_argument(
        "--vertex_poolings",
        type=int,
        nargs="+",
        default=[4, 4, 4, 2],
        help="Vertex pooling sizes"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Percentage left for test"
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
