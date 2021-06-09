from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import pickle
import random
import time

import numpy as np
import tensorflow as tf
from absl import app
from absl import flags
from six.moves import range
from data_generator import get_batch

from maml import MAML

FLAGS = flags.FLAGS

## Dataset/method options
flags.DEFINE_string('datasource', 'pose',
                    'sinusoid or omniglot or miniimagenet')
flags.DEFINE_integer('dim_w', 196, 'dimension of w')
flags.DEFINE_integer('dim_im', 128, 'dimension of image')
flags.DEFINE_integer('dim_y', 1, 'dimension of w')
flags.DEFINE_string('data_dir', None,
                    'Directory of data files.')
get_data_dir = lambda: FLAGS.data_dir
flags.DEFINE_list('data', ['train_data_ins.pkl', 'val_data_ins.pkl'],
                  'data name')

## Training options
flags.DEFINE_integer(
    'num_classes', 1,
    'number of classes used in classification (e.g. 5-way classification).')
flags.DEFINE_integer(
    'update_batch_size', 15,
    'number of examples used for inner gradient update (K for K-shot learning).'
)
flags.DEFINE_integer('num_updates', 5,
                     'number of inner gradient updates during training.')
flags.DEFINE_integer('meta_batch_size', 10,
                     'number of tasks sampled per meta-update')
flags.DEFINE_integer('test_num_updates', 20,
                     'number of inner gradient updates during test.')

flags.DEFINE_integer(
    'num_filters', 64,
    'number of filters for conv nets -- 32 for miniimagenet, 64 for omiglot.')
flags.DEFINE_float('meta_lr', 0.002, 'the base learning rate of the generator')
flags.DEFINE_float(
    'update_lr', 0.002,
    'step size alpha for inner gradient update.')  # 0.1 for omniglot
flags.DEFINE_bool(
    'mix', False,
    'value of mix_alpha.')  # 15k for omniglot, 50k for sinusoid
flags.DEFINE_integer(
    'metatrain_iterations', 30000,
    'number of metatraining iterations.')  # 15k for omniglot, 50k for sinusoid

flags.DEFINE_bool(
    'stop_grad', False,
    'if True, do not use second derivatives in meta-optimization (for speed)')
flags.DEFINE_string('norm', 'batch_norm', 'batch_norm, layer_norm, or None')

## Logging, saving, and testing options
flags.DEFINE_string('logdir', 'xxx',
                    'directory for summaries and checkpoints.')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_integer('test_epoch', -1, 'test load epoch')
flags.DEFINE_integer('trial', 1, 'trial_num')


def train(model, sess, exp_name, saver):
    """Train model."""
    print('Done initializing, starting training.')

    old_time = time.time()
    PRINT_INTERVAL = 100  # pylint: disable=invalid-name
    TEST_PRINT_INTERVAL = 100  # pylint: disable=invalid-name
    SAVE_INTERVAL = 500
    prelosses_supp, postlosses_supp = [], []
    prelosses_query, postlosses_query = [], []

    tf.global_variables_initializer().run()

    for itr in range(FLAGS.metatrain_iterations):
        feed_dict = {}
        input_tensors = [model.metatrain_op] + model.total_loss1 + model.total_losses2

        result = sess.run(input_tensors, feed_dict)

        prelosses_supp.append(result[1])
        postlosses_supp.append(result[2])
        prelosses_query.append(result[3])
        postlosses_query.append(result[4])

        if (itr != 0) and itr % PRINT_INTERVAL == 0:
            print_str = 'Iteration ' + str(itr)
            print_str += ': ' + str(np.mean(prelosses_supp)) + ', ' + str(
                np.mean(postlosses_supp)) + ', ' + str(
                np.mean(prelosses_query)) + ', ' + str(
                np.mean(postlosses_query))
            print(print_str, 'time =', time.time() - old_time)

            prelosses_supp, postlosses_supp = [], []
            prelosses_query, postlosses_query = [], []
            old_time = time.time()

        if (itr != 0) and itr % SAVE_INTERVAL == 0:
            saver.save(sess, FLAGS.logdir + '/' + exp_name + '/model' + str(itr))


def test(model, sess):
    """Test model."""
    np.random.seed(1)
    random.seed(1)
    NUM_TEST_POINTS = 1000
    metaval_accuracies = []

    for test_idx in range(NUM_TEST_POINTS):
        feed_dict = {model.meta_lr: 0.0}
        result = sess.run(model.metaval_total_losses2[-1], feed_dict)
        metaval_accuracies.append(result)

    metaval_accuracies = np.array(metaval_accuracies)
    means = np.mean(metaval_accuracies, 0)
    stds = np.std(metaval_accuracies, 0)
    ci95 = 1.96 * stds / np.sqrt(NUM_TEST_POINTS)

    print('Mean validation accuracy/loss, and confidence intervals')
    print((means, ci95))


def gen(x, y):
    while True:
        yield get_batch(np.array(x), np.array(y))


def main(_):
    dim_output = FLAGS.dim_y
    dim_input = FLAGS.dim_im * FLAGS.dim_im * 1

    exp_name = f'MetaMix.data-{FLAGS.datasource}.ubs-{FLAGS.update_batch_size}.meta_lr-{FLAGS.meta_lr}.update_lr-{FLAGS.update_lr}.num_updates-{FLAGS.num_updates}.trial-{FLAGS.trial}'

    if FLAGS.mix:
        exp_name += '.mix'

    x_train, y_train = pickle.load(
        open(os.path.join(get_data_dir(), FLAGS.data[0]), 'rb'))
    x_val, y_val = pickle.load(
        open(os.path.join(get_data_dir(), FLAGS.data[1]), 'rb'))

    x_train, y_train = np.array(x_train), np.array(y_train)
    y_train = y_train[:, :, -1, None]
    x_val, y_val = np.array(x_val), np.array(y_val)
    y_val = y_val[:, :, -1, None]

    if FLAGS.train == False:
        FLAGS.meta_batch_size = 1

    ds_train = tf.data.Dataset.from_generator(
        functools.partial(gen, x_train, y_train),
        (tf.float32, tf.float32, tf.float32, tf.float32),
        (tf.TensorShape(
            [None, FLAGS.update_batch_size * FLAGS.num_classes, dim_input]),
         tf.TensorShape(
             [None, FLAGS.update_batch_size * FLAGS.num_classes, dim_output]),
         tf.TensorShape(
             [None, FLAGS.update_batch_size * FLAGS.num_classes, dim_input]),
         tf.TensorShape(
             [None, FLAGS.update_batch_size * FLAGS.num_classes, dim_output])))

    ds_val = tf.data.Dataset.from_generator(
        functools.partial(gen, x_val, y_val),
        (tf.float32, tf.float32, tf.float32, tf.float32),
        (tf.TensorShape(
            [None, FLAGS.update_batch_size * FLAGS.num_classes, dim_input]),
         tf.TensorShape(
             [None, FLAGS.update_batch_size * FLAGS.num_classes, dim_output]),
         tf.TensorShape(
             [None, FLAGS.update_batch_size * FLAGS.num_classes, dim_input]),
         tf.TensorShape(
             [None, FLAGS.update_batch_size * FLAGS.num_classes, dim_output])))

    encoder_w = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu', padding='SAME'),

        tf.keras.layers.Conv2D(filters=48, kernel_size=3, strides=(2, 2), activation='relu', padding='SAME'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu', padding='SAME'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(FLAGS.dim_w)
    ])

    xa, labela, xb, labelb = ds_train.make_one_shot_iterator().get_next()
    xa = tf.reshape(xa, [-1, 128, 128, 1])
    xb = tf.reshape(xb, [-1, 128, 128, 1])
    with tf.variable_scope('encoder'):
        inputa = encoder_w(xa)
    inputa = tf.reshape(
        inputa, [-1, FLAGS.update_batch_size * FLAGS.num_classes, FLAGS.dim_w])
    inputb = encoder_w(xb)
    inputb = tf.reshape(
        inputb, [-1, FLAGS.update_batch_size * FLAGS.num_classes, FLAGS.dim_w])

    input_tensors = {'inputa': inputa, \
                     'inputb': inputb, \
                     'labela': labela, 'labelb': labelb}
    # n_task * n_im_per_task * dim_w
    xa_val, labela_val, xb_val, labelb_val = ds_val.make_one_shot_iterator(
    ).get_next()
    xa_val = tf.reshape(xa_val, [-1, 128, 128, 1])
    xb_val = tf.reshape(xb_val, [-1, 128, 128, 1])

    inputa_val = encoder_w(xa_val)
    inputa_val = tf.reshape(
        inputa_val,
        [-1, FLAGS.update_batch_size * FLAGS.num_classes, FLAGS.dim_w])

    inputb_val = encoder_w(xb_val)
    inputb_val = tf.reshape(
        inputb_val,
        [-1, FLAGS.update_batch_size * FLAGS.num_classes, FLAGS.dim_w])

    metaval_input_tensors = {'inputa': inputa_val, \
                             'inputb': inputb_val, \
                             'labela': labela_val, 'labelb': labelb_val}

    model = MAML(encoder_w, FLAGS.dim_w, dim_output)
    if FLAGS.train:
        model.construct_model(input_tensors=input_tensors, prefix='metatrain_')
    else:
        model.construct_model(
            input_tensors=metaval_input_tensors,
            prefix='metaval_',
            test_num_updates=FLAGS.test_num_updates)

    model.summ_op = tf.summary.merge_all()
    sess = tf.InteractiveSession()

    tf.global_variables_initializer().run()

    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=60)

    if FLAGS.train:
        train(model, sess, exp_name, saver)
    else:
        model_file = '{0}/{2}/model{1}'.format(FLAGS.logdir, FLAGS.test_epoch, exp_name)
        saver.restore(sess, model_file)
        test(model, sess)


if __name__ == '__main__':
    app.run(main)
