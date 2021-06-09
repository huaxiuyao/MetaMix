from __future__ import print_function

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from absl import flags
from tensorflow.contrib import layers as contrib_layers
from tensorflow.contrib.layers.python import layers as tf_layers


FLAGS = flags.FLAGS

def conv_block(x, weight, bias, reuse, scope):
    x = tf.nn.conv2d(x, weight, [1, 1, 1, 1], 'SAME') + bias
    x = tf_layers.batch_norm(
        x, activation_fn=tf.nn.relu, reuse=reuse, scope=scope)
    return x


def mixup_data(x_support, y_support, x_query, y_query):
    dist = tfp.distributions.Beta(0.5, 0.5)
    lam = dist.sample([1])
    lam_x = lam
    lam_y = lam
    index = tf.range(FLAGS.update_batch_size)
    shuffled_index = tf.random.shuffle(index)
    mixed_x = lam_x * x_query + (1 - lam_x) * tf.gather(x_support, shuffled_index)
    reweighted_target = lam_y * y_query + (1 - lam_y) * tf.gather(y_support, shuffled_index)
    return mixed_x, reweighted_target


def mse(pred, label):
    pred = tf.reshape(pred, [-1])
    label = tf.reshape(label, [-1])
    return tf.reduce_mean(tf.square(pred - label))


class MAML(object):
    def __init__(self, encoder_w, dim_input=1, dim_output=1):
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.update_lr = FLAGS.update_lr
        self.meta_lr = tf.placeholder_with_default(FLAGS.meta_lr, ())

        self.loss_func = mse
        self.encoder_w = encoder_w

        self.dim_hidden = FLAGS.num_filters
        self.forward = self.forward_conv
        self.construct_weights = self.construct_conv_weights

        self.channels = 1
        self.img_size = int(np.sqrt(self.dim_input / self.channels))

    def construct_model(self,
                        input_tensors=None,
                        prefix='metatrain_',
                        test_num_updates=0):

        self.inputa = input_tensors['inputa']
        self.inputb = input_tensors['inputb']
        self.labela = input_tensors['labela']
        self.labelb = input_tensors['labelb']

        with tf.variable_scope('model', reuse=None) as training_scope:
            if 'weights' in dir(self):
                training_scope.reuse_variables()
                weights = self.weights
            else:
                self.weights = weights = self.construct_weights()

            num_updates = max(test_num_updates, FLAGS.num_updates)

            def task_metalearn(inp, reuse=True):
                TRAIN = 'train' in prefix
                inputa, inputb, labela, labelb = inp
                task_outputbs, task_lossesb, task_lossesa = [], [], []
                task_msesb = []

                task_outputa = self.forward(
                    inputa, weights, reuse=reuse)
                task_lossa = self.loss_func(task_outputa, labela)
                task_lossesa.append(task_lossa)

                grads = tf.gradients(task_lossa, list(weights.values()))
                if FLAGS.stop_grad:
                    grads = [tf.stop_gradient(grad) for grad in grads]
                gradients = dict(zip(weights.keys(), grads))
                fast_weights = dict(
                    zip(weights.keys(), [
                        weights[key] - self.update_lr * gradients[key]
                        for key in weights.keys()
                    ]))

                if TRAIN and FLAGS.mix:
                    output, reweight_labelb = self.forward_conv_mixup(inputa, inputb, labela, labelb, weights,
                                                                      reuse=True)
                    task_outputbs.append(output)
                    task_msesb.append(self.loss_func(output, reweight_labelb))
                    task_lossesb.append(
                        self.loss_func(output, reweight_labelb))
                else:
                    output = self.forward(inputb, weights, reuse=True)
                    task_outputbs.append(output)
                    task_msesb.append(self.loss_func(output, labelb))
                    task_lossesb.append(
                        self.loss_func(output, labelb))

                for j in range(num_updates - 1):
                    loss = self.loss_func(
                        self.forward(inputa, fast_weights, reuse=True), labela)
                    grads = tf.gradients(loss, list(fast_weights.values()))
                    if FLAGS.stop_grad:
                        grads = [tf.stop_gradient(grad) for grad in grads]
                    gradients = dict(zip(fast_weights.keys(), grads))
                    fast_weights = dict(zip(fast_weights.keys(),
                                            [fast_weights[key] - self.update_lr * gradients[key] for key in
                                             fast_weights.keys()]))

                if TRAIN and FLAGS.mix:
                    output, reweight_labelb = self.forward_conv_mixup(inputa, inputb, labela, labelb, fast_weights,
                                                                      reuse=True)
                    task_outputbs.append(output)
                    task_msesb.append(self.loss_func(output, reweight_labelb))
                    task_lossesb.append(
                        self.loss_func(output, reweight_labelb))
                else:
                    output = self.forward(inputb, fast_weights, reuse=True)
                    task_outputbs.append(output)
                    task_msesb.append(self.loss_func(output, labelb))
                    task_lossesb.append(
                        self.loss_func(output, labelb))

                outputa_after = self.forward(inputa, fast_weights, reuse=True)
                task_lossesa.append(self.loss_func(outputa_after, labela))

                task_output = [
                    task_outputa, task_outputbs, task_lossesa, task_lossesb, task_msesb
                ]

                return task_output

            if FLAGS.norm is not None:
                _ = task_metalearn(
                    (self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0]),
                    False)

            out_dtype = [
                tf.float32, [tf.float32] * 2, [tf.float32] * 2, [tf.float32] * 2,
                            [tf.float32] * 2
            ]
            result = tf.map_fn(task_metalearn, elems=(self.inputa, self.inputb, \
                                                      self.labela, self.labelb), dtype=out_dtype, \
                               parallel_iterations=FLAGS.meta_batch_size)
            outputas, outputbs, lossesa, lossesb, msesb = result

        ## Performance & Optimization
        if 'train' in prefix:
            self.total_loss1 = total_loss1 = [
                tf.reduce_sum(lossesa[j]) / tf.to_float(FLAGS.meta_batch_size)
                for j in range(len(lossesa))
            ]
            self.total_losses2 = total_losses2 = [
                tf.reduce_sum(msesb[j]) / tf.to_float(FLAGS.meta_batch_size)
                for j in range(len(msesb))
            ]
            self.total_losses3 = total_losses3 = [
                tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size)
                for j in range(len(lossesb))
            ]
            # after the map_fn
            self.outputas, self.outputbs = outputas, outputbs

            # OUTER LOOP
            if FLAGS.metatrain_iterations > 0:
                optimizer = tf.train.AdamOptimizer(self.meta_lr)
                THETA = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model')  # pylint: disable=invalid-name
                PHI = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder')  # pylint: disable=invalid-name

                self.gvs_theta = gvs_theta = optimizer.compute_gradients(
                    self.total_losses2[-1], THETA)
                metatrain_theta_op = optimizer.apply_gradients(gvs_theta)

                self.gvs_phi = gvs_phi = optimizer.compute_gradients(
                    self.total_losses3[-1], PHI)
                metatrain_phi_op = optimizer.apply_gradients(gvs_phi)

                with tf.control_dependencies([metatrain_theta_op, metatrain_phi_op]):
                    self.metatrain_op = tf.no_op()

                scale_v = [
                    v for v in self.encoder_w.trainable_variables if 'scale' in v.name
                ]
                scale_norm = [tf.reduce_mean(v) for v in scale_v]
                scale_norm = tf.reduce_mean(scale_norm)

                tf.summary.scalar(prefix + 'full_loss', total_losses3[-1])
                tf.summary.scalar(prefix + 'regularizer',
                                  total_losses3[-1] - total_losses2[-1])
                tf.summary.scalar(prefix + 'untransformed_scale', scale_norm)

        else:
            self.metaval_total_loss1 = total_loss1 = [
                tf.reduce_sum(lossesa[j]) / tf.to_float(FLAGS.meta_batch_size)
                for j in range(len(lossesa))
            ]
            self.metaval_total_losses2 = total_losses2 = [
                tf.reduce_sum(msesb[j]) / tf.to_float(FLAGS.meta_batch_size)
                for j in range(len(msesb))
            ]
            self.metaval_total_losses3 = total_losses3 = [
                tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size)
                for j in range(len(lossesb))
            ]

        tf.summary.scalar(prefix + 'Pre-mse', total_losses2[0])
        tf.summary.scalar(prefix + 'Post-mse_' + str(num_updates),
                          total_losses2[-1])

    def construct_conv_weights(self):
        """Construct conv weights."""
        weights = {}

        dtype = tf.float32
        conv_initializer = contrib_layers.xavier_initializer_conv2d(dtype=dtype)
        k = 3

        weights['conv1'] = tf.get_variable(
            'conv1', [k, k, self.channels, self.dim_hidden],
            initializer=conv_initializer,
            dtype=dtype)
        weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv2'] = tf.get_variable(
            'conv2', [k, k, self.dim_hidden, self.dim_hidden],
            initializer=conv_initializer,
            dtype=dtype)
        weights['b2'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv3'] = tf.get_variable(
            'conv3', [k, k, self.dim_hidden, self.dim_hidden],
            initializer=conv_initializer,
            dtype=dtype)
        weights['b3'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv4'] = tf.get_variable(
            'conv4', [k, k, self.dim_hidden, self.dim_hidden],
            initializer=conv_initializer,
            dtype=dtype)
        weights['b4'] = tf.Variable(tf.zeros([self.dim_hidden]))

        weights['w5'] = tf.Variable(
            tf.random_normal([self.dim_hidden, self.dim_output]), name='w5')
        weights['b5'] = tf.Variable(tf.zeros([self.dim_output]), name='b5')
        return weights

    def forward_conv(self, inp, weights, reuse=False, scope=''):
        channels = self.channels
        inp = tf.reshape(inp, [-1, self.img_size, self.img_size, channels])

        hidden1 = conv_block(inp, weights['conv1'], weights['b1'], reuse,
                             scope + '0')
        hidden2 = conv_block(hidden1, weights['conv2'], weights['b2'], reuse,
                             scope + '1')
        hidden3 = conv_block(hidden2, weights['conv3'], weights['b3'], reuse,
                             scope + '2')
        hidden4 = conv_block(hidden3, weights['conv4'], weights['b4'], reuse,
                             scope + '3')

        hidden4 = tf.reduce_mean(hidden4, [1, 2])

        return tf.matmul(hidden4, weights['w5']) + weights['b5']

    def forward_conv_mixup(self, inp_support, inp_query, label_support, label_query, weights, reuse=False, scope=''):
        sel_layer =  tf.random.uniform(shape=(), minval=0, maxval=4, dtype=tf.int32)
        reweighted_target = label_query
        mixed_inp = inp_query

        mixed_inp, reweighted_target = tf.cond(tf.equal(sel_layer, 0),
                                               lambda: mixup_data(inp_support, label_support, inp_query, label_query),
                                               lambda: (mixed_inp, reweighted_target))

        channels = self.channels
        mixed_inp = tf.reshape(mixed_inp, [-1, self.img_size, self.img_size, channels])

        inp_support = tf.reshape(inp_support, [-1, self.img_size, self.img_size, channels])

        hidden1_support = conv_block(inp_support, weights['conv1'], weights['b1'], reuse,
                                     scope + '0')
        hidden1_query = conv_block(mixed_inp, weights['conv1'], weights['b1'], reuse,
                                   scope + '0')

        hidden1_query, reweighted_target = tf.cond(tf.equal(sel_layer, 1),
                                                   lambda: mixup_data(hidden1_support, label_support, hidden1_query,
                                                                      label_query),
                                                   lambda: (hidden1_query, reweighted_target))

        hidden2_support = conv_block(hidden1_support, weights['conv2'], weights['b2'], reuse,
                                     scope + '1')
        hidden2_query = conv_block(hidden1_query, weights['conv2'], weights['b2'], reuse,
                                   scope + '1')

        hidden2_query, reweighted_target = tf.cond(tf.equal(sel_layer, 2),
                                                   lambda: mixup_data(hidden2_support, label_support, hidden2_query,
                                                                      label_query),
                                                   lambda: (hidden2_query, reweighted_target))

        hidden3_support = conv_block(hidden2_support, weights['conv3'], weights['b3'], reuse,
                                     scope + '2')
        hidden3_query = conv_block(hidden2_query, weights['conv3'], weights['b3'], reuse,
                                   scope + '2')

        hidden3_query, reweighted_target = tf.cond(tf.equal(sel_layer, 3),
                                                   lambda: mixup_data(hidden3_support, label_support, hidden3_query,
                                                                      label_query),
                                                   lambda: (hidden3_query, reweighted_target))

        hidden4 = conv_block(hidden3_query, weights['conv4'], weights['b4'], reuse,
                             scope + '3')

        hidden4 = tf.reduce_mean(hidden4, [1, 2])

        return tf.matmul(hidden4, weights['w5']) + weights['b5'], reweighted_target
