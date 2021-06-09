from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
from absl import flags
from six.moves import range


FLAGS = flags.FLAGS


def get_batch(x, y):
    """Get data batch."""
    xs, ys, xq, yq = [], [], [], []
    for _ in range(FLAGS.meta_batch_size):
        classes = np.random.choice(
            list(range(np.shape(x)[0])), size=FLAGS.num_classes, replace=False)

        support_set = []
        query_set = []
        support_sety = []
        query_sety = []
        for k in list(classes):
            idx = np.random.choice(
                list(range(np.shape(x)[1])),
                size=FLAGS.update_batch_size + FLAGS.update_batch_size,
                replace=False)
            x_k = x[k][idx]
            y_k = y[k][idx]

            support_set.append(x_k[:FLAGS.update_batch_size])
            query_set.append(x_k[FLAGS.update_batch_size:])
            support_sety.append(y_k[:FLAGS.update_batch_size])
            query_sety.append(y_k[FLAGS.update_batch_size:])

        xs_k = np.concatenate(support_set, 0)
        xq_k = np.concatenate(query_set, 0)
        ys_k = np.concatenate(support_sety, 0)
        yq_k = np.concatenate(query_sety, 0)

        xs.append(xs_k)
        xq.append(xq_k)
        ys.append(ys_k)
        yq.append(yq_k)

    xs, ys = np.stack(xs, 0), np.stack(ys, 0)
    xq, yq = np.stack(xq, 0), np.stack(yq, 0)

    xs = np.reshape(
        xs,
        [FLAGS.meta_batch_size, FLAGS.update_batch_size * FLAGS.num_classes, -1])
    xq = np.reshape(
        xq,
        [FLAGS.meta_batch_size, FLAGS.update_batch_size * FLAGS.num_classes, -1])
    xs = xs.astype(np.float32) / 255.0
    xq = xq.astype(np.float32) / 255.0
    ys = ys.astype(np.float32) * 10.0
    yq = yq.astype(np.float32) * 10.0
    return xs, ys, xq, yq