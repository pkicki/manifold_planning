import tensorflow as tf

from losses.utils import huber
from utils.table import Table
table = Table()


def air_hockey_table(xyz):
    xyz = xyz[..., 0]
    #v = tf.linalg.norm(xyz[:, 1:] - xyz[:, :-1], axis=-1)
    v = tf.sqrt(tf.reduce_sum(tf.square(xyz[:, 1:] - xyz[:, :-1]), axis=-1))
    v = v / tf.reduce_sum(v, axis=1, keepdims=True)
    xyz = xyz[:, 1:]
    huber_along_path = lambda x: tf.reduce_sum(v * huber(x), axis=-1)
    relu_huber_along_path = lambda x: huber_along_path(tf.nn.relu(x))
    xlow_loss = relu_huber_along_path(table.xlb - xyz[..., 0])
    xhigh_loss = relu_huber_along_path(xyz[..., 0] - table.xrt)
    ylow_loss = relu_huber_along_path(table.ylb - xyz[..., 1])
    yhigh_loss = relu_huber_along_path(xyz[..., 1] - table.yrt)
    z_loss = huber_along_path(xyz[..., 2] - table.z)
    constraint_losses = tf.stack([xlow_loss, xhigh_loss, ylow_loss, yhigh_loss, z_loss], axis=-1)
    return constraint_losses
