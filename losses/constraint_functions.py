import tensorflow as tf

from losses.utils import huber
from utils.table import Table
table = Table()


def air_hockey_table(xyz):
    huber_along_path = lambda x: tf.reduce_mean(huber(x), axis=-1)
    relu_huber_along_path = lambda x: huber_along_path(tf.nn.relu(x))
    xlow_loss = relu_huber_along_path(table.xlb - xyz[..., 0, 0])
    xhigh_loss = relu_huber_along_path(xyz[..., 0, 0] - table.xrt)
    ylow_loss = relu_huber_along_path(table.ylb - xyz[..., 1, 0])
    yhigh_loss = relu_huber_along_path(xyz[..., 1, 0] - table.yrt)
    z_loss = huber_along_path(xyz[..., 2, 0] - table.z)
    constraint_losses = tf.stack([xlow_loss, xhigh_loss, ylow_loss, yhigh_loss, z_loss], axis=-1)
    return constraint_losses
