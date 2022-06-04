import tensorflow as tf
from utils.table import Table
table = Table()


def air_hockey_table(xyz):
    xlow_loss = tf.nn.relu(table.xlb - xyz[..., 0, 0])
    xhigh_loss = tf.nn.relu(xyz[..., 0, 0] - table.xrt)
    ylow_loss = tf.nn.relu(table.ylb - xyz[..., 1, 0])
    yhigh_loss = tf.nn.relu(xyz[..., 1, 0] - table.yrt)
    z_loss = tf.square(xyz[..., 2, 0] - table.z) * 1e2
    constraint_loss = tf.reduce_mean(xlow_loss + xhigh_loss + ylow_loss + yhigh_loss + z_loss, axis=-1)
    return constraint_loss
