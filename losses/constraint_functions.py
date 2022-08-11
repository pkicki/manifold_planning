import tensorflow as tf

from losses.utils import huber
from utils.table import Table

table = Table()


def air_hockey_table(xyz, dt):
    xyz = xyz[..., 0]
    # v = tf.sqrt(tf.reduce_sum(tf.square(xyz[:, 1:] - xyz[:, :-1]), axis=-1))
    # v = v / (tf.reduce_sum(v, axis=1, keepdims=True) + 1e-8)
    # xyz = xyz[:, 1:]
    # huber_along_path = lambda x: tf.reduce_sum(v * huber(x), axis=-1)
    huber_along_path = lambda x: tf.reduce_sum(dt * huber(x), axis=-1)  # / tf.reduce_sum(dt, axis=-1)
    # huber_along_path = lambda x: tf.reduce_sum(dt * tf.square(x), axis=-1)
    # huber_along_path = lambda x: tf.reduce_mean(huber(x), axis=-1)
    relu_huber_along_path = lambda x: huber_along_path(tf.nn.relu(x))
    xlow_loss = relu_huber_along_path(table.xlb - xyz[..., 0])
    xhigh_loss = relu_huber_along_path(xyz[..., 0] - table.xrt)
    ylow_loss = relu_huber_along_path(table.ylb - xyz[..., 1])
    yhigh_loss = relu_huber_along_path(xyz[..., 1] - table.yrt)
    z_loss = huber_along_path(xyz[..., 2] - table.z)
    constraint_losses = tf.stack([xlow_loss, xhigh_loss, ylow_loss, yhigh_loss, z_loss], axis=-1)
    return constraint_losses


def air_hockey_puck(xyz, dt, puck_pose):
    xy = xyz[:, :, :2, 0]
    dist_from_puck = tf.sqrt(tf.reduce_sum((puck_pose[:, tf.newaxis] - xy) ** 2, axis=-1))
    puck_loss = tf.nn.relu(0.09 - dist_from_puck)
    idx_ = tf.argmin(puck_loss[..., ::-1], axis=-1)
    #a = 1000. / xyz.shape[1]
    # threshold = tf.math.sigmoid(-a * (tf.range(xyz.shape[1], dtype=tf.float32)[tf.newaxis] - tf.cast(xyz.shape[1] - idx_, tf.float32)[:, tf.newaxis]))
    # threshold_ = tf.where(threshold > 0.9, tf.ones_like(threshold), tf.zeros_like(threshold))

    # threshold_ = tf.where(, tf.zeros_like(threshold), tf.ones_like(threshold))
    # threshold_ = tf.cast(tf.cast(xyz.shape[1] - idx_, tf.float32)[:, tf.newaxis] - 1 > tf.range(xyz.shape[1], dtype=tf.float32)[tf.newaxis], tf.float32)
    idx = tf.cast(xyz.shape[1] - idx_, tf.float32)[:, tf.newaxis] - 1
    range = tf.range(xyz.shape[1], dtype=tf.float32)[tf.newaxis]
    threshold = tf.where(idx > range, tf.ones_like(puck_loss), tf.zeros_like(puck_loss))

    puck_loss = tf.reduce_sum(puck_loss * threshold * dt, axis=-1)
    return puck_loss
