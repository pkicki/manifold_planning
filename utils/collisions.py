import tensorflow as tf
import numpy as np


def inside_box(xyz, xl, xh, yl, yh, zl, zh):
    pxl = xyz[..., 0] > xl
    pxh = xyz[..., 0] < xh
    pyl = xyz[..., 1] > yl
    pyh = xyz[..., 1] < yh
    pzl = xyz[..., 2] > zl
    pzh = xyz[..., 2] < zh
    return tf.reduce_all(tf.stack([pxl, pxh, pyl, pyh, pzl, pzh], axis=-1), axis=-1)


def inside_rectangle(xyz, xl, xh, yl, yh):
    pxl = xyz[..., 0] > xl
    pxh = xyz[..., 0] < xh
    pyl = xyz[..., 1] > yl
    pyh = xyz[..., 1] < yh
    return tf.reduce_all(tf.stack([pxl, pxh, pyl, pyh], axis=-1), axis=-1)


def dist_point_2_box(xyz, xl, xh, yl, yh, zl, zh):
    def max_dist(v, l, h):
        return tf.reduce_max(tf.stack([l - v, tf.zeros_like(v), v - h], axis=-1), axis=-1)

    #with tf.GradientTape(persistent=True) as tape:
    #    tape.watch(xyz)
    l = np.reshape(np.stack([xl, yl, zl], axis=-1), (-1,) + (1,)*(len(xyz.shape) - 2) + (3,))
    h = np.reshape(np.stack([xh, yh, zh], axis=-1), (-1,) + (1,)*(len(xyz.shape) - 2) + (3,))
    xyz_dist = tf.reduce_max(tf.stack([l - xyz, tf.zeros_like(xyz), xyz - h], axis=-1), axis=-1)
    dist = tf.sqrt(tf.reduce_sum(tf.square(xyz_dist), axis=-1) + 1e-8)
        #x_dist = max_dist(xyz[..., 0], xl, xh)
        #y_dist = max_dist(xyz[..., 1], yl, yh)
        #z_dist = max_dist(xyz[..., 2], zl, zh)
        #dist = tf.sqrt(x_dist ** 2 + y_dist ** 2 + z_dist ** 2)

    #grad = tape.gradient(dist, xyz)
    #grad1 = tape.gradient(dist, [xyz, xyz_dist])
    #grad2 = tape.gradient(xyz_dist, [xyz])
    return dist


def dist_point_2_box_inside(xyz, xl, xh, yl, yh, zl, zh):
    dist = tf.reduce_min(tf.abs(tf.stack([xyz[..., 0] - xl, xyz[..., 0] - xh,
                                          xyz[..., 1] - yl, xyz[..., 1] - yh,
                                          xyz[..., 2] - zl, xyz[..., 2] - zh,
                                          ], axis=-1)), axis=-1)
    return dist


def collision_with_box(xyz, r, xl, xh, yl, yh, zl, zh):
    inside = inside_box(xyz, xl, xh, yl, yh, zl, zh)
    dist2box = dist_point_2_box(xyz, xl, xh, yl, yh, zl, zh)
    #dist2box_inside = dist_point_2_box_inside(xyz, xl, xh, yl, yh, zl, zh)
    dist2box = tf.nn.relu(r - dist2box)
    #collision = tf.where(inside, dist2box_inside, dist2box)
    collision = tf.where(inside, tf.zeros_like(dist2box), dist2box)
    #collision = dist2box
    return collision

def simple_collision_with_box(xyz, xl, xh, yl, yh, zl, zh):
    inside = inside_box(xyz, xl, xh, yl, yh, zl, zh)
    dist2box_inside = dist_point_2_box_inside(xyz, xl, xh, yl, yh, zl, zh)
    collision = tf.where(inside, dist2box_inside, tf.zeros_like(dist2box_inside))
    #collision = tf.where(inside, tf.zeros_like(dist2box), dist2box)
    #collision = dist2box
    return collision
