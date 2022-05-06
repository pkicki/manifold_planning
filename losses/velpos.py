from math import pi
from time import time

import numpy as np

from learn_air_hockey.utils.bspline import BSpline
from learn_air_hockey.utils.constants import Limits
from learn_air_hockey.utils.manipulator import Iiwa
import tensorflow as tf
import pinocchio as pino

from learn_air_hockey.utils.table import Table


class LossVelPos:
    def __init__(self, urdf_path):
        self.man = Iiwa(urdf_path)
        self.table = Table()
        self.model = pino.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

    def __call__(self, qk, qdotk, data):
        x = data[:, 14]
        y = data[:, 15]
        thk = data[:, 16]
        dT = 0.01

        xyz = self.man.forward_kinematics(qk)[..., 0]
        xyzp1 = self.man.forward_kinematics(qk + qdotk * dT)[..., 0]
        xyz_dot = xyzp1 - xyz

        pos_loss = tf.reduce_sum(tf.square(xyz - np.stack([x, y, self.table.z * np.ones_like(x)], axis=-1)), axis=-1)
        pos_loss_abs = tf.reduce_sum(tf.abs(xyz - np.stack([x, y, self.table.z * np.ones_like(x)], axis=-1)), axis=-1)

        vz_loss = tf.reduce_mean(tf.square(xyz_dot[..., -1]), axis=-1)
        vz_loss_abs = tf.reduce_mean(tf.abs(xyz_dot[..., -1]), axis=-1)

        thk_pred = tf.atan2(xyz_dot[..., 1], xyz_dot[..., 0])

        direction_loss = tf.square(thk - thk_pred)
        direction_loss_abs = tf.abs(thk - thk_pred)

        v_loss = tf.linalg.norm(xyz_dot[..., :2] * np.cos(thk_pred - thk)[..., np.newaxis], axis=-1)

        model_loss = pos_loss + vz_loss / v_loss + direction_loss #- 1e-1 * v_loss
        return model_loss, pos_loss, pos_loss_abs, vz_loss, vz_loss_abs, direction_loss, direction_loss_abs, v_loss


class LossVelPosSup:
    def __init__(self):
        pass

    def __call__(self, qk, qdotk, data):
        qd = data[:, 3:9]
        qdotd = data[:, 10:16]

        q_loss = tf.reduce_sum(tf.square(qk - qd), axis=-1)
        q_loss_abs = tf.reduce_sum(tf.abs(qk - qd), axis=-1)

        qdot_loss = tf.reduce_sum(tf.square(qdotk - qdotd), axis=-1)
        qdot_loss_abs = tf.reduce_sum(tf.abs(qdotk - qdotd), axis=-1)

        model_loss = q_loss# + qdot_loss
        #model_loss = q_loss_abs# + qdot_loss_abs
        return model_loss, q_loss, q_loss_abs, qdot_loss, qdot_loss_abs
