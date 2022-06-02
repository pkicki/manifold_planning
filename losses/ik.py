import numpy as np
import tensorflow as tf

from utils.manipulator import Iiwa
from utils.table import Table


class IKHittingPosLoss:
    def __init__(self, urdf_path):
        self.man = Iiwa(urdf_path)
        self.table = Table()

    def __call__(self, qk, data):
        x = data[:, 0]
        y = data[:, 1]
        qd = data[:, 3:9]

        xyz = self.man.forward_kinematics(qk)[..., 0]

        pos_loss = tf.reduce_sum(tf.square(xyz - np.stack([x, y, self.table.z * np.ones_like(x)], axis=-1)), axis=-1)
        pos_loss_abs = tf.reduce_sum(tf.abs(xyz - np.stack([x, y, self.table.z * np.ones_like(x)], axis=-1)), axis=-1)

        q_loss = tf.reduce_sum(tf.square(qk - qd), axis=-1)
        q_loss_abs = tf.reduce_sum(tf.abs(qk - qd), axis=-1)

        model_loss = pos_loss + q_loss
        return model_loss, pos_loss, pos_loss_abs, q_loss, q_loss_abs


class IKHittingLoss:
    def __init__(self):
        pass

    def __call__(self, qk, data):
        qd = data[:, 3:9]

        q_loss = tf.reduce_sum(tf.square(qk - qd), axis=-1)
        q_loss_abs = tf.reduce_sum(tf.abs(qk - qd), axis=-1)

        model_loss = q_loss
        return model_loss, q_loss, q_loss_abs
