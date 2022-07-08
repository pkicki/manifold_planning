import numpy as np
import tensorflow as tf
import pinocchio as pino

from losses.utils import huber
from utils.bspline import BSpline
from utils.data import unpack_data_boundaries
from utils.manipulator import Iiwa


class FeasibilityLoss:
    def __init__(self, N, urdf_path, q_dot_limits, q_ddot_limits):
        self.bsp_t = BSpline(20)
        self.bsp = BSpline(N)
        self.q_dot_limits = q_dot_limits
        self.q_ddot_limits = q_ddot_limits
        self.model = pino.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

    def call(self, q_cps, t_cps, data):
        q = self.bsp.N @ q_cps
        q_dot_tau = self.bsp.dN @ q_cps
        q_ddot_tau = self.bsp.ddN @ q_cps

        dtau_dt = self.bsp_t.N @ t_cps
        ddtau_dtt = self.bsp_t.dN @ t_cps

        dt = 1. / dtau_dt[..., 0] / dtau_dt.shape[1]
        t_cumsum = np.cumsum(dt, axis=-1)
        t = tf.reduce_sum(dt, axis=-1)

        q_dot = q_dot_tau * dtau_dt
        q_ddot = q_ddot_tau * dtau_dt ** 2 + ddtau_dtt * q_dot_tau * dtau_dt

        q_dot_limits = tf.constant(self.q_dot_limits)[tf.newaxis, tf.newaxis]
        q_ddot_limits = tf.constant(self.q_ddot_limits)[tf.newaxis, tf.newaxis]

        #q_dot_loss_ = tf.reduce_sum(tf.nn.relu(tf.abs(q_dot) - q_dot_limits), axis=-1)
        #q_dot_loss = tf.reduce_mean(q_dot_loss_, axis=-1)
        #q_ddot_loss_ = tf.reduce_sum(tf.nn.relu(tf.abs(q_ddot) - q_ddot_limits), axis=-1)
        #q_ddot_loss = tf.reduce_mean(q_ddot_loss_, axis=-1)
        q_dot_loss_ = tf.nn.relu(tf.abs(q_dot) - q_dot_limits)
        q_dot_loss_ = huber(q_dot_loss_)
        #q_dot_loss_ = tf.square(q_dot_loss_)
        q_dot_loss = tf.reduce_sum(q_dot_loss_ * dt[..., tf.newaxis], axis=1)# / t[..., tf.newaxis]
        #q_dot_loss = tf.reduce_mean(q_dot_loss_, axis=1)
        q_ddot_loss_ = tf.nn.relu(tf.abs(q_ddot) - q_ddot_limits)
        q_ddot_loss_ = huber(q_ddot_loss_)
        #q_ddot_loss_ = tf.square(q_ddot_loss_)
        q_ddot_loss = tf.reduce_sum(q_ddot_loss_ * dt[..., tf.newaxis], axis=1)# / t[..., tf.newaxis]
        #q_ddot_loss = tf.reduce_mean(q_ddot_loss_, axis=1)
        model_losses = tf.concat([q_dot_loss, q_ddot_loss], axis=-1)
        model_loss = tf.reduce_sum(model_losses, axis=-1)
        return model_loss, q_dot_loss, q_ddot_loss, q, q_dot, q_ddot, t, t_cumsum, dt

    def __call__(self, q_cps, t_cps, data):
        return self.call(q_cps, t_cps, data)
