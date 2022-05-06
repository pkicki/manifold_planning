import numpy as np
import tensorflow as tf
import pinocchio as pino

from utils.bspline import BSpline
from utils.data import unpack_data
from utils.manipulator import Iiwa


class Loss:
    def __init__(self, N, urdf_path, end_effector_constraint_distance_function, q_dot_limits, q_ddot_limits):
        self.bsp_t = BSpline(20)
        self.bsp = BSpline(N)
        self.man = Iiwa(urdf_path)
        self.end_effector_constraints_distance_function = end_effector_constraint_distance_function
        self.q_dot_limits = q_dot_limits
        self.q_ddot_limits = q_ddot_limits
        self.model = pino.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

    def __call__(self, q_cps, t_cps, data):
        q0, qd, xyth, q_dot_0, q_ddot_0, q_dot_d = unpack_data(data, 7)

        q = self.bsp.N @ q_cps
        q_dot_tau = self.bsp.dN @ q_cps
        q_ddot_tau = self.bsp.ddN @ q_cps

        dtau_dt = self.bsp_t.N @ t_cps
        ddtau_dtt = self.bsp_t.dN @ t_cps

        dt = 1. / dtau_dt[..., 0]
        t_cumsum = np.cumsum(dt, axis=-1)
        t = tf.reduce_mean(dt, axis=-1)

        q_dot = q_dot_tau * dtau_dt
        q_ddot = q_ddot_tau * dtau_dt ** 2 + ddtau_dtt * q_dot_tau * dtau_dt

        q_dot_limits = tf.constant(self.q_dot_limits)[tf.newaxis, tf.newaxis]
        q_ddot_limits = tf.constant(self.q_ddot_limits)[tf.newaxis, tf.newaxis]

        q_dot_loss_ = tf.reduce_sum(tf.nn.relu(tf.abs(q_dot) - q_dot_limits), axis=-1)
        q_dot_loss = tf.reduce_mean(q_dot_loss_, axis=-1)
        q_ddot_loss_ = tf.reduce_sum(tf.nn.relu(tf.abs(q_ddot) - q_ddot_limits), axis=-1)
        q_ddot_loss = tf.reduce_mean(q_ddot_loss_, axis=-1)

        xyz = self.man.forward_kinematics(q)

        constraint_loss = self.end_effector_constraints_distance_function(xyz)

        model_loss = constraint_loss + q_dot_loss + q_ddot_loss + t
        return model_loss, constraint_loss, q_dot_loss, q_ddot_loss, q, q_dot, q_ddot, xyz, t, t_cumsum
