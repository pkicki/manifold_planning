from time import perf_counter

from scipy.sparse.linalg import cg

from losses.feasibility import FeasibilityLoss
from losses.utils import huber
from utils.manipulator import Iiwa
import tensorflow as tf
import numpy as np


class HittingLoss(FeasibilityLoss):
    def __init__(self, N, urdf_path, end_effector_constraint_distance_function, q_dot_limits, q_ddot_limits):
        super(HittingLoss, self).__init__(N, urdf_path, q_dot_limits, q_ddot_limits)
        self.end_effector_constraints_distance_function = end_effector_constraint_distance_function
        self.man = Iiwa(urdf_path)
        self.alpha_constraint = tf.math.log(1e-2)
        self.alpha_q_dot = tf.math.log(1e-2)
        self.alpha_q_ddot = tf.math.log(1e-2)
        self.alpha_torque = tf.math.log(1e-2)
        self.lambdas = np.array([1., 1., 1.])
        self.gamma = 1e-2
        self.bar_constraint = 5e-6
        self.bar_q_dot = 6e-3
        self.bar_q_ddot = 6e-2
        self.bar_torque = 6e-2
        self.jerk_mul = 1e-4

    def call(self, q_cps, t_cps, data):
        _, q_dot_loss, q_ddot_loss, torque_loss, q, q_dot, q_ddot, torque, t, t_cumsum, dt = super().call(q_cps, t_cps, data)

        xyz = self.man.forward_kinematics(q)
        constraint_loss = self.end_effector_constraints_distance_function(xyz, dt)
        t_loss = huber(t[:, tf.newaxis])
        #t_loss = tf.square(t[:, tf.newaxis])
        jerk_loss = tf.reduce_sum(tf.abs(q_ddot[:, 1:] - q_ddot[:, :-1]) * dt[..., 1:, tf.newaxis],
                                  axis=(1, 2))[:, tf.newaxis]
        losses = tf.concat([tf.exp(self.alpha_q_dot) * q_dot_loss,
                            tf.exp(self.alpha_q_ddot) * q_ddot_loss,
                            tf.exp(self.alpha_torque) * torque_loss,
                            tf.exp(self.alpha_constraint) * constraint_loss,
                            t_loss, self.jerk_mul * jerk_loss], axis=-1)
        unscaled_losses = tf.concat([q_dot_loss, q_ddot_loss, torque_loss, constraint_loss, t_loss, jerk_loss], axis=-1)
        sum_q_dot_loss = tf.reduce_sum(q_dot_loss, axis=-1)
        sum_q_ddot_loss = tf.reduce_sum(q_ddot_loss, axis=-1)
        sum_constraint_loss = tf.reduce_sum(constraint_loss, axis=-1)
        sum_torque_loss = tf.reduce_sum(torque_loss, axis=-1)

        model_loss = tf.reduce_sum(losses, axis=-1)
        unscaled_model_loss = tf.reduce_sum(unscaled_losses, axis=-1)
        #print("MLOSS:", model_loss)
        return model_loss, sum_constraint_loss, sum_q_dot_loss, sum_q_ddot_loss, sum_torque_loss, q, q_dot, q_ddot, torque, xyz, t, t_cumsum, t_loss, dt, unscaled_model_loss, jerk_loss

    def alpha_update(self, q_dot_loss, q_ddot_loss, constraint_loss, torque_loss):
        max_alpha_update = 10.0
        alpha_q_dot_update = self.gamma * tf.clip_by_value(tf.math.log(q_dot_loss / self.bar_q_dot), -max_alpha_update, max_alpha_update)
        alpha_q_ddot_update = self.gamma * tf.clip_by_value(tf.math.log(q_ddot_loss / self.bar_q_ddot), -max_alpha_update, max_alpha_update)
        alpha_constraint_update = self.gamma * tf.clip_by_value(tf.math.log(constraint_loss / self.bar_constraint), -max_alpha_update, max_alpha_update)
        alpha_torque_update = self.gamma * tf.clip_by_value(tf.math.log(torque_loss / self.bar_torque), -max_alpha_update, max_alpha_update)
        self.alpha_q_dot += alpha_q_dot_update
        self.alpha_q_ddot += alpha_q_ddot_update
        self.alpha_constraint += alpha_constraint_update
        self.alpha_torque += alpha_torque_update
