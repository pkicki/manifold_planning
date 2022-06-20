from losses.feasibility import FeasibilityLoss
from losses.utils import huber
from utils.manipulator import Iiwa
import tensorflow as tf


class HittingLoss(FeasibilityLoss):
    def __init__(self, N, urdf_path, end_effector_constraint_distance_function, q_dot_limits, q_ddot_limits):
        super(HittingLoss, self).__init__(N, urdf_path, q_dot_limits, q_ddot_limits)
        self.end_effector_constraints_distance_function = end_effector_constraint_distance_function
        self.man = Iiwa(urdf_path)
        self.alpha_constraint = 0.
        self.alpha_q_dot = 0.
        self.alpha_q_ddot = 0.
        self.gamma = 1e-2
        self.bar_constraint = 1e-6
        self.bar_q_dot = 1e-3
        self.bar_q_ddot = 1e-2

    def call(self, q_cps, t_cps, data):
        _, q_dot_loss, q_ddot_loss, q, q_dot, q_ddot, t, t_cumsum = super().call(q_cps, t_cps, data)

        xyz = self.man.forward_kinematics(q)
        constraint_loss = self.end_effector_constraints_distance_function(xyz)
        t_loss = huber(t[:, tf.newaxis])
        losses = tf.concat([tf.exp(self.alpha_q_dot) * q_dot_loss, tf.exp(self.alpha_q_ddot) * q_ddot_loss,
                            tf.exp(self.alpha_constraint) * constraint_loss, t_loss], axis=-1)
        mean_q_dot_loss = tf.reduce_mean(q_dot_loss, axis=-1)
        mean_q_ddot_loss = tf.reduce_mean(q_ddot_loss, axis=-1)
        mean_constraint_loss = tf.reduce_mean(constraint_loss, axis=-1)

        model_loss = tf.reduce_sum(losses, axis=-1)
        return model_loss, mean_constraint_loss, mean_q_dot_loss, mean_q_ddot_loss, q, q_dot, q_ddot, xyz, t, t_cumsum

    def alpha_update(self, q_dot_loss, q_ddot_loss, constraint_loss):
        alpha_q_dot_update = self.gamma * tf.math.log(q_dot_loss / self.bar_q_dot)
        alpha_q_ddot_update = self.gamma * tf.math.log(q_ddot_loss / self.bar_q_ddot)
        alpha_constraint_update = self.gamma * tf.math.log(constraint_loss / self.bar_constraint)
        self.alpha_q_dot += alpha_q_dot_update
        self.alpha_q_ddot += alpha_q_ddot_update
        self.alpha_constraint += alpha_constraint_update
