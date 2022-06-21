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
        self.alpha_constraint = 0.
        self.alpha_q_dot = 0.
        self.alpha_q_ddot = 0.
        self.lambdas = np.array([1., 1., 1.])
        self.gamma = 1e-2
        self.bar_constraint = 1e-6
        self.bar_q_dot = 1e-3
        self.bar_q_ddot = 1e-2
        #self.bar_constraint = 5e-6
        #self.bar_q_dot = 6e-3
        #self.bar_q_ddot = 6e-2

    def call(self, q_cps, t_cps, data):
        _, q_dot_loss, q_ddot_loss, q, q_dot, q_ddot, t, t_cumsum = super().call(q_cps, t_cps, data)

        xyz = self.man.forward_kinematics(q)
        constraint_loss = self.end_effector_constraints_distance_function(xyz)
        t_loss = huber(t[:, tf.newaxis])
        #q_dot_loss = tf.reduce_sum(q_dot_loss, axis=-1)
        #q_ddot_loss = tf.reduce_sum(q_ddot_loss, axis=-1)
        #constraint_loss = tf.reduce_sum(constraint_loss, axis=-1)
        #L0 = t_loss
        #L1 = constraint_loss / self.bar_constraint
        #L2 = q_dot_loss / self.bar_q_dot
        #L3 = q_ddot_loss / self.bar_q_ddot
        #Ls = tf.stack([L1, L2, L3], axis=-1)[..., tf.newaxis]
        #model_loss = L0[:, 0] + (self.lambdas[np.newaxis, np.newaxis] @ Ls)[:, 0, 0]
        #return model_loss, constraint_loss, q_dot_loss, q_ddot_loss, q, q_dot, q_ddot, xyz, t, t_cumsum, t_loss

        losses = tf.concat([tf.exp(self.alpha_q_dot) * q_dot_loss,
                            tf.exp(self.alpha_q_ddot) * q_ddot_loss,
                            tf.exp(self.alpha_constraint) * constraint_loss,
                            t_loss], axis=-1)
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

    def alpha_update_lagrange(self, gL0, gL1, gL2, gL3):
        A = tf.stack([gL1, gL2, gL3], axis=-1) / tf.stack([self.bar_constraint, self.bar_q_dot, self.bar_q_ddot])[tf.newaxis]
        b = -gL0[:, tf.newaxis]
        t0 = perf_counter()
        #x_pinv = np.linalg.pinv(A) @ b
        t1 = perf_counter()
        AT = tf.transpose(A)
        newA = (AT @ A).numpy()
        newb = (AT @ b).numpy()
        t2 = perf_counter()
        #x = cg(newA, newb)
        t3 = perf_counter()
        x_inv = np.linalg.inv(newA) @ newb
        t4 = perf_counter()
        ts = [t0, t1, t2, t3, t4]
        diffs = np.diff(ts)
        self.lambdas = np.maximum(x_inv[:, 0], 0.)
        self.alpha_constraint = tf.clip_by_value(tf.math.log(self.lambdas[0] / self.bar_constraint), -1e10, 1e10)
        self.alpha_q_dot = tf.clip_by_value(tf.math.log(self.lambdas[1] / self.bar_q_dot), -1e10, 1e10)
        self.alpha_q_ddot = tf.clip_by_value(tf.math.log(self.lambdas[2] / self.bar_q_ddot), -1e10, 1e10)
