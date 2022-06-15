from losses.feasibility import FeasibilityLoss
from losses.utils import huber
from utils.data import unpack_data_boundaries, unpack_data_linear_move
import tensorflow as tf

from utils.manipulator import Iiwa


class LinearMoveLoss(FeasibilityLoss):
    def __init__(self, N, urdf_path, q_dot_limits, q_ddot_limits):
        super(LinearMoveLoss, self).__init__(N, urdf_path, q_dot_limits, q_ddot_limits)
        self.man = Iiwa(urdf_path)
        self.alpha_linear_move = 0.
        self.alpha_time = 0.
        self.alpha_q_dot = 0.
        self.alpha_q_ddot = 0.

    def call(self, q_cps, t_cps, data):
        q0, xyz0, xyzk, q_dot_0, q_ddot_0 = unpack_data_linear_move(data, 7)

        _, q_dot_loss, q_ddot_loss, q, q_dot, q_ddot, t, t_cumsum = super().call(q_cps, t_cps, data)

        xyz = self.man.forward_kinematics(q)

        xyz0 = xyz0[..., tf.newaxis]
        xyzk = xyzk[..., tf.newaxis]

        t_end = 2.
        d0 = 0.1
        d1 = 0.02
        d_range_v = (d1 - d0) / t_end
        vx = (xyzk[:, 0] - xyz0[:, 0]) / t_end
        vy = (xyzk[:, 1] - xyz0[:, 1]) / t_end
        vz = (xyzk[:, 2] - xyz0[:, 2]) / t_end

        xlow = xyz0[:, 0] - (d0 + d_range_v * t_cumsum) + vx * t_cumsum
        xhigh = xyz0[:, 0] + d0 + d_range_v * t_cumsum + vx * t_cumsum

        ylow = xyz0[:, 1] - (d0 + d_range_v * t_cumsum) + vy * t_cumsum
        yhigh = xyz0[:, 1] + d0 + d_range_v * t_cumsum + vy * t_cumsum

        zlow = xyz0[:, 2] - (d0 + d_range_v * t_cumsum) + vz * t_cumsum
        zhigh = xyz0[:, 2] + d0 + d_range_v * t_cumsum + vz * t_cumsum

        huber_along_path = lambda x: tf.reduce_mean(huber(x), axis=-1)
        relu_huber_along_path = lambda x: huber_along_path(tf.nn.relu(x))
        xlow_loss = relu_huber_along_path(xlow - xyz[..., 0, 0])
        xhigh_loss = relu_huber_along_path(xyz[..., 0, 0] - xhigh)
        ylow_loss = relu_huber_along_path(ylow - xyz[..., 1, 0])
        yhigh_loss = relu_huber_along_path(xyz[..., 1, 0] - yhigh)
        zlow_loss = relu_huber_along_path(zlow - xyz[..., 2, 0])
        zhigh_loss = relu_huber_along_path(xyz[..., 2, 0] - zhigh)
        linear_move_loss = tf.stack([xlow_loss, xhigh_loss, ylow_loss, yhigh_loss, zlow_loss, zhigh_loss], axis=-1)

        time_loss = huber(t - t_end)

        effort_loss = tf.reduce_mean(huber(q_ddot), axis=-2)

        losses = tf.concat([tf.exp(self.alpha_q_dot) * q_dot_loss, tf.exp(self.alpha_q_ddot) * q_ddot_loss,
                            tf.exp(self.alpha_linear_move) * linear_move_loss, tf.exp(self.alpha_time) * time_loss[:, tf.newaxis],
                            effort_loss], axis=-1)

        mean_q_dot_loss = tf.reduce_mean(q_dot_loss, axis=-1)
        mean_q_ddot_loss = tf.reduce_mean(q_ddot_loss, axis=-1)
        mean_linear_move_loss = tf.reduce_mean(linear_move_loss, axis=-1)

        model_loss = tf.reduce_sum(losses, axis=-1)

        return model_loss, mean_linear_move_loss, effort_loss, time_loss, mean_q_dot_loss, mean_q_ddot_loss, q, q_dot, q_ddot, xyz, t, t_cumsum


    def alpha_update(self, q_dot_loss, q_ddot_loss, linear_move_loss, time_loss):
        gamma = 1e-2
        alpha_q_dot_update = gamma * (q_dot_loss - 4e-3)
        alpha_q_ddot_update = gamma * (q_ddot_loss - 3e-2)
        alpha_linear_move_update = gamma * (linear_move_loss - 1e-5)
        alpha_time_update = gamma * (time_loss - 1e-3)
        self.alpha_time += alpha_time_update
        self.alpha_q_dot += alpha_q_dot_update
        self.alpha_q_ddot += alpha_q_ddot_update
        self.alpha_linear_move += alpha_linear_move_update
