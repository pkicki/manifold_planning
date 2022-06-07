from losses.feasibility import FeasibilityLoss
from utils.data import unpack_data_boundaries, unpack_data_linear_move
import tensorflow as tf

from utils.manipulator import Iiwa


class LinearMoveLoss(FeasibilityLoss):
    def __init__(self, N, urdf_path, q_dot_limits, q_ddot_limits):
        super(LinearMoveLoss, self).__init__(N, urdf_path, q_dot_limits, q_ddot_limits)
        self.man = Iiwa(urdf_path)

    def call(self, q_cps, t_cps, data):
        q0, xyz0, xyzk, q_dot_0, q_ddot_0 = unpack_data_linear_move(data, 7)

        _, q_dot_loss, q_ddot_loss, q, q_dot, q_ddot, t, t_cumsum = super().call(q_cps, t_cps, data)

        xyz = self.man.forward_kinematics(q)

        xyz0 = xyz0[..., tf.newaxis]
        xyzk = xyzk[..., tf.newaxis]

        t_end = 2.
        #x_range = 0.05
        #y_range = 0.05
        #z_range = 0.05
        x_range = tf.linspace(0.1, 0.02, tf.shape(t_cumsum)[-1])[tf.newaxis]
        y_range = tf.linspace(0.1, 0.02, tf.shape(t_cumsum)[-1])[tf.newaxis]
        z_range = tf.linspace(0.1, 0.02, tf.shape(t_cumsum)[-1])[tf.newaxis]
        vx = (xyzk[:, 0] - xyz0[:, 0]) / t_end
        vy = (xyzk[:, 1] - xyz0[:, 1]) / t_end
        vz = (xyzk[:, 2] - xyz0[:, 2]) / t_end

        xlow = xyz0[:, 0] - x_range + vx * t_cumsum
        xhigh = xyz0[:, 0] + x_range + vx * t_cumsum

        ylow = xyz0[:, 1] - y_range + vy * t_cumsum
        yhigh = xyz0[:, 1] + y_range + vy * t_cumsum

        zlow = xyz0[:, 2] - z_range + vz * t_cumsum
        zhigh = xyz0[:, 2] + z_range + vz * t_cumsum

        xlow_loss = tf.nn.relu(xlow - xyz[..., 0, 0])
        xhigh_loss = tf.nn.relu(xyz[..., 0, 0] - xhigh)
        ylow_loss = tf.nn.relu(ylow - xyz[..., 1, 0])
        yhigh_loss = tf.nn.relu(xyz[..., 1, 0] - yhigh)
        zlow_loss = tf.nn.relu(zlow - xyz[..., 2, 0])
        zhigh_loss = tf.nn.relu(xyz[..., 2, 0] - zhigh)
        geometrical_loss = tf.stack([xlow_loss, xhigh_loss, ylow_loss, yhigh_loss, zlow_loss, zhigh_loss], axis=-1)
        #geometrical_loss = tf.reduce_sum(tf.square(geometrical_loss), axis=-1)
        geometrical_loss = tf.reduce_sum(geometrical_loss, axis=-1)
        linear_move_loss = tf.reduce_mean(geometrical_loss, axis=-1)

        effort_loss = tf.reduce_mean(tf.reduce_sum(tf.square(q_ddot), axis=-1), axis=-1)

        time_loss = tf.square(t - t_end)

        #base_loss = linear_move_loss + q_dot_loss + q_ddot_loss
        #model_loss = linear_move_loss + q_dot_loss + q_ddot_loss + effort_loss + time_loss
        model_loss = linear_move_loss + q_dot_loss + q_ddot_loss + 1e-5 * effort_loss + time_loss
        #model_loss = tf.where(tf.less(base_loss, 1e-3), model_loss, base_loss + time_loss)
        return model_loss, linear_move_loss, effort_loss, time_loss, q_dot_loss, q_ddot_loss, q, q_dot, q_ddot, xyz, t, t_cumsum