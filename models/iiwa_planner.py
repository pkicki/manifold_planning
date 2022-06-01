from math import pi

import tensorflow as tf
import numpy as np

from manifold_planning.utils.constants import Limits
from manifold_planning.utils.data import unpack_data_linear_move


class IiwaPlanner(tf.keras.Model):
    def __init__(self, N, n_pts_fixed_begin, bsp, bsp_t):
        super(IiwaPlanner, self).__init__()
        self.N = N - n_pts_fixed_begin
        self.n_dof = 6
        self.n_pts_fixed_begin = n_pts_fixed_begin
        self.qdd1 = bsp.ddN[0, 0, 0]
        self.qdd2 = bsp.ddN[0, 0, 1]
        self.qdd3 = bsp.ddN[0, 0, 2]
        self.qd1 = bsp.dN[0, 0, 1]
        self.td1 = bsp_t.dN[0, 0, 1]

        activation = tf.keras.activations.tanh
        self.fc = [
            tf.keras.layers.Dense(2048, activation),
            tf.keras.layers.Dense(2048, activation),
            tf.keras.layers.Dense(2048, activation),
            tf.keras.layers.Dense(2048, activation),
            tf.keras.layers.Dense(2048, activation),
        ]

        self.q_est = [
            tf.keras.layers.Dense(2048, activation),
            tf.keras.layers.Dense(self.n_dof * self.N, activation),
        ]

        self.t_est = [
            tf.keras.layers.Dense(20, tf.math.exp, name="time_est"),
        ]

    def __call__(self, x, mul=1.):
        q0, xyz0, xyzk, q_dot_0, q_ddot_0 = unpack_data_linear_move(x, self.n_dof + 1)

        xb = q0 / pi
        if self.n_pts_fixed_begin > 1:
            xb = tf.concat([xb, q_dot_0 / Limits.q_dot[np.newaxis]], axis=-1)
        if self.n_pts_fixed_begin > 2:
            xb = tf.concat([xb, q_ddot_0 / Limits.q_ddot[np.newaxis]], axis=-1)

        x = tf.concat([xb, xyzk], axis=-1)

        for l in self.fc:
            x = l(x)

        q_est = x
        for l in self.q_est:
            q_est = l(q_est)

        dtau_dt = x
        for l in self.t_est:
            dtau_dt = l(dtau_dt)

        q = pi * tf.reshape(q_est, (-1, self.N, self.n_dof))

        q1 = q_dot_0 / dtau_dt[:, :1] / self.qd1 + q0
        q2 = ((q_ddot_0 - self.qd1 * self.td1 * (q1 - q0) * (dtau_dt[:, 1] - dtau_dt[:, 0])[:, np.newaxis]) / dtau_dt[:, :1]
              - self.qdd1 * q0 - self.qdd2 * q1) / self.qdd3

        q0 = q0[:, tf.newaxis]
        q1 = q1[:, tf.newaxis]
        q2 = q2[:, tf.newaxis]

        q_begin = [q0]
        if self.n_pts_fixed_begin > 1:
            q_begin.append(q1)
        if self.n_pts_fixed_begin > 2:
            q_begin.append(q2)

        x = tf.concat(q_begin + [q], axis=-2)
        return x, dtau_dt[..., tf.newaxis] + 1e-5