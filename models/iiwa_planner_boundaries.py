from math import pi
from time import perf_counter

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from utils.constants import Limits
from utils.data import unpack_data_boundaries, unpack_data_kinodynamic
from utils.normalize import normalize_xy


class IiwaPlannerBoundaries(tf.keras.Model):
    def __init__(self, N, n_pts_fixed_begin, n_pts_fixed_end, bsp, bsp_t, n_dof=6):
        super(IiwaPlannerBoundaries, self).__init__()
        self.N = N - n_pts_fixed_begin - n_pts_fixed_end
        self.n_dof = n_dof
        self.n_pts_fixed_begin = n_pts_fixed_begin
        self.n_pts_fixed_end = n_pts_fixed_end
        self.qdd1 = bsp.ddN[0, 0, 0]
        self.qdd2 = bsp.ddN[0, 0, 1]
        self.qdd3 = bsp.ddN[0, 0, 2]
        self.qd1 = bsp.dN[0, 0, 1]
        self.td1 = bsp_t.dN[0, 0, 1]

        activation = tf.keras.activations.tanh
        W = 2048
        self.fc = [
            tf.keras.layers.Dense(W, activation),
            tf.keras.layers.Dense(W, activation),
            tf.keras.layers.Dense(W, activation),
            tf.keras.layers.Dense(W, activation),
            tf.keras.layers.Dense(W, activation),
        ]

        self.q_est = [
            tf.keras.layers.Dense(W, activation),
            tf.keras.layers.Dense(self.n_dof * self.N, activation),
        ]

        self.t_est = [
            tf.keras.layers.Dense(20, tf.math.exp, name="time_est"),
            # tf.keras.layers.Dense(20, tf.math.softplus, name="time_est"),
        ]
        self.fc_np = None
        self.q_np = None
        self.t_np = None

    def prepare_weights(self):
        self.fc_np = []
        for l in self.fc:
            self.fc_np.append((l.kernel.numpy().T[np.newaxis], l.bias.numpy()[np.newaxis], np.tanh))
        self.q_np = []
        for l in self.q_est:
            self.q_np.append((l.kernel.numpy().T[np.newaxis], l.bias.numpy()[np.newaxis], np.tanh))
        self.t_np = []
        for l in self.t_est:
            self.t_np.append((l.kernel.numpy().T[np.newaxis], l.bias.numpy()[np.newaxis], np.exp))

    def prepare_data(self, x):
        return NotImplementedError

    def prepare_data_inference(self, x):
        return NotImplementedError

    def inference(self, x):
        t = []
        t.append(perf_counter())
        x, q0, qd, q_dot_0, q_dot_d, q_ddot_0, expected_time = self.prepare_data_inference(x)
        t.append(perf_counter())
        q_ddot_d = np.zeros_like(q_ddot_0)


        #for l in self.fc:
        #    x = l(x)
        for k, b, a in self.fc_np:
            x = a((k @ x[..., np.newaxis])[..., 0] + b)

        q_est = x
        #for l in self.q_est:
        #    q_est = l(q_est)
        for k, b, a in self.q_np:
            q_est = a((k @ q_est[..., np.newaxis])[..., 0] + b)

        dtau_dt = x
        #for l in self.t_est:
        #    dtau_dt = l(dtau_dt)
        for k, b, a in self.t_np:
            dtau_dt = a((k @ dtau_dt[..., np.newaxis])[..., 0] + b)
        t.append(perf_counter())

        #q_est = q_est.numpy()
        #dtau_dt = dtau_dt.numpy()
        t.append(perf_counter())

        # dtau_dt = dtau_dt / expected_time_[:, tf.newaxis]
        dtau_dt = dtau_dt / expected_time[:, np.newaxis]

        t.append(perf_counter())
        q = pi * np.reshape(q_est, (-1, self.N, self.n_dof))
        t.append(perf_counter())
        s = np.linspace(0., 1., q.shape[1] + 2)[np.newaxis, 1:-1, np.newaxis]
        t.append(perf_counter())

        q1 = q_dot_0 / dtau_dt[:, :1] / self.qd1 + q0
        t.append(perf_counter())
        qm1 = qd - q_dot_d / dtau_dt[:, -1:] / self.qd1
        t.append(perf_counter())
        q2 = ((q_ddot_0 / dtau_dt[:, :1] -
               self.qd1 * self.td1 * (q1 - q0) * (dtau_dt[:, 1] - dtau_dt[:, 0])[:, np.newaxis]) / dtau_dt[:, :1]
              - self.qdd1 * q0 - self.qdd2 * q1) / self.qdd3
        qm2 = ((q_ddot_d / dtau_dt[:, -1:] -
              self.qd1 * self.td1 * (qd - qm1) * (dtau_dt[:, -1] - dtau_dt[:, -2])[:, np.newaxis]) / dtau_dt[:, -1:]
             - self.qdd1 * qd - self.qdd2 * qm1) / self.qdd3
        t.append(perf_counter())

        q0 = q0[:, np.newaxis]
        q1 = q1[:, np.newaxis]
        q2 = q2[:, np.newaxis]
        qm1 = qm1[:, np.newaxis]
        qm2 = qm2[:, tf.newaxis]
        qd = qd[:, np.newaxis]

        q_begin = [q0]
        if self.n_pts_fixed_begin > 1:
            q_begin.append(q1)
        if self.n_pts_fixed_begin > 2:
            q_begin.append(q2)
        q_end = [qd]
        if self.n_pts_fixed_end > 1:
            q_end.append(qm1)
        if self.n_pts_fixed_end > 2:
           q_end.append(qm2)
        t.append(perf_counter())

        qb = q_begin[-1] * (1 - s) + q_end[-1] * s

        x = np.concatenate(q_begin + [q + qb] + q_end[::-1], axis=-2)
        t.append(perf_counter())
        for i in range(len(t) - 1):
            print(i, t[i+1] - t[i])
        return x, dtau_dt[..., np.newaxis]

    def __call__(self, x, mul=1.):
        t = []
        t.append(perf_counter())
        x, q0, qd, q_dot_0, q_dot_d, q_ddot_0, expected_time = self.prepare_data(x)
        t.append(perf_counter())
        q_ddot_d = np.zeros_like(q_ddot_0)

        for l in self.fc:
            x = l(x)

        t.append(perf_counter())
        q_est = x
        for l in self.q_est:
            q_est = l(q_est)

        t.append(perf_counter())
        dtau_dt = x
        for l in self.t_est:
            dtau_dt = l(dtau_dt)
        t.append(perf_counter())

        # dtau_dt = dtau_dt / expected_time_[:, tf.newaxis]
        dtau_dt = dtau_dt / expected_time[:, tf.newaxis]

        t.append(perf_counter())
        q = pi * tf.reshape(q_est, (-1, self.N, self.n_dof))
        t.append(perf_counter())
        #s = tf.linspace(0., 1., tf.shape(q)[1] + 2)[tf.newaxis, 1:-1, tf.newaxis]
        s = np.linspace(0., 1., tf.shape(q)[1] + 2)[np.newaxis, 1:-1, np.newaxis]
        t.append(perf_counter())

        q1 = q_dot_0 / dtau_dt[:, :1] / self.qd1 + q0
        t.append(perf_counter())
        qm1 = qd - q_dot_d / dtau_dt[:, -1:] / self.qd1
        t.append(perf_counter())
        q2 = ((q_ddot_0 / dtau_dt[:, :1] -
               self.qd1 * self.td1 * (q1 - q0) * (dtau_dt[:, 1] - dtau_dt[:, 0])[:, tf.newaxis]) / dtau_dt[:, :1]
              - self.qdd1 * q0 - self.qdd2 * q1) / self.qdd3
        qm2 = ((q_ddot_d / dtau_dt[:, -1:] -
              self.qd1 * self.td1 * (qd - qm1) * (dtau_dt[:, -1] - dtau_dt[:, -2])[:, np.newaxis]) / dtau_dt[:, -1:]
             - self.qdd1 * qd - self.qdd2 * qm1) / self.qdd3
        t.append(perf_counter())

        q0 = q0[:, tf.newaxis]
        q1 = q1[:, tf.newaxis]
        q2 = q2[:, tf.newaxis]
        qm1 = qm1[:, tf.newaxis]
        qm2 = qm2[:, tf.newaxis]
        qd = qd[:, tf.newaxis]

        q_begin = [q0]
        if self.n_pts_fixed_begin > 1:
            q_begin.append(q1)
        if self.n_pts_fixed_begin > 2:
            q_begin.append(q2)
        q_end = [qd]
        if self.n_pts_fixed_end > 1:
            q_end.append(qm1)
        if self.n_pts_fixed_end > 2:
           q_end.append(qm2)
        t.append(perf_counter())

        qb = q_begin[-1] * (1 - s) + q_end[-1] * s

        x = tf.concat(q_begin + [q + qb] + q_end[::-1], axis=-2)
        t.append(perf_counter())
        for i in range(len(t) - 1):
            print(i, t[i+1] - t[i])
        return x, dtau_dt[..., tf.newaxis]


class IiwaPlannerBoundariesHitting(IiwaPlannerBoundaries):
    def __init__(self, N, n_pts_fixed_begin, n_pts_fixed_end, bsp, bsp_t):
        super(IiwaPlannerBoundariesHitting, self).__init__(N, n_pts_fixed_begin, n_pts_fixed_end, bsp, bsp_t)

    def prepare_data(self, x):
        q0, qd, xyth, q_dot_0, q_dot_d, q_ddot_0, puck_pose = unpack_data_boundaries(x, self.n_dof + 1)
        # q_ddot_0 = np.zeros_like(q_ddot_0)
        # q_ddot_d = np.zeros_like(q_ddot_0)

        expected_time = tf.reduce_max(tf.abs(qd - q0) / Limits.q_dot[np.newaxis], axis=-1) + 1e-8
        a_0 = q0[:, tf.newaxis]
        a_1 = (q_dot_0 + 3 * q0)[:, tf.newaxis]
        a_3 = qd[:, tf.newaxis]
        a_2 = (3 * qd - q_dot_d)[:, tf.newaxis]
        t = tf.linspace(0., 1., 128)[tf.newaxis, :, tf.newaxis]
        # q_ = a_3 * t ** 3 + a_2 * t ** 2 * (1 - t) + a_1 * t * (1 - t) ** 2 + a_0 * (1 - t) ** 3
        q_dot_ = 3 * a_3 * t ** 2 + a_2 * (-3 * t ** 2 + 2 * t) + a_1 * (
                3 * t ** 2 - 4 * t + 1) - a_0 * 3 * (1 - t) ** 2
        q_ddot_ = 6 * a_3 * t ** 1 + a_2 * (-6 * t + 2) + \
                  a_1 * (6 * t - 4) + a_0 * 6 * (1 - t)

        q_dot_mul = tf.reduce_max(tf.abs(q_dot_) / Limits.q_dot[np.newaxis, np.newaxis], axis=-1)
        q_ddot_mul = tf.reduce_max(tf.abs(q_ddot_) / Limits.q_ddot[np.newaxis, np.newaxis], axis=-1)

        exp_t_q_dot = tf.reduce_mean(q_dot_mul, axis=-1)
        exp_t_q_ddot = tf.reduce_mean(tf.sqrt(q_ddot_mul), axis=-1)

        expected_time_ = tf.maximum(exp_t_q_dot, exp_t_q_ddot)

        # print(q_dot_mul[0])
        # print(q_ddot_mul[0])
        # print(exp_t_q_dot[0])
        # print(exp_t_q_ddot[0])
        # print(expected_time[0])
        # for i in range(6):
        #    plt.subplot(231 + i)
        #    plt.plot(q_[0, :, i], label="q")
        #    plt.plot(q_dot_[0, :, i], label="dq")
        #    plt.plot(q_ddot_[0, :, i], label="ddq")
        #    plt.legend()
        # plt.show()

        xb = q0 / pi
        if self.n_pts_fixed_begin > 1:
            xb = tf.concat([xb, q_dot_0 / Limits.q_dot[np.newaxis]], axis=-1)
        if self.n_pts_fixed_begin > 2:
            xb = tf.concat([xb, q_ddot_0 / Limits.q_ddot[np.newaxis]], axis=-1)
        xe = qd / pi
        if self.n_pts_fixed_end > 1:
            xe = tf.concat([xe, q_dot_d / Limits.q_dot[np.newaxis]], axis=-1)
        # if self.n_pts_fixed_end > 2:
        #    xe = tf.concat([xe, q_ddot_d / Limits.q_ddot[np.newaxis]], axis=-1)
        xp = normalize_xy(puck_pose)

        # x = tf.concat([xb, xe, xp], axis=-1)
        x = tf.concat([xb, xe], axis=-1)
        return x, q0, qd, q_dot_0, q_dot_d, q_ddot_0, expected_time


class IiwaPlannerBoundariesKinodynamic(IiwaPlannerBoundaries):
    def __init__(self, N, n_pts_fixed_begin, n_pts_fixed_end, bsp, bsp_t):
        super(IiwaPlannerBoundariesKinodynamic, self).__init__(N, n_pts_fixed_begin, n_pts_fixed_end, bsp, bsp_t, n_dof=7)

    def prepare_data(self, x):
        q0, qd, xyz0, xyzk, q_dot_0, q_dot_d, q_ddot_0 = unpack_data_kinodynamic(x, self.n_dof)

        expected_time = tf.reduce_max(tf.abs(qd - q0) / Limits.q_dot7[np.newaxis], axis=-1) + 1e-8

        xb = q0 / pi
        if self.n_pts_fixed_begin > 1:
            xb = tf.concat([xb, q_dot_0 / Limits.q_dot7[np.newaxis]], axis=-1)
        if self.n_pts_fixed_begin > 2:
            xb = tf.concat([xb, q_ddot_0 / Limits.q_ddot7[np.newaxis]], axis=-1)
        xe = qd / pi
        if self.n_pts_fixed_end > 1:
            xe = tf.concat([xe, q_dot_d / Limits.q_dot7[np.newaxis]], axis=-1)
        ## if self.n_pts_fixed_end > 2:
        ##    xe = tf.concat([xe, q_ddot_d / Limits.q_ddot[np.newaxis]], axis=-1)

        x = tf.concat([xb, xe], axis=-1)
        return x, q0, qd, q_dot_0, q_dot_d, q_ddot_0, expected_time

    def prepare_data_inference(self, x):
        q0, qd, xyz0, xyzk, q_dot_0, q_dot_d, q_ddot_0 = unpack_data_kinodynamic(x, self.n_dof)

        expected_time = np.max(np.abs(qd - q0) / Limits.q_dot7[np.newaxis], axis=-1) + 1e-8

        xb = q0 / pi
        if self.n_pts_fixed_begin > 1:
            xb = np.concatenate([xb, q_dot_0 / Limits.q_dot7[np.newaxis]], axis=-1)
        if self.n_pts_fixed_begin > 2:
            xb = np.concatenate([xb, q_ddot_0 / Limits.q_ddot7[np.newaxis]], axis=-1)
        xe = qd / pi
        if self.n_pts_fixed_end > 1:
            xe = np.concatenate([xe, q_dot_d / Limits.q_dot7[np.newaxis]], axis=-1)
        ##if self.n_pts_fixed_end > 2:
        ##   xe = tf.concat([xe, q_ddot_d / Limits.q_ddot[np.newaxis]], axis=-1)

        x = np.concatenate([xb, xe], axis=-1)
        return x, q0, qd, q_dot_0, q_dot_d, q_ddot_0, expected_time
