from math import pi

import tensorflow as tf
import numpy as np

from utils.constants import Limits


class VelPos(tf.keras.Model):
    def __init__(self):
        super(VelPos, self).__init__()
        self.n_dof = 6

        activation = tf.keras.activations.tanh
        self.fc = [
            tf.keras.layers.Dense(2048, activation),
            tf.keras.layers.Dense(2048, activation),
            tf.keras.layers.Dense(2048, activation),
            tf.keras.layers.Dense(2048, activation),
            tf.keras.layers.Dense(2048, activation),
            tf.keras.layers.Dense(2048, activation),
            tf.keras.layers.Dense(2 * self.n_dof, activation),
        ]


    def __call__(self, x):
        n_data = self.n_dof + 1
        x = tf.cast(x, tf.float32)
        q0 = x[:, :n_data - 1]
        qk = x[:, n_data:2 * n_data - 1]
        xyth = x[:, 2 * n_data: 2 * n_data + 3]
        q_dot_0 = x[:, 2 * n_data + 3: 3 * n_data + 2]
        q_dot_k = x[:, 3 * n_data + 3: 4 * n_data + 2]

        xy = (xyth[..., :2] - np.array([0.95, 0.])[np.newaxis]) / np.array([0.35, 0.4])[np.newaxis]
        ort = np.stack([np.cos(xyth[..., -1]), np.sin(xyth[..., -1])], axis=-1)
        x = np.concatenate([xy, ort], axis=-1)
        for l in self.fc:
            x = l(x)

        qk = np.pi * x[..., :6]
        qdotk = Limits.q_dot[np.newaxis] * x[..., 6:]

        return qk, qdotk


class VelPosSup(tf.keras.Model):
    def __init__(self):
        super(VelPosSup, self).__init__()
        self.n_dof = 6

        activation = tf.keras.activations.tanh
        self.fc = [
            tf.keras.layers.Dense(2048, activation),
            tf.keras.layers.Dense(2048, activation),
            tf.keras.layers.Dense(2048, activation),
            tf.keras.layers.Dense(2048, activation),
        ]
        self.q = [
            tf.keras.layers.Dense(2048, activation),
            tf.keras.layers.Dense(2048, activation),
            tf.keras.layers.Dense(self.n_dof, activation),
        ]
        self.qdot = [
            tf.keras.layers.Dense(2048, activation),
            tf.keras.layers.Dense(2048, activation),
            tf.keras.layers.Dense(self.n_dof, activation),
        ]


    def __call__(self, x):
        x = tf.cast(x, tf.float32)
        xyth = x[:, :3]
        #xy = (xyth[..., :2] - np.array([0.95, 0.])[np.newaxis]) / np.array([0.35, 0.4])[np.newaxis]
        xy = (xyth[..., :2] - np.array([1.1, 0.])[np.newaxis]) / np.array([0.2, 0.4])[np.newaxis]
        ort = np.stack([np.cos(xyth[..., -1]), np.sin(xyth[..., -1])], axis=-1)
        x = np.concatenate([xy, ort], axis=-1)
        for l in self.fc:
            x = l(x)

        q = x
        for l in self.q:
            q = l(q)

        qdot = x
        for l in self.qdot:
            qdot = l(qdot)

        qk = np.pi * q#[..., :6]
        qdotk = Limits.q_dot[np.newaxis] * qdot#x[..., 6:]

        return qk, qdotk
