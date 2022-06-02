import tensorflow as tf
import numpy as np

from utils.normalize import normalize_xy


class PosSup(tf.keras.Model):
    def __init__(self):
        super(PosSup, self).__init__()
        self.n_dof = 6

        activation = tf.keras.activations.tanh
        self.fc = [
            tf.keras.layers.Dense(2048, activation),
            tf.keras.layers.Dense(2048, activation),
            tf.keras.layers.Dense(2048, activation),
            tf.keras.layers.Dense(2048, activation),
            tf.keras.layers.Dense(2048, activation),
            tf.keras.layers.Dense(2048, activation),
            tf.keras.layers.Dense(self.n_dof, activation),
        ]

    def __call__(self, x):
        x = tf.cast(x, tf.float32)
        xyth = x[:, :3]
        xy = normalize_xy(xyth[..., :2])
        ort = np.stack([np.cos(xyth[..., -1]), np.sin(xyth[..., -1])], axis=-1)
        x = np.concatenate([xy, ort], axis=-1)
        for l in self.fc:
            x = l(x)
        qk = np.pi * x
        return qk
