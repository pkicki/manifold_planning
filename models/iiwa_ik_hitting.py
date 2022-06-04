import tensorflow as tf
import numpy as np

from utils.normalize import normalize_xy
from utils.preprocessing import gamma


class ResDense(tf.keras.Model):
    def __init__(self, nout, act):
        super(ResDense, self).__init__(name='')
        self.d = tf.keras.layers.Dense(nout, activation=act)
        self.nout = nout

    def call(self, input_tensor):
        x = self.d(input_tensor)
        y = tf.pad(input_tensor, [[0, 0], [0, self.nout - tf.shape(input_tensor)[-1]]], "CONSTANT")
        return x + y


class IiwaIKHitting(tf.keras.Model):
    def __init__(self):
        super(IiwaIKHitting, self).__init__()
        self.n_dof = 6
        self.l_preproc = 4

        activation = tf.keras.activations.tanh
        self.fc = [
            ResDense(2048, activation),
            ResDense(2048, activation),
            ResDense(2048, activation),
            ResDense(2048, activation),
            ResDense(2048, activation),
            ResDense(2048, activation),
            tf.keras.layers.Dense(self.n_dof, activation),
        ]

    def __call__(self, x):
        x = tf.cast(x, tf.float32)
        xyth = x[:, :3]
        xy = normalize_xy(xyth[..., :2])
        # x = gamma(xyth[..., 0], self.l_preproc)
        # y = gamma(xyth[..., 1], self.l_preproc)
        # th = gamma(xyth[..., 2], self.l_preproc)

        ort = tf.stack([tf.cos(xyth[..., -1]), tf.sin(xyth[..., -1])], axis=-1)
        x = tf.concat([xy, ort], axis=-1)
        # x = tf.concat([x, y, th], axis=-1)
        for l in self.fc:
            x = l(x)
        qk = np.pi * x
        return qk
