import tensorflow as tf
import numpy as np


def gamma(x, n):
    r = []
    for i in range(n):
        r.append(tf.sin(2**i * np.pi * x))
        r.append(tf.cos(2**i * np.pi * x))
    return tf.stack(r, axis=-1)

