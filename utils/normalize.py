import numpy as np


def normalize_xy(xy):
    o = np.ones_like(xy)[..., 0]
    xy = (xy - np.stack([1.0 * o, 0.0 * o], axis=-1)) / np.stack([0.4 * o, 0.4 * o], axis=-1)
    return xy
