import numpy as np


class TableConstraint:
    XLB = 0.6
    # XLB = 0.55
    YLB = -0.45
    XRT = 2.
    YRT = 0.45
    Z = 0.16

    @staticmethod
    def in_table_xy(x, y):
        return TableConstraint.XLB <= x <= TableConstraint.XRT and TableConstraint.YLB <= y <= TableConstraint.YRT


class Limits:
    q = np.array([2.967, 2.094, 2.967, 2.094, 2.967, 2.094])
    q_dot = 0.8 * np.array([1.4835, 1.4835, 1.7453, 1.3090, 2.2689, 2.3562], dtype=np.float32)
    #q_ddot = 0.8 * np.array([10., 10., 10., 10., 10., 10.], dtype=np.float32)
    q_ddot = 10. * q_dot
    q_ddot = np.min(np.stack([q_ddot, 20. * np.ones((6,), dtype=np.float32)], axis=-1), axis=-1)
    tau = 0.8 * np.array([320, 320, 176, 176, 110, 40], dtype=np.float32)
    q_dddot = 5 * q_ddot


class Base:
    #configuration = [0.0, 0.06811, 0.0, -1.48, 0., 1.2544]
    configuration = [-7.16000830e-06, 6.97494070e-01, 7.26955352e-06, -5.04898567e-01, 6.60813111e-07, 1.92857916e+00]
    position = [0.65, 0., 0.16]
    # [-9.995474726204004e-13, 0.7135205165808098, 5.8125621129156324e-12, -0.5024774869152212,
    # 6.092497576479727e-12, 1.9256622406212651, -8.55655325387349e-12]


class UrdfModels:
    striker = "iiwa_striker_new.urdf"
    iiwa = "iiwa.urdf"
    virtual_box = "virtual_box.urdf"
