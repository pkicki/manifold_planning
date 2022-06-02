import numpy as np


class ManipulatorDimensions:
    L1 = 0.8
    L2 = 0.6
    L3 = 0.3
    X = -0.4
    Y = 0.3
    W1 = 0.005
    W2 = 0.005
    W3 = 0.005


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
    q_dot = np.array([1.4835, 1.4835, 1.7453, 1.3090, 2.2689, 2.3562], dtype=np.float32)
    q_ddot = np.array([10., 10., 10., 10., 10., 10.], dtype=np.float32)


class Base:
    configuration = [-9.995474726204004e-13, 0.7135205165808098, 5.8125621129156324e-12, -0.5024774869152212,
                     6.092497576479727e-12, 1.9256622406212651, -8.55655325387349e-12]
