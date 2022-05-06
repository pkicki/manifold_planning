from time import time

import numpy as np
from scipy import optimize as spo
import pinocchio as pino
from utils.manipulator import Iiwa


class StartPointOptimizer:
    def __init__(self, urdf_path):
        self.model = pino.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.bounds = spo.Bounds(self.model.lowerPositionLimit[:7], self.model.upperPositionLimit[:7])

    def solve(self, point):
        x0 = np.array([0., 0.7135205165808098, 0., -0.5024774869152212, 0., 1.9256622406212651, 0.])
        #x0 = np.array([np.pi/3 * (2*np.random.random() - 1.),
        #               0.7135205 + 0.3 * (2*np.random.random() - 1.),
        #               np.pi/4 * (2*np.random.random() - 1.),
        #               -0.5024775 + 0.3 * (2*np.random.random() - 1.),
        #               np.pi/4 * (2*np.random.random() - 1.),
        #               1.92566224 + 0.3 * (2*np.random.random() - 1.),
        #               0.])
        options = {'maxiter': 200, 'ftol': 1e-06, 'iprint': 1, 'disp': False,
                   'eps': 1.4901161193847656e-08, 'finite_diff_rel_step': None}
        t0 = time()
        r = spo.minimize(lambda x: self.f(x, point), x0, method='SLSQP',
                         bounds=self.bounds, options=options)
        t1 = time()
        print(r)
        print("TIME:", t1 - t0)
        return r.x

    def f(self, q, hit_point):
        pino.forwardKinematics(self.model, self.data, np.concatenate([q, np.zeros(2)]))
        x = self.data.oMi[-1].translation
        diff = x - hit_point
        return np.linalg.norm(diff)


if __name__ == "__main__":
    urdf_path = "../../iiwa_striker.urdf"
    point = np.array([1.0, 0.0, 0.1505])
    po = StartPointOptimizer(urdf_path)
    po.solve(point)
