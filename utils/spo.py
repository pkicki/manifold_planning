from time import time

import numpy as np
from scipy import optimize as spo
import pinocchio as pino

from utils.constants import UrdfModels, TableConstraint, Base
from utils.manipulator import Iiwa


class StartPointOptimizer:
    def __init__(self, urdf_path, n=9):
        self.n = n
        self.model = pino.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.bounds = spo.Bounds(self.model.lowerPositionLimit[:6], self.model.upperPositionLimit[:6])

    def solve(self, point, q0=None):
        if q0 is None:
            #q0 = np.array(Base.configuration + [0.])
            q0 = Base.configuration
        #q0 = np.pad(q0, [[0, 7 - len(q0)]])
        #x0 = np.array([0., 0.7135205165808098, 0., -0.5024774869152212, 0., 1.9256622406212651, 0.])
        #x0 = np.array([0.0, 0.06580,  0.0, -1.45996, 0.,  1.22487, 0.0])
        #x0 = np.array([np.pi/3 * (2*np.random.random() - 1.),
        #               0.7135205 + 0.3 * (2*np.random.random() - 1.),
        #               np.pi/4 * (2*np.random.random() - 1.),
        #               -0.5024775 + 0.3 * (2*np.random.random() - 1.),
        #               np.pi/4 * (2*np.random.random() - 1.),
        #               1.92566224 + 0.3 * (2*np.random.random() - 1.),
        #               0.])
        options = {'maxiter': 300, 'ftol': 1e-06, 'iprint': 1, 'disp': False,
                   'eps': 1.4901161193847656e-08, 'finite_diff_rel_step': None}
        t0 = time()
        r = spo.minimize(lambda x: self.f(x, point), q0, method='SLSQP',
                         bounds=self.bounds, options=options)
        t1 = time()
        print(r)
        print("TIME:", t1 - t0)
        return np.pad(r.x, [[0, 1]])

    def f(self, q, hit_point):
        pino.forwardKinematics(self.model, self.data, np.pad(q, (0, self.n - len(q)), mode='constant'))
        x = self.data.oMi[-1].translation
        diff = x - hit_point
        return np.linalg.norm(diff)


if __name__ == "__main__":
    #urdf_path = "../../iiwa_striker.urdf"
    #urdf_path = "../iiwa.urdf"
    urdf_path = "../" + UrdfModels.striker
    point = np.array([0.65, 0.0, TableConstraint.Z])
    #po = StartPointOptimizer(urdf_path, 7)
    po = StartPointOptimizer(urdf_path)
    q = po.solve(point)
    pino.forwardKinematics(po.model, po.data, np.pad(q, (0, po.n - len(q)), mode='constant'))
    x = po.data.oMi[-1].translation
    a = 0
