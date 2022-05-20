import numpy as np
import scipy
from scipy import optimize as spo
import pinocchio as pino


class VelocityProjector:
    def __init__(self, urdf_path, n=9):
        self.n = n
        self.model = pino.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.bounds = spo.Bounds(self.model.lowerPositionLimit[:7], self.model.upperPositionLimit[:7])

    def compute_q_dot(self, q, v_xyz, alpha):
        q = np.pad(q, (0, self.n - q.shape[0]), mode='constant')
        pino.computeJointJacobians(self.model, self.data, q)
        #J_36 = self.data.J[:3, :6]
        J_36 = self.data.J
        null_J_6 = scipy.linalg.null_space(J_36)
        pinvJ_6 = np.linalg.pinv(J_36)
        q_dot = pinvJ_6 @ v_xyz + null_J_6 @ alpha
        return q_dot

if __name__ == "__main__":
    urdf_path = "../../iiwa_striker.urdf"
    q = np.array([0., 0.7135205165808098, 0., -0.5024774869152212, 0., 1.9256622406212651, 0., 0., 0.])
    v = np.array([1., 0.5, 0.])[:, np.newaxis]
    alpha = np.array([1., 0.5, 0.])[:, np.newaxis]
    vp = VelocityProjector(urdf_path)
    q_dot = vp.compute_q_dot(q, v, alpha)
    print(q_dot)
