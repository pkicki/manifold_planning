import sys
import os
import inspect
import pinocchio as pino

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, os.path.dirname(parentdir))
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
from utils.constants import TableConstraint, Limits, UrdfModels

from models.iiwa_ik_hitting import IiwaIKHitting
from utils.execution import ExperimentHandler

from utils.spo import StartPointOptimizer
from utils.velocity_projection import VelocityProjector


def generate_hitting_direction(x, y):
    """Generates 3 possible hitting directions to score a goal"""
    xg = 2.49
    yg = 0.
    th = np.arctan2(yg - y, xg - x)
    return th

def validate_if_pose_is_possible(x, y, th):
    """Validates if given position and velocity is possible to maintain given TableConstraints"""
    r = 0.05
    xf = x + r * np.cos(th)
    yf = y + r * np.sin(th)
    xb = x - r * np.cos(th)
    yb = y - r * np.sin(th)
    return TableConstraint.in_table_xy(x, y) and TableConstraint.in_table_xy(xf, yf) and TableConstraint.in_table_xy(xb, yb)


def validate_if_pose_is_reachable_with_given_velocity(x, y, v_x, v_y):
    """Validates if given position is reachable with given velocity such that it is possible to maintain given TableConstraints"""
    t = 0.1
    xf = x + v_x * t
    yf = y + v_y * t
    xb = x - v_x * t
    yb = y - v_y * t
    return TableConstraint.in_table_xy(x, y) and TableConstraint.in_table_xy(xf, yf) and TableConstraint.in_table_xy(xb, yb)


def validate_if_initial_mallet_and_puck_positions_makes_hit_possible(xm, ym, xp, yp):
    """Validates if given initial mallet and puck positions enables one to plan reasonable movement"""
    return np.sqrt((ym - yp)**2 + (xm - xp)**2) > 0.3


urdf_path = os.path.join(os.path.dirname(__file__), "..", UrdfModels.striker)
po = StartPointOptimizer(urdf_path)
vp = VelocityProjector(urdf_path)

pino_model = pino.buildModelFromUrdf(urdf_path)
pino_data = pino_model.createData()

opt = tf.keras.optimizers.Adam(1e0)
model = IiwaIKHitting()
experiment_handler = ExperimentHandler("./trainings", "test", 1, model, opt)
#experiment_handler.restore(f"../trainings/velpos/porthos_pos/last_n-20")
#experiment_handler.restore(f"../trained_models/ik_hitting/pos/best-104")
experiment_handler.restore(f"../trained_models/ik_hitting/pos_lossabs/best-77")


def get_hitting_configuration(xk, yk, thk, vz=0.):
    qk = model(np.array([xk, yk, thk])[np.newaxis])
    q = np.concatenate([qk.numpy()[0], np.zeros(3)], axis=-1)
    pino.forwardKinematics(pino_model, pino_data, q)
    xyz_pino = pino_data.oMi[-1].translation
    J = pino.computeJointJacobians(pino_model, pino_data, q)
    pinvJ = np.linalg.pinv(J)
    q_dot = (pinvJ[:6, :3] @ np.array([np.cos(thk), np.sin(thk), vz])[:, np.newaxis])[:, 0]
    max_mul = np.max(np.abs(q_dot) / Limits.q_dot)
    qdotk = q_dot / max_mul
    err = np.abs(xyz_pino - np.array([xk, yk, TableConstraint.Z]))
    return q[:7].tolist(), qdotk.tolist()


data = []
ds = sys.argv[1]
assert ds in ["train", "val", "test", "dummy"]
idx = int(sys.argv[2])
N = int(sys.argv[3])

x0l = 0.6
x0h = 0.9
y0l = -0.4
y0h = 0.4
xkl = 1.0
xkh = 1.3
ykl = -0.4
ykh = 0.4
for i in range(N):
    x0 = (x0h - x0l) * np.random.rand() + x0l
    y0 = (y0h - y0l) * np.random.rand() + y0l
    dz0 = 0.01 * (2*np.random.rand() - 1.)

    xk = (xkh - xkl) * np.random.rand() + xkl
    yk = (ykh - ykl) * np.random.rand() + ykl

    if not validate_if_initial_mallet_and_puck_positions_makes_hit_possible(x0, y0, xk, yk):
        i -= 1
        continue

    point = np.array([x0, y0, TableConstraint.Z+dz0])
    q0 = po.solve(point)

    q_ddot_0 = np.zeros((7))
    q_dot_0 = np.zeros((7))

    ql = Limits.q_dot
    thk = generate_hitting_direction(xk, yk)

    qk, q_dot_k = get_hitting_configuration(xk, yk, thk)
    q_dot_k = np.array(q_dot_k)
    max_gain = np.min(ql / np.abs(q_dot_k[:6]))
    q_dot_k = max_gain * q_dot_k

    if not validate_if_pose_is_reachable_with_given_velocity(xk, yk, np.cos(thk) * max_gain, np.sin(thk) * max_gain):
        i -= 1
        continue

    data.append(q0.tolist() + qk + [xk, yk, thk] + q_dot_0.tolist() + q_ddot_0.tolist() + q_dot_k.tolist() + [0.] + [x0, y0])

dir_name = f"paper/airhockey_maximal_velocity_hitting/{ds}"
os.makedirs(dir_name, exist_ok=True)
np.savetxt(f"{dir_name}/data_{N}_{idx}.tsv", data, delimiter='\t', fmt="%.8f")
os.popen(f'cp {os.path.basename(__file__)} {dir_name}')