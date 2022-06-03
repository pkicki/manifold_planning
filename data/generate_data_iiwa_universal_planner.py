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

from data.spo import StartPointOptimizer
from data.velocity_projection import VelocityProjector


def generate_hitting_directions(x, y):
    """Generates 3 possible hitting directions to score a goal"""
    xg = 2.49
    yg = 0.
    yb = 0.46
    # Direction 1: straight
    th1 = np.arctan2(yg - y, xg - x)
    # Direction 2: upper band
    xbu = (yb * x - xg * y + xg * yb) / (2 * yb - y)
    thu = np.arctan2(yb - y, xbu - x)
    upper_valid = xg > xbu and xbu > x
    # Direction 3: lower band
    xbl = (yb * x + xg * y + xg * yb) / (2 * yb + y)
    thl = np.arctan2(-y - yb, xbl - x)
    lower_valid = xg > xbl and xbl > x

    hitting_directions = [th1]
    if upper_valid:
        hitting_directions.append(thu)
    if lower_valid:
        hitting_directions.append(thl)
    return hitting_directions

def test_generate_hitting_directions(x, y):
    import matplotlib.pyplot as plt
    x = 1.65
    y = 0.3
    r = 0.2
    hd = generate_hitting_directions(x, y)
    plt.plot([0.59, 2.49, 2.49, 0.59, 0.59], [0.46, 0.46, -0.46, -0.46, 0.46])
    plt.plot(x, y, 'gx')
    plt.plot(2.49, 0.0, 'rx')
    plt.plot([x, x + r * np.cos(hd[1])], [y, y + r * np.sin(hd[1])], 'r')
    plt.plot([x, x + r * np.cos(hd[2])], [y, y + r * np.sin(hd[2])], 'b')
    plt.xlim(0.5, 2.5)
    plt.ylim(-1., 1.)
    plt.show()


def validate_if_hitting_direction_is_possible(x, y, th):
    """Validates if given hitting direction is possible to achieve given TableConstraints"""
    r = 0.20
    xc = x - r * np.cos(th)
    yc = y - r * np.sin(th)
    return TableConstraint.in_table_xy(xc, yc)


def validate_if_initial_pose_and_velocity_is_possible(x, y, th):
    """Validates if given initial position and velocity is possible to maintain given TableConstraints"""
    r = 0.2
    xc = x + r * np.cos(th)
    yc = y + r * np.sin(th)
    return TableConstraint.in_table_xy(x, y) and TableConstraint.in_table_xy(xc, yc)

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
experiment_handler.restore(f"../trained_models/ik_hitting/pos/best-104")


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
    err = np.abs(xyz_pino - np.array([xk, yk, 0.1505]))
    return q[:7].tolist(), qdotk.tolist()


data = []
ds = sys.argv[1]
assert ds in ["train", "val", "test", "dummy"]
idx = int(sys.argv[2])
N = int(sys.argv[3])

x0l = 0.6
x0h = 1.4
y0l = -0.4
y0h = 0.4
xkl = 0.6
xkh = 1.4
ykl = -0.4
ykh = 0.4
for i in range(N):
    x0 = (x0h - x0l) * np.random.rand() + x0l
    y0 = (y0h - y0l) * np.random.rand() + y0l
    dz0 = 0.01 * (2*np.random.rand() - 1.)
    th0 = np.pi * (2 * np.random.random() - 1.)

    xk = (xkh - xkl) * np.random.rand() + xkl
    yk = (ykh - ykl) * np.random.rand() + ykl

    if not validate_if_initial_mallet_and_puck_positions_makes_hit_possible(x0, y0, xk, yk):
        i -= 1
        continue
    #if not validate_if_pose_is_possible(x0, y0, th0): continue

    v_xy = np.array([np.cos(th0), np.sin(th0)])
    v_z = 0.1 * (2 * np.random.random(1) - 1)
    v_xyz = np.concatenate([v_xy, v_z, np.zeros(3)])

    point = np.array([x0, y0, TableConstraint.Z+dz0])
    q0 = po.solve(point)
    alpha = 2 * np.random.random(3) - 1.
    q_dot_0 = vp.compute_q_dot(q0, v_xyz, alpha)[:7]

    dth = 0.2 * (2 * np.random.random() - 1)
    v_xy_ = np.array([np.cos(th0+dth), np.sin(th0+dth)])
    v_z_ = v_z + 0.02 * (2 * np.random.random(1) - 1)
    v_xyz_ = np.concatenate([v_xy_, v_z_, np.zeros(3)])
    alpha_ = alpha + 0.2 * (2 * np.random.random(3) - 1.)
    q_dot_0_ = vp.compute_q_dot(q0, v_xyz_, alpha_)[:7]
    q_ddot_0 = (q_dot_0_ - q_dot_0)[:6]
    q_ddot_0_mul = np.min(Limits.q_ddot / np.abs(q_ddot_0))
    q_ddot_0 = np.random.random() * q_ddot_0_mul * q_ddot_0

    ql = Limits.q_dot
    max_gain = np.min(ql / np.abs(q_dot_0[:6]))
    q_dot_mul = max_gain * np.random.random()
    if np.random.random() < 0.2:
        q_dot_mul = 0.
    q_dot_0 = q_dot_mul * q_dot_0

    if not validate_if_pose_is_reachable_with_given_velocity(x0, y0, v_xy[0] * q_dot_mul, v_xy[1] * q_dot_mul):
        i -= 1
        continue

    thk = np.pi * (2 * np.random.random() - 1.)
    qk, q_dot_k = get_hitting_configuration(xk, yk, thk)
    q_dot_k = np.array(q_dot_k)
    max_gain = np.min(ql / np.abs(q_dot_k[:6]))
    q_dot_mul = max_gain * np.random.random()
    if np.random.random() < 0.5:
        q_dot_mul = max_gain
    q_dot_k = q_dot_mul * q_dot_k

    dth = 0.2 * (2 * np.random.random() - 1)
    vz = 0.02 * np.random.random()
    _, q_dot_k_ = get_hitting_configuration(xk, yk, thk + dth, vz=vz)
    q_ddot_k = (np.array(q_dot_k_) - q_dot_k)
    q_ddot_k_mul = np.min(Limits.q_ddot / np.abs(q_ddot_k[:6]))
    q_ddot_k = np.random.random() * q_ddot_k_mul * q_ddot_k

    if not validate_if_pose_is_reachable_with_given_velocity(xk, yk, np.cos(thk) * q_dot_mul, np.sin(thk) * q_dot_mul):
        i -= 1
        continue


    data.append(q0.tolist() + qk + [xk, yk, thk] + q_dot_0.tolist() + q_ddot_0.tolist() + [0.] + q_dot_k.tolist())
    data.append(qk + q0.tolist() + [x0, y0, -th0] + q_dot_k.tolist() + q_ddot_k.tolist() + [0.] + (-q_dot_0).tolist())

dir_name = f"paper/airhockey_table_moves/{ds}"
ranges = [x0l, x0h, y0l, y0h, xkl, xkh, ykl, ykh]
os.makedirs(dir_name, exist_ok=True)
np.savetxt(f"{dir_name}/data_{N}_{idx}.tsv", data, delimiter='\t', fmt="%.8f")
os.popen(f'cp {os.path.basename(__file__)} {dir_name}')