import os
from copy import copy
from time import perf_counter

import pinocchio as pino
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils.hpo_interface import get_hitting_configuration_opt as ghc, get_hitting_configuration_opt
from utils.spo import StartPointOptimizer
from utils.velocity_projection import VelocityProjector
from utils.bspline import BSpline
from utils.constants import Limits, TableConstraint, UrdfModels, Base
from utils.manipulator import Iiwa
from losses.hittting import HittingLoss
from losses.constraint_functions import air_hockey_table
from models.iiwa_planner_boundaries import IiwaPlannerBoundaries
from models.iiwa_ik_hitting import IiwaIKHitting

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import numpy as np

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

urdf_path = os.path.join(os.path.dirname(__file__), UrdfModels.striker)

N = 15
batch_size = 1
opt = tf.keras.optimizers.Adam(1e0)
man = Iiwa(urdf_path)

vp = VelocityProjector(urdf_path)
po = StartPointOptimizer(urdf_path)
pino_model = pino.buildModelFromUrdf(urdf_path)
pino_data = pino_model.createData()

loss = HittingLoss(N, urdf_path, air_hockey_table, Limits.q_dot, Limits.q_ddot, Limits.q_dddot, Limits.tau)

model = IiwaPlannerBoundaries(N, 3, 2, loss.bsp, loss.bsp_t)
ckpt_striker = tf.train.Checkpoint(model=model)
# ckpt_striker.restore("./trained_models/striker/opt_man_lp/best-149")

hpo_model = IiwaIKHitting()
ckpt_hpo = tf.train.Checkpoint(model=hpo_model)
ckpt_hpo.restore(f"./trained_models/ik_hitting/pos_lossabs/best-77")  # beyond

np.random.seed(444)

XD = 1.


def get_hitting_configuration_hpo(xk, yk, thk):
    qk, q_dot_k = get_hitting_configuration_opt(xk, yk, Base.position[-1], thk)
    if qk is None or q_dot_k is None:
        return None, None
    q = np.concatenate([qk, np.zeros(2)], axis=-1)
    pino.forwardKinematics(pino_model, pino_data, q)
    xyz_pino = pino_data.oMi[-1].translation
    return q[:7].tolist(), q_dot_k[:7]


def get_hitting_configuration(xk, yk, thk):
    # yk *= -1.
    # thk *= -1.
    r = ghc(xk, yk, TableConstraint.Z, thk)
    q_ = np.concatenate([np.array(r[0]), np.zeros(2)], axis=-1)
    print("XYTH:", xk, yk, thk)
    qk = hpo_model(np.array([xk, yk, thk])[np.newaxis]).numpy()[0]
    # print("QK:", qk)
    # qk[0] *= -1.
    # qk[2] *= -1.
    # qk[4] *= -1.
    print("QK,:", qk)
    # thk *= -1.
    q = np.concatenate([qk, np.zeros(3)], axis=-1)
    pino.forwardKinematics(pino_model, pino_data, q)
    xyz_pino = copy(pino_data.oMi[-1].translation)
    print(xyz_pino)
    # assert False
    pino.forwardKinematics(pino_model, pino_data, q_)
    xyz_pino_ = copy(pino_data.oMi[-1].translation)
    # J = pino.computeJointJacobians(pino_model, pino_data, q)
    idx_ = pino_model.getFrameId("F_striker_tip")
    J = pino.computeFrameJacobian(pino_model, pino_data, q, idx_, pino.LOCAL_WORLD_ALIGNED)[:3, :6]
    # pinvJ = np.linalg.pinv(J)
    pinvJ = np.linalg.pinv(J)
    # q_dot = (pinvJ[:6, :3] @ np.array([np.cos(thk), np.sin(thk), 0])[:, np.newaxis])[:, 0]
    q_dot = (pinvJ @ np.array([np.cos(thk), np.sin(thk), 0])[:, np.newaxis])[:, 0]
    max_mul = np.max(np.abs(q_dot)[:6] / Limits.q_dot)
    qdotk = q_dot / max_mul

    pino.forwardKinematics(pino_model, pino_data, q + 0.001 * np.concatenate([qdotk[:6], np.zeros(3)]))
    xyz_pino_ = pino_data.oMi[-1].translation

    diff = xyz_pino_ - xyz_pino

    d = J[:, :6] @ qdotk
    return q[:7].tolist(), qdotk.tolist() + [0.]


def make_d_back():
    x0 = 1.25
    y0 = -0.3
    point = np.array([x0, y0, TableConstraint.Z])
    th0 = 0.1
    q0, q_dot_0 = get_hitting_configuration(x0, y0, th0)
    q_dot_0 = np.array(q_dot_0)
    # q0 = po.solve(point).tolist()

    xk = 0.65
    yk = 0.0
    thk = 0.
    point = np.array([xk, yk, TableConstraint.Z])
    qk = po.solve(point).tolist()

    v_xy = np.array([1., -0.3])
    v_xyz = np.concatenate([v_xy, np.zeros(1)])
    orts = np.array([0., 0., 0.])  # np.zeros(3)
    # alpha = 2 * np.random.random(3) - 1.
    alpha = np.zeros(3)
    # alpha = 2 * np.random.random(3) - 1
    # q_dot_0 = vp.compute_q_dot(np.array(q0), v_xyz, alpha)
    # q_dot_0 = vp.compute_q_dot(np.array(q0), np.concatenate([v_xyz, orts]), alpha)[:6]
    # print(q_dot_0)
    ql = Limits.q_dot
    max_gain = np.min(ql / np.abs(q_dot_0[:6]))
    q_dot_mul = max_gain * (0.8 + 0.2 * np.random.random())
    q_dot_0 = q_dot_mul * q_dot_0
    d = np.array(q0 + qk + [xk, yk, thk] + q_dot_0.tolist() + [0.] * 7)[np.newaxis]
    return d


def make_d():
    q0 = [-9.995474726204004e-13, 0.7135205165808098, 5.8125621129156324e-12, -0.5024774869152212,
          6.092497576479727e-12, 1.9256622406212651, -8.55655325387349e-12]
    x0 = 0.65
    y0 = 0.0
    # y0 = -0.35
    # x0 = 1.05
    # y0 = 0.37
    # x0 = 1.0
    # y0 = 0.3

    # x0 = 0.9355
    # y0 = -0.1432
    point = np.array([x0, y0, TableConstraint.Z])
    q0 = po.solve(point).tolist()
    # xk = 1.11
    # yk = 0.0
    # thk = 0.0
    # xk = 1.16188642
    # xk = 1.05
    # xk = 0.92
    xk = 1.0
    # xk = 0.8733
    # xk = 0.65
    # yk = 0.0
    # yk = -0.28954 * XD
    yk = -0.4 * XD
    # yk = -0.35
    # yk = -0.38
    # yk = 0.35
    # yk = -0.23
    vec_puck_goal = np.array([2.49, 0.]) - np.array([xk, yk])
    thk = np.arctan2(vec_puck_goal[1], vec_puck_goal[0])
    r = 0.06
    xk -= r * np.cos(thk)
    yk -= r * np.sin(thk)
    # thk = 0.2#np.pi
    print("THk", thk)
    # thk = -0.09
    #qk, qdotk = get_hitting_configuration(xk, yk, thk)
    qk, qdotk = get_hitting_configuration_hpo(xk, yk, thk)
    v_xy = np.array([1., -0.3])
    # v_xy = np.array([0., 0.])
    # v_xy = np.array([1.0, 0.])
    # v_xy = np.array([1.0, 0.])
    # v_xy = np.array([1., 0.3])
    # v_xyz = np.concatenate([v_xy, 0.05 * (2 * np.random.random(1) -1)])
    v_xyz = np.concatenate([v_xy, np.zeros(1)])
    # v_xyz = np.array([0., 0., 1.])
    orts = np.array([0., 0., 0.])  # np.zeros(3)
    # alpha = 2 * np.random.random(3) - 1.
    # alpha = np.zeros(3)
    alpha = 2 * np.random.random(3) - 1
    # q_dot_0 = vp.compute_q_dot(np.array(q0), v_xyz, alpha)
    q_dot_0 = vp.compute_q_dot(np.array(q0), np.concatenate([v_xyz, orts]), alpha)[:7]
    print(q_dot_0)
    ql = Limits.q_dot
    max_gain = np.min(ql / np.abs(q_dot_0[:6]))
    q_dot_0 = q_dot_0 * max_gain * np.random.random()

    # q_ddot_0[5] = 10.
    q_ddot_0 = (2 * np.random.random(6) - 1.) * Limits.q_ddot

    # max_gain = np.min(ql / np.abs(qdotk[:6]))
    # qdotk = np.array(qdotk) * max_gain * 0.5#* np.random.random()
    # qdotk = qdotk.tolist()

    q_dot_0 = np.zeros(7)
    # qdotk = np.zeros(7).tolist()
    q_ddot_0 = np.zeros(6)
    # q_dot_0 = np.array([-0.3940178, 0.95033467, -0.25665435, 0.56859636, -0.17929167,
    #                    -2.1379993, 0.])
    # q_ddot_0 = np.array([-1.0400519, 2.025691, -0.25785685, -1.2441797, 0.7523296,
    #                     -5.330265])
    d = np.array(q0 + qk + [xk, yk, thk] + q_dot_0.tolist() + q_ddot_0.tolist() + [0.] + qdotk)[np.newaxis]
    return d.astype(np.float32)


# d = make_d_back()
d = make_d()
# d = np.array([[-0.06196516,  0.83090883, -0.01465779, -0.38815674, -0.04784946,
#         1.5575922 ,  0.        , -0.15483348,  1.186566  , -0.08904937,
#        -0.20075469, -0.08440627,  0.9018738 ,  0.        ,  1.25      ,
#        -0.3       ,  0.23737425, -0.3624336 ,  1.2251222 , -0.4160895 ,
#         0.94839084, -0.06398483, -2.2415693 ,  0.        ,  0.0498569 ,
#         1.3014944 , -1.4711021 , -0.8840394 ,  1.7827575 ,  1.8937368 ,
#         0.        ,  0.05763921,  1.1559542 ,  0.11664998,  0.63290846,
#         0.08206452, -1.88496   ,  0.        ]], dtype=np.float32)

# d = np.array([[-0.11046184,  0.95511037, -0.05292767, -0.31788218, -0.05623439,
#               1.304178  ,  0.        , -0.15472788,  1.18867373, -0.08927208,
#               -0.19621719, -0.08459416,  0.90466259,  0.        ,  1.25      ,
#               -0.3       ,  0.23737425, -0.48904368,  1.45952749, -0.44247532,
#               0.83052087, -0.33957988, -2.0,  0.        ,  0.86444139,
#               1.5174942 ,  0.64008093, -1.85985088,  0.31211853,  2.01080322,
#               0.        ,  0.05728356,  1.15270824,  0.11630429,  0.62892265,
#               0.08247798, -1.88496006,  0.        ]], dtype=np.float32)
# d = np.array([[-0.13851088,  1.0105947 , -0.07288995, -0.27603358, -0.06789164,
#         1.234086  ,  0.        , -0.15472788,  1.1886737 , -0.08927207,
#        -0.19621718, -0.08459416,  0.9046626 ,  0.        ,  1.25      ,
#        -0.3       ,  0.23737425, -0.30997348,  1.1325953 , -0.23810978,
#         0.53202784, -0.27736855, -2.1222696 ,  0.        ,  2.1285472 ,
#        -0.67227364,  0.73738205, -1.3939161 ,  0.63358665,  1.3329544 ,
#         0.        ,  0.05728357,  1.1527083 ,  0.11630429,  0.62892264,
#         0.08247799, -1.88496   ,  0.        ]], dtype=np.float32)
# d[0, 17:23] /= 1.13
# d = np.array([[-1.6155314e-05,  7.0458817e-01,  3.1680618e-06, -4.9303254e-01,
#         1.2753627e-05,  1.9207168e+00,  0.0000000e+00, -1.5469772e-01,
#         1.1879861e+00, -8.9556523e-02, -1.9915390e-01, -8.5135072e-02,
#         9.0284872e-01,  0.0000000e+00,  1.2500000e+00, -3.0000001e-01,
#         2.3737425e-01,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
#         0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
#         0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
#         0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  5.6747824e-02,
#         1.1542890e+00,  1.1664838e-01,  6.3165033e-01,  8.2235627e-02,
#        -1.8849601e+00,  0.0000000e+00]], dtype=np.float32)
# file_name = f"./data/paper/airhockey_table_moves_v08_a08_fixedjac/train/data.tsv"
# file_name = f"./data/paper/airhockey_table_moves_v08_a08_short_/train/data.tsv"
# file_name = f"./data/paper/airhockey_table_moves_v08_a08_short_anglepi2/train/data.tsv"
# file_name = f"./data/paper/airhockey_table_moves_v08_a08_replanning/train/data.tsv"
file_name = f"./data/paper/airhockey_table_moves_v08_a10v_all/train/data.tsv"
data = np.loadtxt(file_name, delimiter="\t").astype(np.float32)

i = 242
# d = data[i:i+1]

# d = np.concatenate([d, d], axis=0)
# d = np.array([[-1.6155314e-05,  7.0458817e-01,  3.1680618e-06, -4.9303254e-01,
#                        1.2753627e-05,  1.9207168e+00,  0.0000000e+00, -1.5469772e-01,
#                        1.1879861e+00, -8.9556523e-02, -1.9915390e-01, -8.5135072e-02,
#                        9.0284872e-01,  0.0000000e+00,  1.2500000e+00, -3.0000001e-01,
#                        2.3737425e-01,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
#                        0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
#                        0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
#                        0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  5.6747824e-02,
#                        1.1542890e+00,  1.1664838e-01,  6.3165033e-01,  8.2235627e-02,
#                        -1.8849601e+00,  0.0000000e+00]], dtype=np.float32)

q_cps, t_cps = model(d)
# for i in range(6):
#    plt.subplot(321 + i)
#    plt.plot(q_cps[0, :, i])
# plt.show()
# model_loss, constraint_loss, z_loss, z_loss_abs, vz_loss, vz_last_loss, q_dot_loss, q_ddot_loss, \
# q_dot_0_loss, q_dot_k_loss, total_dq, q_, q_dot_, q_ddot_, t_, t_cumsum = loss(q_cps, t_cps, d)
# xyz_ = man.forward_kinematics(q_)
# x_ = xyz_[0, :, 0, 0]
# y_ = xyz_[0, :, 1, 0]
# z_ = xyz_[0, :, 2, 0]
##plt.plot(z_)
##plt.show()
# plt.plot(x_, y_)
##
# i = 700
# d[0, :6] = q_[0, i]
# d[0, 17:23] = q_dot_[0, i]
# d[0, 23:29] = q_ddot_[0, i]
# d = np.concatenate([d, d], axis=0)
# d = tf.tile(d, [32, 1])
t0 = perf_counter()
# d[0, 7] *= XD
# d[0, 9] *= XD
# d[0, 11] *= XD
# d[0, 31] *= XD
# d[0, 33] *= XD
# d[0, 35] *= XD
q_cps, t_cps = model(d)
q_cps = q_cps.numpy()
# q_cps[0, :, 0] *= XD
# q_cps[0, :, 2] *= XD
# q_cps[0, :, 4] *= XD
t1 = perf_counter()
print("INFERENCE TIME", t1 - t0)

model_loss, constraint_loss, q_dot_loss, q_ddot_loss, q_dddot_loss, torque_loss, q, q_dot, q_ddot, q_dddot, torque, \
centrifugal, xyz, t, t_cumsum, t_loss, dt, _, jerk, int_toqrue_loss, centrifugal_loss = loss(q_cps, t_cps, d)
z_loss_abs = np.mean(np.abs(xyz[..., -1, 0] - TableConstraint.Z), axis=-1)

# for i in range(6):
#    plt.plot(q_cps[0, :, i], '.', label=f"q_cps_{i}")
# plt.legend()
# plt.show()

t2 = perf_counter()
print("T NN:", t1 - t0)
print("T LOSS:", t2 - t1)
print("T ALL:", t2 - t0)
print("MODEL LOSS", model_loss)
print("Z LOSS ABS", z_loss_abs)
print("Q DOT LOSS", q_dot_loss)
print("Q DDOT LOSS", q_ddot_loss)
print("CENTRIFUGAL LOSS", centrifugal_loss)

sigmoid = lambda x: 1. / (1 + np.exp(-x))
s = t_cumsum / t_cumsum[:, -1]
# c = sigmoid(100.*(s - 2./3))
c = sigmoid(100. * (s - 3. / 4))
centrifugal_end_loss = np.sum(np.abs(centrifugal) * dt[..., np.newaxis] * c[..., np.newaxis], axis=(1, 2))

print("CENTRIFUGAL END LOSS", centrifugal_end_loss)

bsp = BSpline(N)
bsp_t = BSpline(20)

t0 = perf_counter()
q = bsp.N @ q_cps
q_dot_tau = bsp.dN @ q_cps
q_ddot_tau = bsp.ddN @ q_cps

dtau_dt = bsp_t.N @ t_cps
ddtau_dtt = bsp_t.dN @ t_cps

# plt.plot(ddtau_dtt[0, :, 0])
# plt.plot((dtau_dt[0, 1:, 0] - dtau_dt[0, :-1, 0]) * 1024.)
# plt.show()
ts = 1. / dtau_dt[..., 0] / 1024.
t = tf.cumsum(ts, axis=-1)

from scipy.interpolate import BSpline as BSp
from scipy.interpolate import interp1d

# arg = np.linspace(0., 1., 1024)
# k = 7
# knots = np.linspace(0., 1., 15 + 1 + 7 - 2 * k)
##knots = np.array([0., 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 1.])
# knots = np.array([0., 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 1.])
# knots = np.pad(knots, k, 'edge')
# qb = BSp(knots, q_cps.numpy()[0], 7)
# for i in range(6):
#    plt.subplot(231 + i)
#    plt.plot(q[0, :, i], '-')
#    plt.plot(qb(arg)[:, i], '--')
#    plt.plot([0, 1023], [q_cps[0, 0, i], q_cps[0, -1, i]], '*')
# plt.show()

# for i in range(6):
#    plt.subplot(231 + i)
#    plt.plot(arg, q[0, :, i], '-')
#    plt.plot(t[0] / t[0, -1], q[0, :, i], '--')
# plt.show()

# assert False


t = np.concatenate([np.zeros((1, 1)), t[:, :-1]], axis=-1)[0]
si = interp1d(t, np.linspace(0., 1., 1024), axis=-1)
n = 200
targ = np.linspace(t[0], t[-1], int(n * t[-1]))
s = si(targ)

dtau_dt_bs = BSp(bsp_t.u, t_cps[0, :, 0], 7)
ddtau_dtt_bs = dtau_dt_bs.derivative()
q_bs = BSp(bsp.u, q_cps[0, :], 7)
dq_bs = q_bs.derivative()
ddq_bs = dq_bs.derivative()

q_ = q_bs(s)
dq_ = dq_bs(s)
ddq_ = ddq_bs(s)
dtau_dt_ = dtau_dt_bs(s)[..., np.newaxis]
ddtau_dtt_ = ddtau_dtt_bs(s)[..., np.newaxis]
# plt.plot(targ, q_, '*')
# plt.plot(t, q[0, :], '.')
# plt.show()

q_dot_ = dq_ * dtau_dt_
q_ddot_ = ddq_ * dtau_dt_ ** 2 + ddtau_dtt_ * dq_ * dtau_dt_

q_dot = q_dot_tau * dtau_dt
q_ddot = q_ddot_tau * dtau_dt ** 2 + ddtau_dtt * q_dot_tau * dtau_dt

# plt.plot(targ, q_dot_, '*')
# plt.plot(t, q_dot[0], '.')
# plt.show()
#
# plt.plot(targ, q_ddot_, 'r')
# plt.plot(t, q_ddot[0], 'b')
# plt.show()

drift = ddtau_dtt * q_dot_tau

# plt.plot(q_ddot_tau[0, :, 0])
# plt.plot((q_dot_tau[0, 1:, 0] - q_dot_tau[0, :-1, 0]) * 1024.)
# plt.show()

ts = 1. / dtau_dt[..., 0] / 1024.
t = tf.cumsum(ts, axis=-1)

# for i in range(1):
#    plt.plot(t[0], q_dot[0, :, i], label=f"dq_{i}")
#    plt.plot(t[0, :-1], (q[0, 1:, i] - q[0, :-1, i]) / np.diff(t[0]), '--', label=f"est_dq_{i}")
# plt.legend()
# plt.grid()
# plt.show()
#
# for i in range(1):
#    plt.plot(t[0], q_ddot[0, :, i], label=f"ddq_{i}")
#    plt.plot(t[0], q_ddot_[0, :, i], '-.', label=f"ddq____{i}")
#    plt.plot(t[0], drift[0, :, i], '-.', label=f"drift{i}")
#    plt.plot(t[0, :-1], (q_dot[0, 1:, i] - q_dot[0, :-1, i]) / np.diff(t[0]), '--', label=f"est_ddq_{i}")
# plt.legend()
# plt.grid()
# plt.show()

t1 = perf_counter()
print("T BSPP", t1 - t0)

# np.savetxt("1.path", q[0, ::16])

xyz = man.forward_kinematics(q)
x = xyz[0, :, 0, 0]
y = xyz[0, :, 1, 0]
z = xyz[0, :, 2, 0]

pino.forwardKinematics(pino_model, pino_data, np.concatenate([q[0, -1], np.zeros(3)], axis=-1))
xyz_pino_m1 = pino_data.oMi[-1].translation.copy()
pino.forwardKinematics(pino_model, pino_data, np.concatenate([q[0, -2], np.zeros(3)], axis=-1))
xyz_pino_m2 = pino_data.oMi[-1].translation.copy()
diff = xyz_pino_m1 - xyz_pino_m2

plt.plot(x, XD * y)
plt.xlim(0.6, 1.4)
plt.ylim(-0.4, 0.4)
plt.show()

# plt.plot(t[0], centrifugal[0])
# plt.show()

# vz = np.abs(np.diff(z)) / ts[0, :-1] * 1024.
# vx = np.abs(np.diff(x)) / ts[0, :-1] * 1024.
# vy = np.abs(np.diff(y)) / ts[0, :-1] * 1024.
# vxy = np.sqrt(vx**2 + vy**2)
# plt.plot(vxy)
# plt.show()

q_ = np.pad(q[0], [[0, 0], [0, 3]], mode='constant')
dq_ = np.pad(q_dot[0], [[0, 0], [0, 3]], mode='constant')
ddq_ = np.pad(q_ddot[0], [[0, 0], [0, 3]], mode='constant')

# torque = []
# centrifugal = []
# for i in range(1024):
#    torque.append(pino.rnea(pino_model, pino_data, q_[i], dq_[i], ddq_[i]))
#    centrifugal.append(pino.rnea(pino_model, pino_data, q_[i], dq_[i], np.zeros_like(dq_[i])) -
#                       pino.rnea(pino_model, pino_data, q_[i], np.zeros_like(dq_[i]), np.zeros_like(dq_[i])))
# torque = np.stack(torque, axis=0)
# centrifugal = np.stack(centrifugal, axis=0)
#
# centrifugal_loss = np.sum(centrifugal * dt[..., np.newaxis])
int_centrifugal = np.cumsum(np.abs(centrifugal) * dt[..., np.newaxis], axis=1)

plt.subplot(241)
for i in range(6):
    plt.plot(t[0], q[0, :, i], label=str(i))
plt.legend()
plt.subplot(242)
for i in range(6):
    plt.plot(t[0], q_dot[0, :, i], label="d" + str(i))
plt.legend()
plt.subplot(243)
for i in range(6):
    plt.plot(t[0], q_ddot[0, :, i], label="dd" + str(i))
plt.legend()
plt.subplot(244)
plt.plot(ts[0])
plt.subplot(245)
for i in range(6):
    plt.plot(t[0], torque[0, :, i], label="torque" + str(i))
    # plt.plot(q_dddot[0, :, i], label="jerk" + str(i))
#    #plt.plot(t[0, :-1, 0], (q[0, 1:, i] - q[0, :-1, i]) * 1024., label=str(i))
plt.legend()
plt.subplot(246)
for i in range(6):
    # plt.plot(q_dot_tau[0, :, i], label="d" + str(i))
    plt.plot(t[0], centrifugal[0, :, i], label=f"cc {i}")
    plt.ylim(-2.5, 2.5)
plt.legend()
plt.subplot(247)
# for i in range(6):
# plt.plot(q_ddot_tau[0, :, i], label="dd" + str(i))
# plt.plot(t[0], int_centrifugal[0, :, i], label=f"int cc {i}")
# plt.ylim(0., 0.3)
plt.plot(t[0], x, label=f"x")
plt.plot(t[0], y, label=f"y")
plt.legend()
plt.subplot(248)
plt.plot(z)

plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, 'bo', markersize=0.1)
ax.set_xlim3d(0.7, 1.5)
ax.set_ylim3d(-0.4, 0.4)
ax.set_zlim3d(0.05, 0.25)
plt.show()

plt.plot(t[0], z)
plt.show()
