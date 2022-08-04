import os, sys
from copy import copy
from time import perf_counter

SCRIPT_DIR = os.path.dirname(__file__)
PACKAGE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(PACKAGE_DIR)

import pinocchio as pino
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils.hpo_interface import get_hitting_configuration_opt
from utils.model import model_inference
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
# ckpt_striker.restore("./trained_models/striker/best-60")
# ckpt_striker.restore("./trained_models/striker/v08a08z1e3/best-157")

# ckpt_striker.restore("./trained_models/striker/v08a08huberalpha/best-183")
# ckpt_striker.restore("./trained_models/striker/slow1stjoint/best-95")

# ckpt_striker.restore("./trained_models/striker/v08a08huberalphaint/best-26")
# ckpt_striker.restore("./trained_models/striker/v08a08huberalphaint/best-60")
# ckpt_striker.restore("./trained_models/striker/v08a08huberalphaint/best-79")
# ckpt_striker.restore("./trained_models/striker/v08a08huberalphaint/best-87")
# ckpt_striker.restore("./trained_models/striker/v08a08huberalphaint/best-119")
# ckpt_striker.restore("./trained_models/striker/v08a10v_huberalpha/best-272")
# ckpt_striker.restore("./trained_models/striker/v08a08fixedjac/best-79")
# ckpt_striker.restore("./trained_models/striker/v08a08fixedjac/best-134")
# ckpt_striker.restore("./trained_models/striker/v08a08fixedjac/best-67")
# ckpt_striker.restore("./trained_models/striker/v08a10v_jerk/best-161")
# ckpt_striker.restore("./trained_models/striker/v08a10v_beyond/best-168")
# ckpt_striker.restore("./trained_models/striker/v08a10v_beyond/best-271")
# ckpt_striker.restore("./trained_models/striker/v08a10v_torqueint/last_n-99") # torqueint 1em2 -> adas1
# ckpt_striker.restore("./trained_models/striker/v08a10v_torqueint/last_n-107") # torqueint 1em2 -> adas1
# ckpt_striker.restore("./trained_models/striker/v08a10v_torqueint/last_n-37") # torqueint 1em1 -> athos
# ckpt_striker.restore("./trained_models/striker/v08a10v_torqueint/last_n-48") # torqueint 1em0 -> robot40
# ckpt_striker.restore("./trained_models/striker/v08a10v_torqueint/last_n-37") # torqueint 1em1 -> aramis
# ckpt_striker.restore("./trained_models/striker/v08a10v_torqueint/last_n-39") # torqueint 1em1 -> aramis
# ckpt_striker.restore("./trained_models/striker/v08a10v_torqueint/best-67") # torqueint 1em1 80k -> aramis
# ckpt_striker.restore("./trained_models/striker/v08a10v_torqueint/last_n-80") # torqueint 0em0 -> robot40
# ckpt_striker.restore("./trained_models/striker/v08a10v_centrifugal/last_n-65") # centrifugal 1em2 time 0 -> adas1
# ckpt_striker.restore("./trained_models/striker/v08a10v_centrifugal/best-88") # centrifugal true 1em1 -> porthos
# ckpt_striker.restore("./trained_models/striker/v08a10v_centrifugal/best-92") # centrifugal true 1em1 -> porthos
# ckpt_striker.restore("./trained_models/striker/v08a10v_centrifugal/best-99") # centrifugal true 1em1 -> porthos
#ckpt_striker.restore("./trained_models/striker/v08a10v_centrifugal/best-115")  # centrifugal true 1em1 -> porthos
# ckpt_striker.restore("./trained_models/striker/v08a10v_centrifugal/last_n-103") # centrifugal true 1em3 -> aramis
# ckpt_striker.restore("./trained_models/striker/v08a10v_centrifugalsigmoid/best-51") # centrifugal a25b06c025 -> athos

# ckpt_striker.restore("./trained_models/striker/v08a10v_myopt_centrifugalsigmoid/last_n-64") # robot40
ckpt_striker.restore("./trained_models/striker/v08a10v_newopt/best-44")  # porthos jerk1em4 hitting
#ckpt_striker.restore("./trained_models/striker/v08a10v_newopt/last_n-35")  # adas1 hitting

# ckpt_striker.restore("./trained_models/striker/v08a10v_tilted/last_n-67") # tilted -> porthos
# ckpt_striker.restore("./trained_models/striker/v08a10v_tilted/best-143") # tilted -> porthos
# ckpt_striker.restore("./trained_models/striker/v08a10v_tilted/last_n-134") # tilted 44 -> athos

hpo_model = IiwaIKHitting()
ckpt_hpo = tf.train.Checkpoint(model=hpo_model)
# ckpt_hpo.restore("./trained_models/ik_hitting/pos/best-104")
# ckpt_hpo.restore("./trained_models/ik_hitting/nopos/best-113")
ckpt_hpo.restore(f"./trained_models/ik_hitting/pos_lossabs/best-77")
# ckpt_hpo.restore(f"./trained_models/ik_hitting/pos_lossabs/best-23")
# ckpt_hpo.restore(f"./trained_models/ik_hitting/pos_lossabs/best-138")

np.random.seed(444)


def make_d():
    q0 = [-9.995474726204004e-13, 0.7135205165808098, 5.8125621129156324e-12, -0.5024774869152212,
          6.092497576479727e-12, 1.9256622406212651, -8.55655325387349e-12]
    x0 = 0.65
    y0 = 0.0
    point = np.array([x0, y0, TableConstraint.Z])
    q0 = po.solve(point).tolist()
    xk = 0.9
    yk = -0.38
    vec_puck_goal = np.array([2.49, 0.]) - np.array([xk, yk])
    thk = np.arctan2(vec_puck_goal[1], vec_puck_goal[0])
    # thk = 0.2#np.pi
    print("THk", thk)
    # thk = -0.09
    qk, qdotk = get_hitting_configuration_opt(xk, yk, Base.position[-1], thk)
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
    q_ddot_0 = np.zeros(6)
    d = np.array(q0 + qk + [xk, yk, thk] + q_dot_0.tolist() + q_ddot_0.tolist() + [0.] + qdotk)[np.newaxis]
    return d.astype(np.float32)


# d = make_d_back()
d = make_d()

# d = np.concatenate([d, d], axis=0)

q, q_dot, q_ddot, t, q_cps, t_cps = model_inference(model, d, loss.bsp, loss.bsp_t)

d_rep = np.concatenate([q[-1], [0.], d[0, :7], d[0, 14:17], q_dot[-1], [0.], q_ddot[-1], [0.] * 8,
                        ], axis=-1).astype(np.float32)[np.newaxis]

#r = 0.15
#thk = 0.2346
#xk = 0.9 + r * np.cos(thk)
#yk = -0.38 + r * np.sin(thk)
#qk, qdotk = get_hitting_configuration_opt(xk, yk, Base.position[-1], thk)
#print(len(qk))
#print(len(qdotk))
#d_rep = np.concatenate([q[-1], [0.], qk, d[0, 14:17], q_dot[-1], [0.], q_ddot[-1], [0.], qdotk
#                        ], axis=-1).astype(np.float32)[np.newaxis]

qr, dqr, ddqr, tr, qr_cps, tr_cps = model_inference(model, d_rep, loss.bsp, loss.bsp_t)
_ = loss(qr_cps[np.newaxis].astype(np.float32), tr_cps[np.newaxis].astype(np.float32), d_rep.astype(np.float32))
# model_loss, constraint_loss, q_dot_loss, q_ddot_loss, q, q_dot, q_ddot, xyz, t, t_cumsum, t_loss = loss(q_cps, t_cps, d)
# z_loss_abs = np.mean(np.abs(xyz[..., -1, 0] - TableConstraint.Z), axis=-1)

#for i in range(6):
#    plt.subplot(231 + i)
#    plt.plot(t, q[:, i], label=f"base_{i}")
#    # plt.plot(tl, ql[:, i], label=f"use_last_{i}")
#    plt.plot(tr + t[-1], qr[:, i], label=f"replan_{i}")
#    plt.legend()
#plt.show()

xyz = man.forward_kinematics(q)
x = xyz[:, 0, 0]
y = xyz[:, 1, 0]
z = xyz[:, 2, 0]
xyzr = man.forward_kinematics(qr)
xr = xyzr[:, 0, 0]
yr = xyzr[:, 1, 0]
zr = xyzr[:, 2, 0]

plt.plot(x, y, label='hit')
plt.plot(xr, yr, label='back')
# plt.plot(xl, yl, label='base_shift')
plt.legend()
plt.xlim(0.6, 1.4)
plt.ylim(-0.4, 0.4)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, 'bo', markersize=0.1)
ax.plot(xr, yr, zr, 'ro', markersize=0.1)
# ax.plot(xl, yl, zl, 'mo', markersize=0.1)
ax.set_xlim3d(0.7, 1.5)
ax.set_ylim3d(-0.4, 0.4)
ax.set_zlim3d(0.05, 0.25)
plt.show()

plt.plot(t, z)
# plt.plot(tl, zl)
plt.plot(t[-1] + tr, zr)
plt.show()

plt.subplot(231)
for i in range(6):
    plt.plot(t, q[:, i], label=str(i))
    plt.plot(t[-1] + tr, qr[:, i], label="r" + str(i))
plt.legend()
plt.subplot(232)
for i in range(6):
    plt.plot(t, q_dot[:, i], label="d" + str(i))
    plt.plot(t[-1] + tr, dqr[:, i], label= "rd" + str(i))
plt.legend()
plt.subplot(233)
for i in range(6):
    plt.plot(t, q_ddot[:, i], label="dd" + str(i))
    plt.plot(t[-1] + tr, ddqr[:, i], label="rdd" + str(i))
plt.legend()
plt.subplot(234)
plt.plot(t, z)
plt.plot(t[-1] + tr, z)

plt.show()
