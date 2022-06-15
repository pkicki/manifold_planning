import os
from copy import copy
from time import perf_counter

import pinocchio as pino
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils.hpo_interface import get_hitting_configuration_opt as ghc
from utils.spo import StartPointOptimizer
from utils.velocity_projection import VelocityProjector
from utils.bspline import BSpline
from utils.constants import Limits, TableConstraint, UrdfModels
from utils.manipulator import Iiwa
from losses.hittting import HittingLoss
from losses.constraint_functions import air_hockey_table
from models.iiwa_planner_boundaries import IiwaPlannerBoundaries
from models.iiwa_ik_hitting import IiwaIKHitting

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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

loss = HittingLoss(N, urdf_path, air_hockey_table, Limits.q_dot, Limits.q_ddot)

model = IiwaPlannerBoundaries(N, 3, 2, loss.bsp, loss.bsp_t)
ckpt_striker = tf.train.Checkpoint(model=model)
#ckpt_striker.restore("./trained_models/striker/best-60")
#ckpt_striker.restore("./trained_models/striker/v08a08z1e3/best-157")
ckpt_striker.restore("./trained_models/striker/v08a08huberalpha/best-133")

hpo_model = IiwaIKHitting()
ckpt_hpo = tf.train.Checkpoint(model=hpo_model)
#ckpt_hpo.restore("./trained_models/ik_hitting/pos/best-104")
#ckpt_hpo.restore("./trained_models/ik_hitting/nopos/best-113")
ckpt_hpo.restore(f"./trained_models/ik_hitting/pos_lossabs/best-77")

np.random.seed(444)

def get_hitting_configuration(xk, yk, thk):

    r = ghc(xk, yk, TableConstraint.Z, thk)
    q_ = np.concatenate([np.array(r[0]), np.zeros(2)], axis=-1)
    qk = hpo_model(np.array([xk, yk, thk])[np.newaxis])
    q = np.concatenate([qk.numpy()[0], np.zeros(3)], axis=-1)
    pino.forwardKinematics(pino_model, pino_data, q)
    xyz_pino = copy(pino_data.oMi[-1].translation)
    J = pino.computeJointJacobians(pino_model, pino_data, q)
    pinvJ = np.linalg.pinv(J)
    q_dot = (pinvJ[:6, :3] @ np.array([np.cos(thk), np.sin(thk), 0])[:, np.newaxis])[:, 0]
    max_mul = np.max(np.abs(q_dot)[:6] / Limits.q_dot)
    qdotk = q_dot / max_mul

    pino.forwardKinematics(pino_model, pino_data, q + 0.001 * np.concatenate([qdotk[:6], np.zeros(3)]))
    xyz_pino_ = pino_data.oMi[-1].translation

    diff = xyz_pino_ - xyz_pino

    d = J[:, :6] @ qdotk
    return q[:7].tolist(), qdotk.tolist() + [0.]


def make_d_back():
    x0 = 1.1
    y0 = -0.3
    point = np.array([x0, y0, TableConstraint.Z])
    th0 = 0.1
    q0, q_dot_0 = get_hitting_configuration(x0, y0, th0)
    q_dot_0 = np.array(q_dot_0)
    #q0 = po.solve(point).tolist()

    xk = 0.65
    yk = 0.0
    thk = 0.
    point = np.array([xk, yk, TableConstraint.Z])
    qk = po.solve(point).tolist()

    v_xy = np.array([1., -0.3])
    v_xyz = np.concatenate([v_xy, np.zeros(1)])
    orts = np.array([0., 0., 0.])#np.zeros(3)
    #alpha = 2 * np.random.random(3) - 1.
    alpha = np.zeros(3)
    #alpha = 2 * np.random.random(3) - 1
    #q_dot_0 = vp.compute_q_dot(np.array(q0), v_xyz, alpha)
    #q_dot_0 = vp.compute_q_dot(np.array(q0), np.concatenate([v_xyz, orts]), alpha)[:6]
    #print(q_dot_0)
    ql = Limits.q_dot
    max_gain = np.min(ql / np.abs(q_dot_0[:6]))
    q_dot_mul = max_gain * (0.8 + 0.2 * np.random.random())
    q_dot_0 = q_dot_mul * q_dot_0
    d = np.array(q0 + qk + [xk, yk, thk] + q_dot_0.tolist() + [0.]*7)[np.newaxis]
    return d


def make_d():
    q0 = [-9.995474726204004e-13, 0.7135205165808098, 5.8125621129156324e-12, -0.5024774869152212, 6.092497576479727e-12, 1.9256622406212651, -8.55655325387349e-12]
    x0 = 0.66
    #x0 = 1.05
    y0 = 0.0
    #y0 = 0.37
    #x0 = 1.0
    #y0 = 0.3
    point = np.array([x0, y0, TableConstraint.Z])
    q0 = po.solve(point).tolist()
    #xk = 1.11
    #yk = 0.0
    #thk = 0.0
    #xk = 1.16188642
    xk = 1.2
    #xk = 0.95
    #xk = 1.1
    #yk = 0.0
    yk = -0.35
    #yk = 0.35
    #yk = -0.23
    vec_puck_goal = np.array([2.49, 0.]) - np.array([xk, yk])
    thk = np.arctan2(vec_puck_goal[1], vec_puck_goal[0])
    #thk = 0.2#np.pi
    print("THk", thk)
    #thk = -0.09
    qk, qdotk = get_hitting_configuration(xk, yk, thk)
    v_xy = np.array([1., -0.3])
    #v_xy = np.array([0., 0.])
    #v_xy = np.array([1.0, 0.])
    #v_xy = np.array([1.0, 0.])
    #v_xy = np.array([1., 0.3])
    #v_xyz = np.concatenate([v_xy, 0.05 * (2 * np.random.random(1) -1)])
    v_xyz = np.concatenate([v_xy, np.zeros(1)])
    #v_xyz = np.array([0., 0., 1.])
    orts = np.array([0., 0., 0.])#np.zeros(3)
    #alpha = 2 * np.random.random(3) - 1.
    #alpha = np.zeros(3)
    alpha = 2 * np.random.random(3) - 1
    #q_dot_0 = vp.compute_q_dot(np.array(q0), v_xyz, alpha)
    q_dot_0 = vp.compute_q_dot(np.array(q0), np.concatenate([v_xyz, orts]), alpha)[:7]
    print(q_dot_0)
    ql = Limits.q_dot
    max_gain = np.min(ql / np.abs(q_dot_0[:6]))
    q_dot_0 = q_dot_0 * max_gain * np.random.random()
    #q_ddot_0 = np.zeros(6)
    #q_ddot_0[5] = 10.
    q_ddot_0 = (2 * np.random.random(6) - 1.) * Limits.q_ddot

    #max_gain = np.min(ql / np.abs(qdotk[:6]))
    #qdotk = np.array(qdotk) * max_gain * 0.5#* np.random.random()
    #qdotk = qdotk.tolist()
    d = np.array(q0 + qk + [xk, yk, thk] + q_dot_0.tolist() + q_ddot_0.tolist() + [0.] + qdotk)[np.newaxis]
    return d

#d = make_d_back()
d = make_d()

#d = np.concatenate([d, d], axis=0)

q_cps, t_cps = model(d)
#for i in range(6):
#    plt.subplot(321 + i)
#    plt.plot(q_cps[0, :, i])
#plt.show()
#model_loss, constraint_loss, z_loss, z_loss_abs, vz_loss, vz_last_loss, q_dot_loss, q_ddot_loss, \
#q_dot_0_loss, q_dot_k_loss, total_dq, q_, q_dot_, q_ddot_, t_, t_cumsum = loss(q_cps, t_cps, d)
#xyz_ = man.forward_kinematics(q_)
#x_ = xyz_[0, :, 0, 0]
#y_ = xyz_[0, :, 1, 0]
#z_ = xyz_[0, :, 2, 0]
##plt.plot(z_)
##plt.show()
#plt.plot(x_, y_)
##
#i = 700
#d[0, :6] = q_[0, i]
#d[0, 17:23] = q_dot_[0, i]
#d[0, 23:29] = q_ddot_[0, i]
#d = np.concatenate([d, d], axis=0)
#d = tf.tile(d, [32, 1])
t0 = perf_counter()
q_cps, t_cps = model(d)
t1 = perf_counter()
print("INFERENCE TIME", t1 - t0)

model_loss, constraint_loss, q_dot_loss, q_ddot_loss, q, q_dot, q_ddot, xyz, t, t_cumsum = loss(q_cps, t_cps, d)
z_loss_abs = np.mean(np.abs(xyz[..., -1, 0] - TableConstraint.Z), axis=-1)

t2 = perf_counter()
print("T NN:", t1 - t0)
print("T LOSS:", t2 - t1)
print("T ALL:", t2 - t0)
print("MODEL LOSS", model_loss)
print("Z LOSS ABS", z_loss_abs)
print("Q DOT LOSS", q_dot_loss)
print("Q DDOT LOSS", q_ddot_loss)

bsp = BSpline(N)
bsp_t = BSpline(20)

t0 = perf_counter()
q = bsp.N @ q_cps
q_dot_tau = bsp.dN @ q_cps
q_ddot_tau = bsp.ddN @ q_cps

dtau_dt = bsp_t.N @ t_cps
ddtau_dtt = bsp_t.dN @ t_cps

#plt.plot(ddtau_dtt[0, :, 0])
#plt.plot((dtau_dt[0, 1:, 0] - dtau_dt[0, :-1, 0]) * 1024.)
#plt.show()

q_dot = q_dot_tau * dtau_dt
q_ddot = q_ddot_tau * dtau_dt ** 2 + ddtau_dtt * q_dot_tau * dtau_dt
q_ddot_ = q_ddot_tau * dtau_dt ** 2 + ddtau_dtt * q_dot_tau

drift = ddtau_dtt * q_dot_tau

#plt.plot(q_ddot_tau[0, :, 0])
#plt.plot((q_dot_tau[0, 1:, 0] - q_dot_tau[0, :-1, 0]) * 1024.)
#plt.show()

ts = 1. / dtau_dt[..., 0] / 1024.
t = tf.cumsum(ts, axis=-1)

for i in range(1):
    plt.plot(t[0], q_dot[0, :, i], label=f"dq_{i}")
    plt.plot(t[0, :-1], (q[0, 1:, i] - q[0, :-1, i]) / np.diff(t[0]), '--', label=f"est_dq_{i}")
plt.legend()
plt.grid()
plt.show()

for i in range(1):
    plt.plot(t[0], q_ddot[0, :, i], label=f"ddq_{i}")
    plt.plot(t[0], q_ddot_[0, :, i], '-.', label=f"ddq____{i}")
    plt.plot(t[0], drift[0, :, i], '-.', label=f"drift{i}")
    plt.plot(t[0, :-1], (q_dot[0, 1:, i] - q_dot[0, :-1, i]) / np.diff(t[0]), '--', label=f"est_ddq_{i}")
plt.legend()
plt.grid()
plt.show()

t1 = perf_counter()
print("T BSPP", t1 - t0)

np.savetxt("1.path", q[0, ::16])

xyz = man.forward_kinematics(q)
x = xyz[0, :, 0, 0]
y = xyz[0, :, 1, 0]
z = xyz[0, :, 2, 0]

pino.forwardKinematics(pino_model, pino_data, np.concatenate([q.numpy()[0, -1], np.zeros(3)], axis=-1))
xyz_pino_m1 = pino_data.oMi[-1].translation.copy()
pino.forwardKinematics(pino_model, pino_data, np.concatenate([q.numpy()[0, -2], np.zeros(3)], axis=-1))
xyz_pino_m2 = pino_data.oMi[-1].translation.copy()
diff = xyz_pino_m1 - xyz_pino_m2

plt.plot(x, y)
plt.xlim(0.6, 1.4)
plt.ylim(-0.4, 0.4)
plt.show()

vz = np.abs(np.diff(z)) / ts[0, :-1] * 1024.
vx = np.abs(np.diff(x)) / ts[0, :-1] * 1024.
vy = np.abs(np.diff(y)) / ts[0, :-1] * 1024.
vxy = np.sqrt(vx**2 + vy**2)
plt.plot(vxy)
plt.show()

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
# plt.subplot(245)
# for i in range(6):
#    plt.plot(q[0, :, i], label=str(i))
#    #plt.plot(t[0, :-1, 0], (q[0, 1:, i] - q[0, :-1, i]) * 1024., label=str(i))
# plt.legend()
plt.subplot(246)
for i in range(6):
    plt.plot(q_dot_tau[0, :, i], label="d" + str(i))
plt.legend()
plt.subplot(247)
for i in range(6):
    plt.plot(q_ddot_tau[0, :, i], label="dd" + str(i))
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