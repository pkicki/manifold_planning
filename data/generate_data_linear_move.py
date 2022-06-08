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
from utils.constants import Limits

from utils.spo import StartPointOptimizer
from utils.velocity_projection import VelocityProjector


urdf_path = os.path.join(os.path.dirname(__file__), "../iiwa.urdf")
print(urdf_path)
po = StartPointOptimizer(urdf_path, 7)
vp = VelocityProjector(urdf_path, 7)
pino_model = pino.buildModelFromUrdf(urdf_path)
pino_data = pino_model.createData()

data = []
ds = sys.argv[1]
assert ds in ["train", "val", "test"]
idx = int(sys.argv[2])
N = int(sys.argv[3])

x0l = 0.3
x0h = 0.6
y0l = -0.5
y0h = -0.4
z0l = 0.2
z0h = 0.4
xkl = 0.3
xkh = 0.6
ykl = 0.4
ykh = 0.5
zkl = 0.2
zkh = 0.4
for i in range(N):
    x0 = (x0h - x0l) * np.random.rand() + x0l
    y0 = (y0h - y0l) * np.random.rand() + y0l
    z0 = (z0h - z0l) * np.random.rand() + z0l

    xk = (xkh - xkl) * np.random.rand() + xkl
    yk = (ykh - ykl) * np.random.rand() + ykl
    zk = (zkh - zkl) * np.random.rand() + zkl

    point = np.array([x0, y0, z0])
    q0 = po.solve(point)
    alpha = 2 * np.random.random(1) - 1.
    v_xyz = 2 * np.random.rand(6) - 1.
    q_dot_0 = vp.compute_q_dot(q0, v_xyz, alpha)[:7]

    dv_xyz = 0.1 * (2 * np.random.random(6) - 1)
    v_xyz_ = v_xyz + dv_xyz
    alpha_ = alpha + 0.2 * (2 * np.random.random(1) - 1.)
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

    data.append(q0.tolist() + [x0, y0, z0] + [xk, yk, zk] + q_dot_0.tolist() + q_ddot_0.tolist())

dir_name = f"paper/linear_move/{ds}"
ranges = [x0l, x0h, y0l, y0h, xkl, xkh, ykl, ykh]
os.makedirs(dir_name, exist_ok=True)
np.savetxt(f"{dir_name}/data_{N}_{idx}.tsv", data, delimiter='\t', fmt="%.8f")
os.popen(f'cp {os.path.basename(__file__)} {dir_name}')
