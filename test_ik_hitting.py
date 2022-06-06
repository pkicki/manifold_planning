import os
from time import perf_counter
import pinocchio as pino
import matplotlib.pyplot as plt

from models.iiwa_ik_hitting import IiwaIKHitting
from utils.constants import UrdfModels, TableConstraint
from utils.execution import ExperimentHandler

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import numpy as np

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

urdf_path = os.path.join(os.path.dirname(__file__), UrdfModels.striker)

batch_size = 1
opt = tf.keras.optimizers.Adam(1e0)
model = IiwaIKHitting()
experiment_handler = ExperimentHandler("./trainings", "test", 1, model, opt)
experiment_handler.restore(f"./trained_models/ik_hitting/pos_lossabs/best-77")
#experiment_handler.restore(f"./trained_models/ik_hitting/pos/best-104")
#experiment_handler.restore(f"./trained_models/ik_hitting/nopos/best-113")

pino_model = pino.buildModelFromUrdf(urdf_path)
data = pino_model.createData()

xk = 0.9
yk = 0.1
thk = 0.0

N, M = 11, 11

x = np.linspace(0.9, 1.4, N)
y = np.linspace(-0.4, 0.4, M)
X, Y = np.meshgrid(x, y)
x_flat = np.reshape(X, -1)
y_flat = np.reshape(Y, -1)
xy = np.stack([x_flat, y_flat], axis=-1)


file_name = f"data/paper/ik_hitting/train/data.tsv"
d = np.loadtxt(file_name, delimiter="\t")
d = d[d[:, 0] < 1.3]
x_flat = d[:, 0]
y_flat = d[:, 1]

#d = np.array([xk, yk, thk])[np.newaxis]
#d = np.concatenate([xy, thk * np.ones_like(xy[:, :1])], axis=-1)
d = d[:, :3]

qk = model(d)
t0 = perf_counter()
qk = model(d)
t1 = perf_counter()
print("INFERENCE TIME", t1 - t0)
print(qk)

e_z = []
for q in qk:
    q = np.concatenate([q.numpy(), np.zeros(3)], axis=-1)
    pino.forwardKinematics(pino_model, data, q)
    xyz_pino = data.oMi[-1].translation
    print(xyz_pino)
    e_z.append(np.abs(xyz_pino[-1] - TableConstraint.Z))

    pos_loss = tf.reduce_sum(tf.square(tf.stack([xk - xyz_pino[0], yk - xyz_pino[1], TableConstraint.Z - xyz_pino[2]], axis=-1)), axis=-1)
    print(pos_loss)

e_z = np.array(e_z)
#e_z = np.reshape(e_z, (N, M))

plt.scatter(x_flat, y_flat, c=e_z)
plt.colorbar()
plt.show()

#q, q_dot, _, _ = get_hitting_configuration(xk, yk, thk, q[:7].tolist())
#
#q = np.concatenate([np.array(q), np.zeros(2)], axis=-1)
#pino.forwardKinematics(pino_model, data, q)
#xyz_pino = data.oMi[-1].translation
#print(xyz_pino)
