import os
from time import perf_counter
import pinocchio as pino

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
experiment_handler.restore(f"./trained_models/ik_hitting/pos/best-104")
#experiment_handler.restore(f"./trained_models/ik_hitting/nopos/best-113")

pino_model = pino.buildModelFromUrdf(urdf_path)
data = pino_model.createData()

xk = 0.9
yk = 0.1
thk = 1.2

d = np.array([xk, yk, thk])[np.newaxis]

qk = model(d)
t0 = perf_counter()
qk = model(d)
t1 = perf_counter()
print("INFERENCE TIME", t1 - t0)
print(qk)

q = np.concatenate([qk.numpy()[0], np.zeros(3)], axis=-1)
pino.forwardKinematics(pino_model, data, q)
xyz_pino = data.oMi[-1].translation
print(xyz_pino)

pos_loss = tf.reduce_sum(tf.square(tf.stack([xk - xyz_pino[0], yk - xyz_pino[1], TableConstraint.Z - xyz_pino[2]], axis=-1)), axis=-1)
print(pos_loss)

#q, q_dot, _, _ = get_hitting_configuration(xk, yk, thk, q[:7].tolist())
#
#q = np.concatenate([np.array(q), np.zeros(2)], axis=-1)
#pino.forwardKinematics(pino_model, data, q)
#xyz_pino = data.oMi[-1].translation
#print(xyz_pino)
