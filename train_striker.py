import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from utils.dataset import _ds
from utils.execution import ExperimentHandler
from utils.plotting import plot_qs
from losses.constraint_functions import air_hockey_table
from losses.hittting import HittingLoss
from models.iiwa_planner_boundaries import IiwaPlannerBoundaries
from utils.constants import Limits, TableConstraint

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

class args:
    batch_size = 1
    working_dir = './trainings'
    out_name = 'xD'
    log_interval = 5
    learning_rate = 1e-6


train_data = np.loadtxt("./data/paper/airhockey_table_moves/train/data_striker.tsv", delimiter='\t').astype(np.float32)[:1]
train_size = train_data.shape[0]
train_ds = tf.data.Dataset.from_tensor_slices(train_data)

urdf_path = os.path.join(os.path.dirname(__file__), "iiwa_striker.urdf")

N = 15
opt = tf.keras.optimizers.Adam(args.learning_rate)
loss = HittingLoss(N, urdf_path, air_hockey_table, Limits.q_dot, Limits.q_ddot)
model = IiwaPlannerBoundaries(N, 3, 2, loss.bsp, loss.bsp_t)

experiment_handler = ExperimentHandler(args.working_dir, args.out_name, args.log_interval, model, opt)

train_step = 0
val_step = 0
for epoch in range(30000):
    dataset_epoch = train_ds.shuffle(train_size)
    dataset_epoch = dataset_epoch.batch(args.batch_size).prefetch(args.batch_size)
    epoch_loss = []
    experiment_handler.log_training()
    for i, d in _ds('Train', dataset_epoch, train_size, epoch, args.batch_size):
        with tf.GradientTape() as tape:
            q_cps, t_cps = model(d)
            model_loss, constraint_loss, q_dot_loss, q_ddot_loss, q, q_dot, q_ddot, xyz, t, t_cumsum = loss(q_cps, t_cps, d)
            z_loss_abs = np.mean(np.abs(xyz[..., -1, 0] - TableConstraint.Z), axis=-1)
        grads = tape.gradient(model_loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))

        epoch_loss.append(model_loss)
        with tf.summary.record_if(train_step % args.log_interval == 0):
            tf.summary.scalar('metrics/model_loss', tf.reduce_mean(model_loss), step=train_step)
            tf.summary.scalar('metrics/constraint_loss', tf.reduce_mean(constraint_loss), step=train_step)
            tf.summary.scalar('metrics/z_loss_abs', tf.reduce_mean(z_loss_abs), step=train_step)
            tf.summary.scalar('metrics/q_dot_loss', tf.reduce_mean(q_dot_loss), step=train_step)
            tf.summary.scalar('metrics/q_ddot_loss', tf.reduce_mean(q_ddot_loss), step=train_step)
            tf.summary.scalar('metrics/t', tf.reduce_mean(t), step=train_step)
        train_step += 1

    w = 10
    if epoch % w == w - 1:
        plot_qs(epoch, q, q_dot, q_ddot, t_cumsum)
        plt.plot(xyz[0, ..., -1, 0])
        plt.savefig(f"z_{epoch:05d}.png")
        plt.clf()
    epoch_loss = tf.reduce_mean(tf.concat(epoch_loss, -1))

    with tf.summary.record_if(True):
        tf.summary.scalar('epoch/loss', epoch_loss, step=epoch)
    # w = 100
    # if epoch % w == w - 1:
    #    experiment_handler.save_last()
