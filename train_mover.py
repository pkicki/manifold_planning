import os

from models.iiwa_planner import IiwaPlanner
from utils.data import unpack_data_linear_move

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from utils.dataset import _ds
from utils.execution import ExperimentHandler
from utils.plotting import plot_qs
from losses.linear_move import LinearMoveLoss
from utils.constants import Limits, TableConstraint, UrdfModels


# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

class args:
    batch_size = 32
    working_dir = './trainings'
    #out_name = 'lm_if1em5_errorrange_lr5em5_nosq'
    out_name = 'lm_if1em5_errorrange_lr5em6_nosq'
    log_interval = 5
    learning_rate = 5e-6


train_data = np.loadtxt("./data/paper/linear_move/train/data.tsv", delimiter='\t').astype(np.float32)#[:1]
train_size = train_data.shape[0]
train_ds = tf.data.Dataset.from_tensor_slices(train_data)

urdf_path = os.path.join(os.path.dirname(__file__), UrdfModels.iiwa)

N = 15
opt = tf.keras.optimizers.Adam(args.learning_rate)
loss = LinearMoveLoss(N, urdf_path, Limits.q_dot, Limits.q_ddot)
model = IiwaPlanner(N, 3, loss.bsp, loss.bsp_t)

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
            q0, xyz0, xyzk, q_dot_0, q_ddot_0 = unpack_data_linear_move(d, 7)
            q_cps, t_cps = model(d)
            model_loss, linear_move_loss, effort_loss, time_loss, q_dot_loss, q_ddot_loss, q, q_dot, q_ddot, xyz, t, t_cumsum = loss(q_cps, t_cps, d)
        grads = tape.gradient(model_loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))

        epoch_loss.append(model_loss)
        with tf.summary.record_if(train_step % args.log_interval == 0):
            tf.summary.scalar('metrics/model_loss', tf.reduce_mean(model_loss), step=train_step)
            tf.summary.scalar('metrics/linear_move_loss', tf.reduce_mean(linear_move_loss), step=train_step)
            tf.summary.scalar('metrics/time_loss', tf.reduce_mean(time_loss), step=train_step)
            tf.summary.scalar('metrics/effort_loss', tf.reduce_mean(effort_loss), step=train_step)
            tf.summary.scalar('metrics/q_dot_loss', tf.reduce_mean(q_dot_loss), step=train_step)
            tf.summary.scalar('metrics/q_ddot_loss', tf.reduce_mean(q_ddot_loss), step=train_step)
            tf.summary.scalar('metrics/t', tf.reduce_mean(t), step=train_step)
        train_step += 1

    w = 100
    #if epoch % w == w - 1:
    #    #plot_qs(epoch, q, q_dot, q_ddot, t_cumsum)
    #    plt.xlim([-0.2, 0.8])
    #    plt.ylim([-0.5, 0.5])
    #    plt.plot(xyz[0, ..., 0, 0], xyz[0, ..., 1, 0])
    #    plt.plot(xyz0[0, 0], xyz0[0, 1], 'gx')
    #    plt.plot(xyzk[0, 0], xyzk[0, 1], 'rx')
    #    plt.savefig(f"xy_{epoch:05d}.png")
    #    plt.clf()
    epoch_loss = tf.reduce_mean(tf.concat(epoch_loss, -1))

    with tf.summary.record_if(True):
        tf.summary.scalar('epoch/loss', epoch_loss, step=epoch)
    w = 100
    if epoch % w == w - 1:
       experiment_handler.save_last()
