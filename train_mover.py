import os

from models.iiwa_planner import IiwaPlanner
from utils.data import unpack_data_linear_move

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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
    batch_size = 64
    working_dir = './trainings'
    out_name = 'linear_move_bs64_lr5em5_huber_alphalog_4em3_3em2_1em5_1em3'
    log_interval = 10
    learning_rate = 5e-5
    dataset_path = "./data/paper/linear_move/train/data.tsv"


train_data = np.loadtxt(args.dataset_path, delimiter='\t').astype(np.float32)#[:10]
train_size = train_data.shape[0]
train_ds = tf.data.Dataset.from_tensor_slices(train_data)

val_data = np.loadtxt(args.dataset_path.replace("train", "val"), delimiter='\t').astype(np.float32)#[:10]
val_size = val_data.shape[0]
val_ds = tf.data.Dataset.from_tensor_slices(val_data)

urdf_path = os.path.join(os.path.dirname(__file__), UrdfModels.iiwa)

N = 15
opt = tf.keras.optimizers.Adam(args.learning_rate)
loss = LinearMoveLoss(N, urdf_path, Limits.q_dot, Limits.q_ddot)
model = IiwaPlanner(N, 3, loss.bsp, loss.bsp_t)

experiment_handler = ExperimentHandler(args.working_dir, args.out_name, args.log_interval, model, opt)

train_step = 0
val_step = 0
best_epoch_loss = 1e10
for epoch in range(30000):
    dataset_epoch = train_ds.shuffle(train_size)
    dataset_epoch = dataset_epoch.batch(args.batch_size).prefetch(args.batch_size)
    epoch_loss = []
    experiment_handler.log_training()
    linear_move_losses = []
    time_losses = []
    q_dot_losses = []
    q_ddot_losses = []
    for i, d in _ds('Train', dataset_epoch, train_size, epoch, args.batch_size):
        with tf.GradientTape() as tape:
            q_cps, t_cps = model(d)
            model_loss, linear_move_loss, effort_loss, time_loss, q_dot_loss, q_ddot_loss, q, q_dot, q_ddot, xyz, t, t_cumsum = loss(
                q_cps, t_cps, d)
        grads = tape.gradient(model_loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))

        linear_move_losses.append(linear_move_loss)
        time_losses.append(time_loss)
        q_dot_losses.append(q_dot_loss)
        q_ddot_losses.append(q_ddot_loss)
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

    epoch_loss = tf.reduce_mean(tf.concat(epoch_loss, -1))
    linear_move_losses = tf.reduce_mean(tf.concat(linear_move_losses, -1))
    time_losses = tf.reduce_mean(tf.concat(time_losses, -1))
    q_dot_losses = tf.reduce_mean(tf.concat(q_dot_losses, -1))
    q_ddot_losses = tf.reduce_mean(tf.concat(q_ddot_losses, -1))
    loss.alpha_update(q_dot_losses, q_ddot_losses, linear_move_losses, time_losses)

    with tf.summary.record_if(True):
        tf.summary.scalar('epoch/loss', epoch_loss, step=epoch)
        tf.summary.scalar('epoch/alpha_q_dot', loss.alpha_q_dot, step=epoch)
        tf.summary.scalar('epoch/alpha_q_ddot', loss.alpha_q_ddot, step=epoch)
        tf.summary.scalar('epoch/alpha_linear_move', loss.alpha_linear_move, step=epoch)
        tf.summary.scalar('epoch/alpha_time', loss.alpha_time, step=epoch)

    # validation
    dataset_epoch = val_ds.shuffle(val_size)
    dataset_epoch = dataset_epoch.batch(args.batch_size).prefetch(args.batch_size)
    epoch_loss = []
    experiment_handler.log_validation()
    for i, d in _ds('Val', dataset_epoch, val_size, epoch, args.batch_size):
        q_cps, t_cps = model(d)
        model_loss, linear_move_loss, effort_loss, time_loss, q_dot_loss, q_ddot_loss, q, q_dot, q_ddot, xyz, t, t_cumsum = loss(
            q_cps, t_cps, d)

        epoch_loss.append(model_loss)
        with tf.summary.record_if(val_step % args.log_interval == 0):
            tf.summary.scalar('metrics/model_loss', tf.reduce_mean(model_loss), step=val_step)
            tf.summary.scalar('metrics/linear_move_loss', tf.reduce_mean(linear_move_loss), step=val_step)
            tf.summary.scalar('metrics/time_loss', tf.reduce_mean(time_loss), step=val_step)
            tf.summary.scalar('metrics/effort_loss', tf.reduce_mean(effort_loss), step=val_step)
            tf.summary.scalar('metrics/q_dot_loss', tf.reduce_mean(q_dot_loss), step=val_step)
            tf.summary.scalar('metrics/q_ddot_loss', tf.reduce_mean(q_ddot_loss), step=val_step)
            tf.summary.scalar('metrics/t', tf.reduce_mean(t), step=val_step)
        val_step += 1

    epoch_loss = tf.reduce_mean(tf.concat(epoch_loss, -1))

    with tf.summary.record_if(True):
        tf.summary.scalar('epoch/loss', epoch_loss, step=epoch)

    w = 500
    if epoch % w == w - 1:
        experiment_handler.save_last()
    if best_epoch_loss > epoch_loss:
        best_epoch_loss = epoch_loss
        experiment_handler.save_best()