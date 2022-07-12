import os


#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from utils.dataset import _ds
from utils.execution import ExperimentHandler
from utils.plotting import plot_qs
from losses.constraint_functions import air_hockey_table
from losses.hittting import HittingLoss
from models.iiwa_planner_boundaries import IiwaPlannerBoundaries
from utils.constants import Limits, TableConstraint, UrdfModels


#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
#config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

class args:
    batch_size = 128
    working_dir = './trainings'
    out_name = 'striker_bs64_lr5em5_N15_huberloss_alphatraining_constraints_integrated'
    log_interval = 100
    learning_rate = 5e-5
    dataset_path = "./data/paper/airhockey_table_moves_v08_a10v_beyond/train/data.tsv"


train_data = np.loadtxt(args.dataset_path, delimiter='\t').astype(np.float32)
train_size = train_data.shape[0]
train_ds = tf.data.Dataset.from_tensor_slices(train_data)

val_data = np.loadtxt(args.dataset_path.replace("train", "val"), delimiter='\t').astype(np.float32)
val_size = val_data.shape[0]
val_ds = tf.data.Dataset.from_tensor_slices(val_data)

urdf_path = os.path.join(os.path.dirname(__file__), UrdfModels.striker)

N = 15
opt = tf.keras.optimizers.Adam(args.learning_rate)
loss = HittingLoss(N, urdf_path, air_hockey_table, Limits.q_dot, Limits.q_ddot)
model = IiwaPlannerBoundaries(N, 3, 2, loss.bsp, loss.bsp_t)

experiment_handler = ExperimentHandler(args.working_dir, args.out_name, args.log_interval, model, opt)

train_step = 0
val_step = 0
best_epoch_loss = 1e10
best_unscaled_epoch_loss = 1e10
for epoch in range(30000):
    # training
    dataset_epoch = train_ds.shuffle(train_size)
    dataset_epoch = dataset_epoch.batch(args.batch_size).prefetch(args.batch_size)
    epoch_loss = []
    unscaled_epoch_loss = []
    experiment_handler.log_training()
    constraint_losses = []
    q_dot_losses = []
    q_ddot_losses = []
    torque_losses = []
    for i, d in _ds('Train', dataset_epoch, train_size, epoch, args.batch_size):
        with tf.GradientTape(persistent=True) as tape:
            q_cps, t_cps = model(d)
            model_loss, constraint_loss, q_dot_loss, q_ddot_loss, torque_loss, q, q_dot, q_ddot, torque, xyz, t, t_cumsum, t_loss, dt, unscaled_model_loss, jerk_loss = loss(q_cps, t_cps, d)
            z_loss_abs = np.mean(np.abs(xyz[..., -1, 0] - TableConstraint.Z), axis=-1)
        grads = tape.gradient(model_loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))

        constraint_losses.append(constraint_loss)
        q_dot_losses.append(q_dot_loss)
        q_ddot_losses.append(q_ddot_loss)
        torque_losses.append(torque_loss)
        epoch_loss.append(model_loss)
        unscaled_epoch_loss.append(unscaled_model_loss)
        with tf.summary.record_if(train_step % args.log_interval == 0):
            tf.summary.scalar('metrics/model_loss', tf.reduce_mean(model_loss), step=train_step)
            tf.summary.scalar('metrics/unscaled_model_loss', tf.reduce_mean(unscaled_model_loss), step=train_step)
            tf.summary.scalar('metrics/constraint_loss', tf.reduce_mean(constraint_loss / 5), step=train_step)
            tf.summary.scalar('metrics/z_loss_abs', tf.reduce_mean(z_loss_abs), step=train_step)
            tf.summary.scalar('metrics/q_dot_loss', tf.reduce_mean(q_dot_loss / 6), step=train_step)
            tf.summary.scalar('metrics/q_ddot_loss', tf.reduce_mean(q_ddot_loss / 6), step=train_step)
            tf.summary.scalar('metrics/torque_loss', tf.reduce_mean(torque_loss / 6), step=train_step)
            tf.summary.scalar('metrics/t', tf.reduce_mean(t), step=train_step)
            tf.summary.scalar('metrics/jerk_loss', tf.reduce_mean(jerk_loss), step=train_step)
        train_step += 1

    constraint_losses = tf.reduce_mean(tf.concat(constraint_losses, 0))
    q_dot_losses = tf.reduce_mean(tf.concat(q_dot_losses, 0))
    q_ddot_losses = tf.reduce_mean(tf.concat(q_ddot_losses, 0))
    torque_losses = tf.reduce_mean(tf.concat(torque_losses, 0))
    loss.alpha_update(q_dot_losses, q_ddot_losses, constraint_losses, torque_losses)
    epoch_loss = tf.reduce_mean(tf.concat(epoch_loss, -1))
    unscaled_epoch_loss = tf.reduce_mean(tf.concat(unscaled_epoch_loss, -1))

    with tf.summary.record_if(True):
        tf.summary.scalar('epoch/loss', epoch_loss, step=epoch)
        tf.summary.scalar('epoch/unscaled_loss', unscaled_epoch_loss, step=epoch)
        tf.summary.scalar('epoch/alpha_q_dot', loss.alpha_q_dot, step=epoch)
        tf.summary.scalar('epoch/alpha_q_ddot', loss.alpha_q_ddot, step=epoch)
        tf.summary.scalar('epoch/alpha_constraint', loss.alpha_constraint, step=epoch)

    # validation
    dataset_epoch = val_ds.shuffle(val_size)
    dataset_epoch = dataset_epoch.batch(args.batch_size).prefetch(args.batch_size)
    epoch_loss = []
    unscaled_epoch_loss = []
    experiment_handler.log_validation()
    for i, d in _ds('Val', dataset_epoch, val_size, epoch, args.batch_size):
        q_cps, t_cps = model(d)
        model_loss, constraint_loss, q_dot_loss, q_ddot_loss, torque_loss, q, q_dot, q_ddot, torque, xyz, t, t_cumsum, t_loss, dt, unscaled_model_loss, jerk_loss = loss(q_cps, t_cps, d)
        z_loss_abs = np.mean(np.abs(xyz[..., -1, 0] - TableConstraint.Z), axis=-1)

        epoch_loss.append(model_loss)
        unscaled_epoch_loss.append(unscaled_model_loss)
        with tf.summary.record_if(val_step % args.log_interval == 0):
            tf.summary.scalar('metrics/model_loss', tf.reduce_mean(model_loss), step=val_step)
            tf.summary.scalar('metrics/unscaled_model_loss', tf.reduce_mean(unscaled_model_loss), step=val_step)
            tf.summary.scalar('metrics/constraint_loss', tf.reduce_mean(constraint_loss / 5), step=val_step)
            tf.summary.scalar('metrics/z_loss_abs', tf.reduce_mean(z_loss_abs), step=val_step)
            tf.summary.scalar('metrics/q_dot_loss', tf.reduce_mean(q_dot_loss / 6), step=val_step)
            tf.summary.scalar('metrics/q_ddot_loss', tf.reduce_mean(q_ddot_loss / 6), step=val_step)
            tf.summary.scalar('metrics/torque_loss', tf.reduce_mean(torque_loss / 6), step=val_step)
            tf.summary.scalar('metrics/t', tf.reduce_mean(t), step=val_step)
            tf.summary.scalar('metrics/jerk_loss', tf.reduce_mean(jerk_loss), step=val_step)
        val_step += 1

    epoch_loss = tf.reduce_mean(tf.concat(epoch_loss, -1))
    unscaled_epoch_loss = tf.reduce_mean(tf.concat(unscaled_epoch_loss, -1))

    with tf.summary.record_if(True):
        tf.summary.scalar('epoch/loss', epoch_loss, step=epoch)
        tf.summary.scalar('epoch/unscaled_loss', unscaled_epoch_loss, step=epoch)

    w = 500
    if epoch % w == w - 1:
        experiment_handler.save_last()
    if best_unscaled_epoch_loss > unscaled_epoch_loss:
        best_unscaled_epoch_loss = unscaled_epoch_loss
        experiment_handler.save_best()
    else:
        if best_epoch_loss > epoch_loss:
            best_epoch_loss = epoch_loss
            experiment_handler.save_best()
