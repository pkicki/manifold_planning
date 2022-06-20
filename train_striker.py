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
from utils.constants import Limits, TableConstraint, UrdfModels


#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
#config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

class args:
    batch_size = 4
    working_dir = './trainings'
    out_name = 'striker_bs64_lr5em5_N15_huberloss_alphatraining'
    log_interval = 10
    learning_rate = 5e-5
    dataset_path = "./data/paper/airhockey_table_moves/train/data.tsv"


train_data = np.loadtxt(args.dataset_path.replace("train", "val"), delimiter='\t').astype(np.float32)[:10]
train_size = train_data.shape[0]
train_ds = tf.data.Dataset.from_tensor_slices(train_data)

val_data = np.loadtxt(args.dataset_path.replace("train", "val"), delimiter='\t').astype(np.float32)[:10]
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
for epoch in range(30000):
    # training
    dataset_epoch = train_ds.shuffle(train_size)
    dataset_epoch = dataset_epoch.batch(args.batch_size).prefetch(args.batch_size)
    epoch_loss = []
    experiment_handler.log_training()
    constraint_losses = []
    q_dot_losses = []
    q_ddot_losses = []
    grads_L0 = []
    grads_L1 = []
    grads_L2 = []
    grads_L3 = []
    for i, d in _ds('Train', dataset_epoch, train_size, epoch, args.batch_size):
        with tf.GradientTape(persistent=True) as tape:
            q_cps, t_cps = model(d)
            model_loss, constraint_loss, q_dot_loss, q_ddot_loss, q, q_dot, q_ddot, xyz, t, t_cumsum, t_loss = loss(q_cps, t_cps, d)
            L1 = constraint_loss
            L2 = q_dot_loss
            L3 = q_ddot_loss
            z_loss_abs = np.mean(np.abs(xyz[..., -1, 0] - TableConstraint.Z), axis=-1)
        grads = tape.gradient(model_loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        # grad w.r.t. bspline control points
        #grad_L0 = tape.gradient(t_loss, [t_cps])
        #grad_L1 = tape.gradient(L1, [q_cps])
        #grad_L2 = tape.gradient(L2, [q_cps, t_cps])
        #grad_L3 = tape.gradient(L3, [q_cps, t_cps])
        #grad_L0 = tf.concat([tf.reshape(tf.zeros_like(grad_L2[0]), (-1, 90)), tf.reshape(grad_L0[0], (-1, 20))], axis=-1)
        #grad_L1 = tf.concat([tf.reshape(grad_L1[0], (-1, 90)), tf.reshape(tf.zeros_like(grad_L2[1]), (-1, 20))], axis=-1)
        #grad_L2 = tf.concat([tf.reshape(grad_L2[0], (-1, 90)), tf.reshape(grad_L2[1], (-1, 20))], axis=-1)
        #grad_L3 = tf.concat([tf.reshape(grad_L3[0], (-1, 90)), tf.reshape(grad_L3[1], (-1, 20))], axis=-1)

        # grad w.r.t. nn params
        grad_L0 = tape.gradient(t_loss, model.trainable_variables)
        grad_L1 = tape.gradient(L1, model.trainable_variables)
        grad_L2 = tape.gradient(L2, model.trainable_variables)
        grad_L3 = tape.gradient(L3, model.trainable_variables)
        for i in range(len(grad_L0)):
            if grad_L0[i] is None:
                grad_L0[i] = tf.zeros_like(grad_L2[i])
            if grad_L1[i] is None:
                grad_L1[i] = tf.zeros_like(grad_L2[i])
            grad_L0[i] = tf.reshape(grad_L0[i], (-1))
            grad_L1[i] = tf.reshape(grad_L1[i], (-1))
            grad_L2[i] = tf.reshape(grad_L2[i], (-1))
            grad_L3[i] = tf.reshape(grad_L3[i], (-1))

        grad_L0 = tf.concat(grad_L0, axis=0)
        grad_L1 = tf.concat(grad_L1, axis=0)
        grad_L2 = tf.concat(grad_L2, axis=0)
        grad_L3 = tf.concat(grad_L3, axis=0)


        grads_L0.append(grad_L0)
        grads_L1.append(grad_L1)
        grads_L2.append(grad_L2)
        grads_L3.append(grad_L3)

        constraint_losses.append(constraint_loss)
        q_dot_losses.append(q_dot_loss)
        q_ddot_losses.append(q_ddot_loss)
        epoch_loss.append(model_loss)
        with tf.summary.record_if(train_step % args.log_interval == 0):
            tf.summary.scalar('metrics/model_loss', tf.reduce_mean(model_loss), step=train_step)
            tf.summary.scalar('metrics/constraint_loss', tf.reduce_mean(constraint_loss / 5), step=train_step)
            tf.summary.scalar('metrics/z_loss_abs', tf.reduce_mean(z_loss_abs), step=train_step)
            tf.summary.scalar('metrics/q_dot_loss', tf.reduce_mean(q_dot_loss / 6), step=train_step)
            tf.summary.scalar('metrics/q_ddot_loss', tf.reduce_mean(q_ddot_loss / 6), step=train_step)
            tf.summary.scalar('metrics/t', tf.reduce_mean(t), step=train_step)
        train_step += 1

    constraint_losses = tf.reduce_mean(tf.concat(constraint_losses, 0))
    q_dot_losses = tf.reduce_mean(tf.concat(q_dot_losses, 0))
    q_ddot_losses = tf.reduce_mean(tf.concat(q_ddot_losses, 0))
    #loss.alpha_update(q_dot_losses, q_ddot_losses, constraint_losses)
    grads_L0 = tf.reduce_mean(tf.stack(grads_L0, axis=0), axis=0)
    grads_L1 = tf.reduce_mean(tf.stack(grads_L1, axis=0), axis=0)
    grads_L2 = tf.reduce_mean(tf.stack(grads_L2, axis=0), axis=0)
    grads_L3 = tf.reduce_mean(tf.stack(grads_L3, axis=0), axis=0)
    #grads_L0 = tf.reduce_mean(tf.concat(grads_L0, axis=0), axis=0)
    #grads_L1 = tf.reduce_mean(tf.concat(grads_L1, axis=0), axis=0)
    #grads_L2 = tf.reduce_mean(tf.concat(grads_L2, axis=0), axis=0)
    #grads_L3 = tf.reduce_mean(tf.concat(grads_L3, axis=0), axis=0)
    loss.alpha_update_lagrange(grads_L0, grads_L1, grads_L2, grads_L3)




    #w = 10
    #if epoch % w == w - 1:
    #    plot_qs(epoch, q, q_dot, q_ddot, t_cumsum)
    #    plt.plot(xyz[0, ..., -1, 0])
    #    plt.savefig(f"z_{epoch:05d}.png")
    #    plt.clf()
    epoch_loss = tf.reduce_mean(tf.concat(epoch_loss, -1))

    with tf.summary.record_if(True):
        tf.summary.scalar('epoch/loss', epoch_loss, step=epoch)
        tf.summary.scalar('epoch/alpha_q_dot', loss.alpha_q_dot, step=epoch)
        tf.summary.scalar('epoch/alpha_q_ddot', loss.alpha_q_ddot, step=epoch)
        tf.summary.scalar('epoch/alpha_constraint', loss.alpha_constraint, step=epoch)

    # validation
    dataset_epoch = val_ds.shuffle(val_size)
    dataset_epoch = dataset_epoch.batch(args.batch_size).prefetch(args.batch_size)
    epoch_loss = []
    experiment_handler.log_validation()
    for i, d in _ds('Val', dataset_epoch, val_size, epoch, args.batch_size):
        q_cps, t_cps = model(d)
        model_loss, constraint_loss, q_dot_loss, q_ddot_loss, q, q_dot, q_ddot, xyz, t, t_cumsum, t_loss = loss(q_cps, t_cps, d)
        z_loss_abs = np.mean(np.abs(xyz[..., -1, 0] - TableConstraint.Z), axis=-1)

        epoch_loss.append(model_loss)
        with tf.summary.record_if(val_step % args.log_interval == 0):
            tf.summary.scalar('metrics/model_loss', tf.reduce_mean(model_loss), step=val_step)
            tf.summary.scalar('metrics/constraint_loss', tf.reduce_mean(constraint_loss / 5), step=val_step)
            tf.summary.scalar('metrics/z_loss_abs', tf.reduce_mean(z_loss_abs), step=val_step)
            tf.summary.scalar('metrics/q_dot_loss', tf.reduce_mean(q_dot_loss / 6), step=val_step)
            tf.summary.scalar('metrics/q_ddot_loss', tf.reduce_mean(q_ddot_loss / 6), step=val_step)
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
