import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import numpy as np

from utils.constants import UrdfModels
from utils.dataset import _ds
from utils.execution import ExperimentHandler
from losses.ik import IKHittingLoss, IKHittingPosLoss
from models.iiwa_ik_hitting import IiwaIKHitting


physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

class args:
    batch_size = 64
    working_dir = './trainings'
    out_name = 'ik_hitting_pos_data10k_fixedtraininganddata_1em5'
    log_interval = 10
    learning_rate = 1e-5
    dataset_path = "./data/paper/ik_hitting/train/data.tsv"

train_data = np.loadtxt(args.dataset_path, delimiter='\t').astype(np.float32)
train_size = train_data.shape[0]
train_ds = tf.data.Dataset.from_tensor_slices(train_data)

val_data = np.loadtxt(args.dataset_path.replace("train", "val"), delimiter='\t').astype(np.float32)
val_size = val_data.shape[0]
val_ds = tf.data.Dataset.from_tensor_slices(val_data)

urdf_path = os.path.join(os.path.dirname(__file__), UrdfModels.striker)


opt = tf.keras.optimizers.Adam(args.learning_rate)
#loss = IKHittingLoss()
loss = IKHittingPosLoss(urdf_path)
model = IiwaIKHitting()

experiment_handler = ExperimentHandler(args.working_dir, args.out_name, args.log_interval, model, opt)

train_step = 0
val_step = 0
best_epoch_loss = 1e10
for epoch in range(3000):
    dataset_epoch = train_ds.shuffle(train_size)
    dataset_epoch = dataset_epoch.batch(args.batch_size).prefetch(args.batch_size)
    epoch_loss = []
    experiment_handler.log_training()
    for i, d in _ds('Train', dataset_epoch, train_size, epoch, args.batch_size):
        with tf.GradientTape() as tape:
            qk = model(d)
            model_loss, q_loss, q_loss_abs, *_ = loss(qk, d)
        grads = tape.gradient(model_loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))


        epoch_loss.append(model_loss)
        with tf.summary.record_if(train_step % args.log_interval == 0):
            tf.summary.scalar('metrics/model_loss', tf.reduce_mean(model_loss), step=train_step)
            tf.summary.scalar('metrics/q_loss', tf.reduce_mean(q_loss), step=train_step)
            tf.summary.scalar('metrics/q_loss_abs', tf.reduce_mean(q_loss_abs), step=train_step)
        train_step += 1

    epoch_loss = tf.reduce_mean(tf.concat(epoch_loss, -1))

    with tf.summary.record_if(True):
        tf.summary.scalar('epoch/loss', epoch_loss, step=epoch)

    # validation
    dataset_epoch = val_ds.shuffle(val_size)
    dataset_epoch = dataset_epoch.batch(args.batch_size).prefetch(args.batch_size)
    epoch_loss = []
    experiment_handler.log_validation()
    for i, d in _ds('Val', dataset_epoch, val_size, epoch, args.batch_size):
        qk = model(d)
        model_loss, q_loss, q_loss_abs, *_ = loss(qk, d)

        epoch_loss.append(model_loss)
        with tf.summary.record_if(val_step % args.log_interval == 0):
            tf.summary.scalar('metrics/model_loss', tf.reduce_mean(model_loss), step=val_step)
            tf.summary.scalar('metrics/q_loss', tf.reduce_mean(q_loss), step=val_step)
            tf.summary.scalar('metrics/q_loss_abs', tf.reduce_mean(q_loss_abs), step=val_step)
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
