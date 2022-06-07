from time import perf_counter
import numpy as np
import tensorflow as tf

from manifold_planning.models.iiwa_planner import IiwaPlanner
from manifold_planning.models.iiwa_planner_boundaries import IiwaPlannerBoundaries
from manifold_planning.models.iiwa_ik_hitting import IiwaIKHitting


def load_model(path, N):
    model = IiwaPlanner(N)
    model(np.zeros([1, 23]))
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(path).expect_partial()
    return model


def load_model_boundaries(path, N, n_pts_fixed_begin, n_pts_fixed_end, bsp, bsp_t):
    model = IiwaPlannerBoundaries(N, n_pts_fixed_begin, n_pts_fixed_end, bsp, bsp_t)
    model(np.zeros([1, 38]))
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(path).expect_partial()
    return model


def load_model_hpo(path):
    model = IiwaIKHitting()
    model(np.zeros([1, 17]))
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(path).expect_partial()
    return model


def model_inference(model, data, bsp, bspt):
    q_cps, t_cps = model(data)
    q = bsp.N @ q_cps
    q_dot_tau = bsp.dN @ q_cps
    q_ddot_tau = bsp.ddN @ q_cps

    dtau_dt = bspt.N @ t_cps
    ddtau_dtt = bspt.dN @ t_cps

    q_dot = q_dot_tau * dtau_dt
    q_ddot = q_ddot_tau * dtau_dt ** 2 + ddtau_dtt * q_dot_tau * dtau_dt

    ts = 1. / dtau_dt[..., 0] / 1024.
    t = tf.cumsum(np.concatenate([np.zeros_like(ts[..., :1]), ts[..., :-1]], axis=-1), axis=-1)

    return q.numpy()[0], q_dot.numpy()[0], q_ddot.numpy()[0], t.numpy()[0]
