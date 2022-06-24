from time import perf_counter
import numpy as np
import tensorflow as tf
from scipy.interpolate import BSpline as BSp
from scipy.interpolate import interp1d

from manifold_planning.models.iiwa_planner import IiwaPlanner
from manifold_planning.models.iiwa_planner_boundaries import IiwaPlannerBoundaries
from manifold_planning.models.iiwa_ik_hitting import IiwaIKHitting


def load_model(path, N, n_pts_fixed_begin, bsp, bsp_t):
    model = IiwaPlanner(N, n_pts_fixed_begin, bsp, bsp_t)
    model(np.zeros([1, 30]))
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

    t = np.concatenate([np.zeros((1, 1)), t[:, :-1]], axis=-1)[0]
    si = interp1d(t, np.linspace(0., 1., 1024), axis=-1)
    n = 100
    targ = np.linspace(t[0], t[-1], n)
    s = si(targ)

    dtau_dt_bs = BSp(bspt.u, t_cps[0, :, 0], 7)
    ddtau_dtt_bs = dtau_dt_bs.derivative()
    q_bs = BSp(bsp.u, q_cps[0, :], 7)
    dq_bs = q_bs.derivative()
    ddq_bs = dq_bs.derivative()

    q = q_bs(s)
    q_dot_tau = dq_bs(s)
    q_ddot_tau = ddq_bs(s)
    dtau_dt = dtau_dt_bs(s)[..., np.newaxis]
    ddtau_dtt = ddtau_dtt_bs(s)[..., np.newaxis]
    t = targ

    q_dot = q_dot_tau * dtau_dt
    q_ddot = q_ddot_tau * dtau_dt ** 2 + ddtau_dtt * q_dot_tau * dtau_dt

    return q, q_dot, q_ddot, t
