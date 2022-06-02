import subprocess
from time import perf_counter

import numpy as np
import pinocchio as pino

from utils.constants import Limits
import matplotlib.pyplot as plt

import utils.hpo_opt as hpoo
import utils.hpo as hpo

#import tensorflow as tf

def get_hitting_configuration(x, y, th, q0=None):
    if q0 is None:
        q0 = [0., 0.7135, 0., -0.5025, 0., 1.9257, 0.]
    s = hpoo.optimize(x, y, np.cos(th), np.sin(th), q0)
    #s = hpo.optimize(x, y, np.cos(th), np.sin(th), q0)
    q = s[:7]
    q_dot = np.array(s[7:14])
    mul = np.max(np.abs(q_dot[:6]) / Limits.q_dot)
    q_dot = q_dot / mul
    mag = s[-2]
    v = s[-1]
    #print()
    #print("MAG:", s[-1])
    return q, q_dot.tolist(), mag, v

def get_hitting_configuration_opt(x, y, th, q0=None):
    if q0 is None:
        q0 = [0., 0.7135, 0., -0.5025, 0., 1.9257, 0.]
    s = hpoo.optimize(x, y, np.cos(th), np.sin(th), q0)
    if not s:
        return None, None, None, None
    q = s[:7]
    q_dot = np.array(s[7:14])
    mul = np.max(np.abs(q_dot[:6]) / Limits.q_dot)
    q_dot = q_dot / mul
    mag = s[-2]
    v = s[-1]
    return q, q_dot.tolist(), mag, v

if __name__ == "__main__":
    q0 = [0., 0.7135, 0., -0.5025, 0., 1.9257, 0.]
    urdf_path = "/home/piotr/b8/rl/3dof_planning/iiwa_striker.urdf"
    model = pino.buildModelFromUrdf(urdf_path)
    data = model.createData()
    #x = 1.0261847865032634
    x = 0.9
    #x = 1.3
    #y = 0.1638526734661639
    #y = 0.16384
    #y = 0.1638259
    #y = 0.0
    y = -0.35
    y = -0.3
    #y = -0.25
    #th = -0.1114713380540327
    #th = 0.1114713380540327
    th = 0.0
    #q0[0] = y / 2
    print(np.cos(th), np.sin(th))
    q, q_dot, mag, v = get_hitting_configuration(x, y, th, q0)
    q1, q_dot1, mag1, v1 = get_hitting_configuration_opt(x, y, th, q0)
    q = np.concatenate([np.array(q), np.zeros(2)], axis=-1)
    pino.forwardKinematics(model, data, q)
    xyz_pino = data.oMi[-1].translation
    J = pino.computeJointJacobian(model, data, q, 6)
    print()
    print("J:", J.shape)
    print()
    print(xyz_pino)
    print()
    print("Q:", q)
    #print(q_dot)
    q0 = [0., 0.7135, 0., -0.5025, 0., 1.9257, 0.]
    print(np.sum(np.abs(np.array(q0)[:6] - np.array(q)[:6]) / Limits.q_dot))
    assert False

    X, Y = 15, 51
    x = 1.
    th = 0.1
    #xs = np.linspace(0.9, 1.3, X)
    ths = np.linspace(-0.2, 0.2, X)
    xs = [1.1]
    ys = np.linspace(-0.4, 0.4, Y)
    qi0s = []
    times = []
    #q0 = [-0.1199166764301418, 0.925443585720224, -0.09106010491374625, -0.3463669629041281, -0.08702749577145877, 1.3843531121676345, 0.0]
    vdiffs = []
    vmaxdiffs = []
    q0diffs = []
    qdiffs = []
    q0s = []
    qs = []
    vs = []
    vs_opt = []
    mags = []
    mags_opt = []
    for x in xs:
        for y in ys:
            for th in ths:
                t0 = perf_counter()
                #q0[0] = y / 2

                #q0[0] = y * 3 / 4
                #q0[3] = 0.5 * (x - 0.9) + 0.1
                #q0[2] = y / 4
                #q0[4] = y / 4
                #q0[6] = q0[6] + y / 4

                q0s.append(q0.copy())
                q, q_dot, mag, v = get_hitting_configuration(x, y, th, q0)
                q_, q_dot_, mag_opt, v_opt = get_hitting_configuration_opt(x, y, th, q0)
                qs.append(q)
                #q0 = q
                t1 = perf_counter()
                times.append(t1 - t0)
                qi0s.append(q[0])
                vdiffs.append(np.sum(np.abs(np.array(q0) - np.array(q))[:6] / Limits.q_dot))
                vmaxdiffs.append(np.max(np.abs(np.array(q0) - np.array(q))[:6] / Limits.q_dot))
                q0diffs.append(np.abs(np.array(q0) - np.array(q))[0])
                qdiffs.append(np.abs(np.array(q0) - np.array(q))[:6])
                vs.append(v)
                vs_opt.append(v_opt)
                mags.append(mag)
                mags_opt.append(mag_opt)
    print()
    print("MEAN T: ", np.mean(times))
    print(np.std(times))
    print(np.min(times))
    print("MAX T: ", np.max(times))
    print("MAX Q0DIFF: ", np.max(q0diffs))
    q0s = np.array(q0s)
    qs = np.array(qs)

    w = np.array(mags_opt) - np.array(mags)

    suffix = "bestopt_th01"
    #suffix = "bound015_noh"
    np.savetxt(f"vels_{suffix}.tsv", np.array(vs), delimiter='\t', fmt='%.4f')
    np.savetxt(f"mags_{suffix}.tsv", np.array(mags), delimiter='\t', fmt='%.4f')

    vdiffs = np.array(vdiffs)
    q0diffs = np.array(q0diffs)
    qdiffs = np.array(qdiffs)
    vdiffs = np.reshape(np.reshape(vdiffs, (X, Y)).T, -1)
    q0diffs = np.reshape(np.reshape(q0diffs, (X, Y)).T, -1)
    qdiffs = np.reshape(np.transpose(np.reshape(qdiffs, (X, Y, 6)), (1, 0, 2)), (-1, 6))
    #x, y = np.meshgrid(xs, ys)
    y, th = np.meshgrid(ys, ths)
    #x = np.reshape(x, -1)
    y = np.reshape(np.reshape(y, (X, Y)).T, -1)
    th = np.reshape(np.reshape(th, (X, Y)).T, -1)
    plt.scatter(y, th, c=vdiffs)
    #plt.scatter(x, y, c=vdiffs)
    plt.colorbar()
    plt.show()
    #plt.scatter(x, y, c=q0diffs)
    #plt.colorbar()
    #plt.show()
    for i in range(6):
        plt.subplot(231 + i)
        plt.scatter(y, th, c=qdiffs[..., i])
        #plt.scatter(x, y, c=qdiffs[..., i])
        plt.colorbar()
    plt.show()


    #plt.subplot(331)
    #plt.plot(ys, qi0s)
    #plt.plot(ys, vdiffs)
    #plt.plot(ys, vmaxdiffs)
    #for i in range(6):
    #    plt.subplot(332 + i)
    #    plt.plot(ys, qs[:, i])
    #plt.show()

