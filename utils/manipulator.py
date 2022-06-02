import numpy as np
import tensorflow as tf
import pinocchio as pino
import xml.etree.ElementTree as ET


def make_R(roll, pitch, yaw):
    o = tf.ones_like(roll)
    z = tf.zeros_like(roll)
    Rx1 = tf.stack([o, z, z], axis=-1)
    Rx2 = tf.stack([z, tf.cos(roll), -tf.sin(roll)], axis=-1)
    Rx3 = tf.stack([z, tf.sin(roll), tf.cos(roll)], axis=-1)
    Rx = tf.stack([Rx1, Rx2, Rx3], axis=-2)
    Ry1 = tf.stack([tf.cos(pitch), z, tf.sin(pitch)], axis=-1)
    Ry2 = tf.stack([z, o, z], axis=-1)
    Ry3 = tf.stack([-tf.sin(pitch), z, tf.cos(pitch)], axis=-1)
    Ry = tf.stack([Ry1, Ry2, Ry3], axis=-2)
    Rz1 = tf.stack([tf.cos(yaw), -tf.sin(yaw), z], axis=-1)
    Rz2 = tf.stack([tf.sin(yaw), tf.cos(yaw), z], axis=-1)
    Rz3 = tf.stack([z, z, o], axis=-1)
    Rz = tf.stack([Rz1, Rz2, Rz3], axis=-2)
    return Rz @ Ry @ Rx


class JointTF:
    def __init__(self, parent, child, rpy, xyz, axis, lb, ub):
        self.parent = parent
        self.child = child
        self.rpy = rpy
        self.roll = rpy[0]
        self.pitch = rpy[1]
        self.yaw = rpy[2]
        self.xyz = np.array(xyz)
        self.axis = axis
        self.fixed = self.axis is None
        self.lb = lb
        self.ub = ub

    def R(self, q):
        Rb = make_R(self.roll, self.pitch, self.yaw)
        if self.axis is None:
            return Rb
        Rq = make_R(q * self.axis[0], q * self.axis[1], q * self.axis[2])
        return Rb @ Rq

    def T(self, q):
        R = self.R(q)
        Rp = tf.concat([R, np.array(self.xyz)[:, tf.newaxis]], axis=-1)
        T = tf.concat([Rp, np.array([0., 0., 0., 1.])[tf.newaxis]], axis=0)
        return T

    def Rp(self, q):
        return self.R(q), self.xyz[:, np.newaxis]


class Iiwa:
    def __init__(self, urdf_path):
        self.joints = Iiwa.parse_urdf(urdf_path)
        self.n_dof = len(self.joints)

    @staticmethod
    def parse_urdf(urdf_path):
        root = ET.parse(urdf_path).getroot()
        joints = []
        for joint in root.findall("joint"):
            parent = joint.find("parent").get('link')
            child = joint.find("child").get('link')
            lb = float(joint.find("limit").get("lower")) if joint.find("limit") is not None else 0.0
            ub = float(joint.find("limit").get("upper")) if joint.find("limit") is not None else 0.0
            rpy = [float(x) for x in joint.find("origin").get('rpy').split()]
            xyz = [float(x) for x in joint.find("origin").get('xyz').split()]
            axis = joint.find("axis")
            if axis is not None:
                axis = [float(x) for x in axis.get('xyz').split()]
            joints.append(JointTF(parent, child, rpy, xyz, axis, lb, ub))
            # end at striker_tip
            if joints[-1].child.endswith("striker_tip"):
                break
        return joints

    def forward_kinematics(self, q):
        q = tf.concat([q, tf.zeros_like(q)[..., :7 - q.shape[-1]]], axis=-1)
        qs = []
        qidx = 0
        for i in range(self.n_dof):
            if self.joints[i].fixed:
                qs.append(tf.zeros_like(q)[..., 0])
            else:
                qs.append(q[..., qidx])
                qidx += 1

        q = tf.stack(qs, axis=-1)
        q = tf.cast(q, tf.float32)
        Racc = tf.eye(3, batch_shape=tf.shape(q)[:-1])
        xyz = tf.stack([0.0, 0.0, 0.0])[:, tf.newaxis]
        for i in range(len(tf.shape(q)) - 1):
            xyz = xyz[tf.newaxis]
        for i in range(self.n_dof):
            qi = q[..., i]
            j = self.joints[i]
            R, p = j.Rp(qi)
            for i in range(len(tf.shape(q)) - 1):
                p = p[tf.newaxis]
            xyz = xyz + Racc @ p
            Racc = Racc @ R
        return xyz


if __name__ == "__main__":
    urdf_path = "/airhockey/manifold_planning/iiwa_striker_new.urdf"
    pino_model = pino.buildModelFromUrdf(urdf_path)
    pino_data = pino_model.createData()

    #qk = np.zeros(6)
    qk = np.array([0.2, 0.2, 0.5, 0.9, 0., 0.])
    q = np.concatenate([qk, np.zeros(3)], axis=-1)
    pino.forwardKinematics(pino_model, pino_data, q)
    xyz_pino = pino_data.oMi[-1].translation

    man = Iiwa(urdf_path)
    xyz = man.forward_kinematics(qk)
    print(xyz_pino)
    print(xyz.numpy()[..., 0])
