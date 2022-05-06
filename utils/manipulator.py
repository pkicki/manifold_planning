import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET
from .constants import ManipulatorDimensions, TableConstraint


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
        self.lb = lb
        self.ub = ub

    def R(self, q):
        Rb = make_R(self.roll, self.pitch, self.yaw)
        Rq = make_R(q * self.axis[0], q * self.axis[1], q * self.axis[2])
        return Rb @ Rq

    def T(self, q):
        R = self.R(q)
        Rp = tf.concat([R, np.array(self.xyz)[:, tf.newaxis]], axis=-1)
        T = tf.concat([Rp, np.array([0., 0., 0., 1.])[tf.newaxis]], axis=0)
        return T

    def Rp(self, q):
        return self.R(q), self.xyz[:, np.newaxis]


#class Manipulator:
#    def __init__(self):
#        self.l1 = ManipulatorDimensions.L1
#        self.l2 = ManipulatorDimensions.L2
#        self.l3 = ManipulatorDimensions.L3
#        self.w1 = ManipulatorDimensions.W1
#        self.w2 = ManipulatorDimensions.W2
#        self.w3 = ManipulatorDimensions.W3
#        self.pos_x = ManipulatorDimensions.X
#        self.pos_y = ManipulatorDimensions.Y
#
#    def forward_kinematics(self, th):
#        th1 = th[..., 0]
#        th2 = th[..., 1]
#        th3 = th[..., 2]
#        x1 = self.pos_x * tf.ones_like(th1)
#        y1 = self.pos_y * tf.ones_like(th1)
#        x2 = self.l1 * tf.cos(th1) + x1
#        y2 = self.l1 * tf.sin(th1) + y1
#        x3 = self.l2 * tf.cos(th1 + th2) + x2
#        y3 = self.l2 * tf.sin(th1 + th2) + y2
#        x4 = self.l3 * tf.cos(th1 + th2 + th3) + x3
#        y4 = self.l3 * tf.sin(th1 + th2 + th3) + y3
#        return x4, y4, x3, y3, x2, y2, x1, y1
#
#    def plot(self, th):
#        def get_link_contour(l, w, th, x, y):
#            xy = np.stack([x, y], axis=-1)
#            p1 = np.array([0., -w / 2])
#            p2 = np.array([0., +w / 2])
#            p3 = np.array([l, +w / 2])
#            p4 = np.array([l, -w / 2])
#            d = np.stack([p1, p2, p3, p4, p1], axis=0)
#            R = Rot(th)[np.newaxis]
#            d = d[..., np.newaxis]
#            xy = xy[np.newaxis, :, np.newaxis]
#            contour = xy + R @ d
#            return contour[..., 0]
#
#        x4, y4, x3, y3, x2, y2, x1, y1 = self.forward_kinematics(th)
#        p1 = get_link_contour(self.l1, self.w1, th[..., 0], x1, y1)
#        plt.fill(p1[:, 0], p1[:, 1], 'r')
#        p2 = get_link_contour(self.l2, self.w2, np.sum(th[..., :2], axis=-1), x2, y2)
#        plt.fill(p2[:, 0], p2[:, 1], 'g')
#        p3 = get_link_contour(self.l3, self.w3, np.sum(th[..., :3], axis=-1), x3, y3)
#        plt.fill(p3[:, 0], p3[:, 1], 'b')
#        return p1, p2, p3


class Iiwa:
    def __init__(self, urdf_path):
        self.joints = Iiwa.parse_urdf(urdf_path)
        self.n_dof = len(self.joints)

    @staticmethod
    def parse_urdf(urdf_path):
        root = ET.parse(urdf_path).getroot()
        joints = []
        for joint in root.findall("joint"):
            if joint.get('name').startswith('F_joint') or joint.get('name') == 'F_striker_joint_1':
                parent = joint.find("parent").get('link')
                child = joint.find("child").get('link')
                lb = float(joint.find("limit").get("lower")) if joint.find("limit") is not None else 0.0
                ub = float(joint.find("limit").get("upper")) if joint.find("limit") is not None else 0.0
                rpy = [float(x) for x in joint.find("origin").get('rpy').split()]
                xyz = [float(x) for x in joint.find("origin").get('xyz').split()]
                axis = [float(x) for x in joint.find("axis").get('xyz').split()]
                joints.append(JointTF(parent, child, rpy, xyz, axis, lb, ub))
        return joints

    def forward_kinematics(self, q):
        #q = tf.concat([q, tf.zeros_like(q)[..., :2]], axis=-1)
        q = tf.concat([q, tf.zeros_like(q)[..., :3]], axis=-1)
        Racc = tf.eye(3, batch_shape=tf.shape(q)[:-1])
        #xyz = tf.stack([TableConstraint.XLB - 0.4, TableConstraint.YLB / 2., 0.])[:, tf.newaxis]
        xyz = tf.stack([0.0, 0.0, 0.0])[:, tf.newaxis]
        for i in range(len(tf.shape(q)) - 1):
            xyz = xyz[tf.newaxis]
        for i in range(9):
            qi = q[..., i]
            j = self.joints[i]
            R, p = j.Rp(qi)
            for i in range(len(tf.shape(q)) - 1):
                p = p[tf.newaxis]
            xyz = xyz + Racc @ p
            Racc = Racc @ R
        return xyz
