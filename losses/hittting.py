from losses.feasibility import FeasibilityLoss
from utils.data import unpack_data_boundaries


class HittingLoss(FeasibilityLoss):
    def __init__(self, N, urdf_path, end_effector_constraint_distance_function, q_dot_limits, q_ddot_limits):
        super(HittingLoss, self).__init__(N, urdf_path, q_dot_limits, q_ddot_limits)
        self.end_effector_constraints_distance_function = end_effector_constraint_distance_function

    def call(self, q_cps, t_cps, data):
        q0, qd, xyth, q_dot_0, q_ddot_0, q_dot_d = unpack_data_boundaries(data, 7)

        _, q_dot_loss, q_ddot_loss, q, q_dot, q_ddot, t, t_cumsum = super().call(q_cps, t_cps, data)

        xyz = self.man.forward_kinematics(q)

        constraint_loss = self.end_effector_constraints_distance_function(xyz)

        model_loss = constraint_loss + q_dot_loss + q_ddot_loss + t
        return model_loss, constraint_loss, q_dot_loss, q_ddot_loss, q, q_dot, q_ddot, xyz, t, t_cumsum