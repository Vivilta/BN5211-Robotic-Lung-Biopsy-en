import os
import json
import numpy as np

from robot_geometry import rotation_from_z_axis


# The UR5 model used by the CoppeliaSim scene exposes revolute joints with a
# nominal +/-2*pi interval.  Keeping this contract in one module avoids the
# analytical planner and the simulator silently using different limits.
UR5_JOINT_LIMITS_RAD = np.tile(np.asarray([[-2.0 * np.pi, 2.0 * np.pi]], dtype=float), (6, 1))
UR5_JOINT_LIMIT_MARGIN_RAD = np.deg2rad(5.0)


def wrap_to_pi(angle):
    """Wrap revolute-joint differences to the shortest signed difference."""
    return (np.asarray(angle, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def joint_limit_margin_rad(joint_angles, limits=UR5_JOINT_LIMITS_RAD):
    """Return the smallest distance from a configuration to any joint limit."""
    q = np.asarray(joint_angles, dtype=float)
    if q.shape != (6,):
        raise ValueError("joint_angles must contain six values")
    limits = np.asarray(limits, dtype=float)
    if limits.shape != (6, 2):
        raise ValueError("limits must have shape (6, 2)")
    return float(np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q)))

class UR5Kinematics:
    """
    UR5 6-DOF manipulator kinematics solver using standard DH parameters.
    Units: length in mm and angles in radians.
    """
    def __init__(self):
        # Standard UR5 geometric parameters (mm).
        self.d = np.array([89.159, 0.0, 0.0, 109.15, 94.65, 82.3])
        self.a = np.array([0.0, -425.0, -392.25, 0.0, 0.0, 0.0])
        self.alpha = np.array([np.pi/2, 0.0, 0.0, np.pi/2, -np.pi/2, 0.0])

    def dh_transform(self, a, alpha, d, theta):
        """Calculate the 4x4 homogeneous transform for one link."""
        return np.array([
            [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
            [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
            [0.0,            np.sin(alpha),                np.cos(alpha),               d],
            [0.0,            0.0,                          0.0,                         1.0]
        ])

    def forward_kinematics(self, joint_angles):
        """
        Forward kinematics (FK): compute the end-effector 4x4 homogeneous
        transform from six joint angles in radians.
        """
        T = np.eye(4)
        for i in range(6):
            T_i = self.dh_transform(self.a[i], self.alpha[i], self.d[i], joint_angles[i])
            T = np.dot(T, T_i)
        return T

    def solve_ik_numerical(
        self,
        target_pose,
        initial_q=None,
        max_iters=300,
        tol=0.5,
        orientation_tol=1e-3,
        orientation_weight_mm=100.0,
        joint_limits=UR5_JOINT_LIMITS_RAD,
        joint_limit_margin_rad=UR5_JOINT_LIMIT_MARGIN_RAD,
    ):
        """
        Solve numerical inverse kinematics using damped least squares
        (Levenberg-Marquardt). target_pose is a 4x4 desired homogeneous transform.
        """
        limits = np.asarray(joint_limits, dtype=float)
        if limits.shape != (6, 2) or np.any(limits[:, 1] <= limits[:, 0]):
            raise ValueError("joint_limits must have shape (6, 2) with upper > lower")
        margin = float(joint_limit_margin_rad)
        if margin < 0 or np.any((limits[:, 1] - limits[:, 0]) <= 2.0 * margin):
            raise ValueError("joint_limit_margin_rad is incompatible with joint_limits")
        lower = limits[:, 0] + margin
        upper = limits[:, 1] - margin
        if initial_q is None:
            # Use a typical forward-bent working pose by default.
            q = np.array([0.0, -np.pi/4, np.pi/2, -np.pi/4, np.pi/2, 0.0])
        else:
            q = np.array(initial_q, dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError("initial_q must contain six finite values")
        q = np.clip(q, lower, upper)

        target_pos = target_pose[:3, 3]
        target_rot = target_pose[:3, :3]

        for it in range(max_iters):
            T_current = self.forward_kinematics(q)
            current_pos = T_current[:3, 3]
            current_rot = T_current[:3, :3]

            # Position error.
            pos_err = target_pos - current_pos

            # Orientation error (rotation-matrix logarithm / axis-angle approximation).
            rot_err_mat = np.dot(target_rot, current_rot.T)
            rot_err = 0.5 * np.array([
                rot_err_mat[2, 1] - rot_err_mat[1, 2],
                rot_err_mat[0, 2] - rot_err_mat[2, 0],
                rot_err_mat[1, 0] - rot_err_mat[0, 1]
            ])

            err = np.hstack([pos_err, rot_err * orientation_weight_mm])
            if np.linalg.norm(pos_err) <= tol and np.linalg.norm(rot_err) <= orientation_tol:
                return q, True, np.linalg.norm(pos_err)

            # Compute the Jacobian using numerical differentiation.
            J = np.zeros((6, 6))
            delta = 1e-6
            for j in range(6):
                q_plus = q.copy()
                q_plus[j] += delta
                T_plus = self.forward_kinematics(q_plus)
                dp = (T_plus[:3, 3] - current_pos) / delta
                
                # Approximate angular velocity.
                R_diff = np.dot(T_plus[:3, :3], current_rot.T)
                dw = 0.5 * np.array([
                    R_diff[2, 1] - R_diff[1, 2],
                    R_diff[0, 2] - R_diff[2, 0],
                    R_diff[1, 0] - R_diff[0, 1]
                ]) / delta
                J[:, j] = np.hstack([dp, dw * orientation_weight_mm])

            # Update joint angles using damped least squares.
            damping = 1e-2
            J_dls = np.dot(J.T, np.linalg.inv(np.dot(J, J.T) + damping**2 * np.eye(6)))
            dq = np.dot(J_dls, err)

            # Keep every numerical IK iterate inside a conservative interior
            # of the configured joint interval.  This prevents a solver from
            # converging to a mathematically valid but mechanically unusable
            # branch that is sitting on a hard stop.
            q += dq
            q = np.clip(q, lower, upper)

        final_pose = self.forward_kinematics(q)
        final_pos_err = np.linalg.norm(target_pose[:3, 3] - final_pose[:3, 3])
        final_rot = target_pose[:3, :3] @ final_pose[:3, :3].T
        final_rot_err = 0.5 * np.array([
            final_rot[2, 1] - final_rot[1, 2],
            final_rot[0, 2] - final_rot[2, 0],
            final_rot[1, 0] - final_rot[0, 1],
        ])
        converged = final_pos_err <= tol and np.linalg.norm(final_rot_err) <= orientation_tol
        return q, bool(converged), float(final_pos_err)


def build_needle_pose(needle_tip, direction_vector, needle_tip_offset_mm=0.0):
    """
    Build a 4x4 end-effector homogeneous transform from the physical needle-tip
    coordinates and insertion unit vector. The manipulator's local Z axis is
    aligned with direction_vector.
    """
    z_axis = np.asarray(direction_vector, dtype=float)
    z_axis = z_axis / np.linalg.norm(z_axis)
    rotation = rotation_from_z_axis(z_axis)

    pose = np.eye(4)
    pose[:3, :3] = rotation
    # The analytical DH chain ends at the tool flange/TCP. The needle tip is
    # offset along the tool's local Z axis, so IK targets the flange position.
    pose[:3, 3] = np.asarray(needle_tip, dtype=float) - rotation[:, 2] * float(needle_tip_offset_mm)
    return pose


def needle_tip_from_tcp_pose(tcp_pose, needle_tip_offset_mm=0.0):
    """Recover the physical needle tip from a TCP pose."""
    tcp_pose = np.asarray(tcp_pose, dtype=float)
    return tcp_pose[:3, 3] + tcp_pose[:3, 2] * float(needle_tip_offset_mm)


def main():
    print("=== BN5211: 6-DOF manipulator inverse kinematics and targeting error evaluation ===")

    # 1. Load trajectory data.
    trajectory_file = r"F:\Lung-Nodule\data\demo_nodules\planned_trajectory_0037.json"
    if not os.path.exists(trajectory_file):
        print(f"❌ Trajectory plan not found: {trajectory_file}")
        return

    with open(trajectory_file, 'r', encoding='utf-8') as f:
        plan_data = json.load(f)

    target_pt = np.array(plan_data["target_point"])
    entry_pt = np.array(plan_data["best_trajectory"]["entry_point"])
    needle_vec = np.array(plan_data["best_trajectory"]["vector"])

    print(f">> Entry point: {entry_pt}")
    print(f">> Target point: {target_pt}")
    print(f">> Insertion direction vector: {needle_vec}")

    # 2. Build the desired end-effector pose when the needle tip reaches the target.
    target_pose = build_needle_pose(needle_tip=target_pt, direction_vector=needle_vec)

    # 3. Solve inverse kinematics (IK).
    ur5 = UR5Kinematics()
    joint_angles, converged, final_err = ur5.solve_ik_numerical(target_pose)

    joint_angles_deg = np.rad2deg(joint_angles)

    print("\n================ Kinematics solution ================")
    print(f"IK convergence: {'✅ Converged' if converged else '⚠️ Did not fully converge'}")
    print("Solved joint angles (degrees):")
    for i, deg in enumerate(joint_angles_deg):
        print(f"  Joint {i+1} (Theta_{i+1}): {deg:.2f}°")

    # 4. Verify with forward kinematics and calculate targeting error (key metric in proposal section 5).
    fk_result = ur5.forward_kinematics(joint_angles)
    needle_tip_offset_mm = float(plan_data.get("needle_tip_offset_mm_for_analytical_dh", 0.0))
    simulated_tip_pos = needle_tip_from_tcp_pose(fk_result, needle_tip_offset_mm)

    # Euclidean distance.
    targeting_error = np.linalg.norm(simulated_tip_pos - target_pt)

    print("\n================ Key proposal metrics ================")
    print(f"Theoretical nodule centroid: {[round(c, 2) for c in target_pt]} mm")
    print(f"Manipulator needle-tip position: {[round(c, 2) for c in simulated_tip_pos]} mm")
    print(f"🎯 Targeting error: {targeting_error:.4f} mm")
    print("==================================================\n")

    # 5. Save results for downstream integration or visualization.
    out_kinematics_file = r"F:\Lung-Nodule\data\demo_nodules\kinematics_result_0037.json"
    result_data = {
        "joint_angles_deg": joint_angles_deg.tolist(),
        "joint_angles_rad": joint_angles.tolist(),
        "simulated_tip_pos": simulated_tip_pos.tolist(),
        "theoretical_target": target_pt.tolist(),
        "targeting_error_mm": float(targeting_error),
        "ik_converged": bool(converged)
    }
    os.makedirs(os.path.dirname(out_kinematics_file), exist_ok=True)
    with open(out_kinematics_file, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=4)
    print(f"🎉 Kinematics results saved to: {out_kinematics_file}")


if __name__ == "__main__":
    main()
