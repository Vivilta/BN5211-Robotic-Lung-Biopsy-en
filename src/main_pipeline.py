"""Offline AI-to-robot planning pipeline.

This file prepares a checked planning contract for CoppeliaSim. It does not claim
that a robot moved until ``coppeliasim_remote.py`` reports a successful remote run.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from robot_geometry import identity_transform, transform_direction, transform_point
from robot_kinematics import UR5Kinematics, build_needle_pose, needle_tip_from_tcp_pose
from trajectory_planner import BiopsyTrajectoryPlanner
from model_3D_cbam import AttentionResNet3D


def _load_attention_model(checkpoint_path):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionResNet3D(num_classes=2)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    # Lightning stores the network below the ``model.`` prefix.
    normalized_state_dict = {
        key.removeprefix("model."): value for key, value in state_dict.items()
    }
    model.load_state_dict(normalized_state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, device


def _infer_malignancy(model, device, roi_cube_path):
    if not os.path.isfile(roi_cube_path):
        raise FileNotFoundError(f"ROI cube not found: {roi_cube_path}")
    cube = np.load(roi_cube_path)
    if cube.shape != (32, 32, 32):
        raise ValueError(f"Expected a 32x32x32 ROI, got {cube.shape}")
    if not np.isfinite(cube).all():
        raise ValueError("ROI cube contains NaN or infinite values")
    tensor = torch.from_numpy(cube.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        if tuple(logits.shape) != (1, 2):
            raise ValueError(f"Expected two-class logits with shape (1, 2), got {tuple(logits.shape)}")
        probabilities = torch.softmax(logits, dim=1)
    return float(probabilities[0, 1].item())


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _load_transform(path):
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    transform = np.asarray(value, dtype=float)
    if transform.shape != (4, 4):
        raise ValueError(f"Patient-to-robot transform must be a 4x4 JSON matrix, got {transform.shape}")
    return transform


def run_end_to_end_biopsy_pipeline(
    ckpt_path,
    roi_cube_path,
    target_coords_mm,
    obstacles_json_path,
    malignancy_threshold=0.45,
    output_plan_path=None,
    patient_to_robot_base=None,
    needle_tip_offset_mm=0.0,
    force_robot_demo=False,
    patient_to_robot_transform_confirmed=False,
    patient_to_robot_transform_path=None,
    target_roi_match_confirmed=False,
    approach_distance_mm=30.0,
    planning_mode="multi_direction",
    entry_distances_mm=(60.0, 80.0, 100.0),
    angular_step_deg=15.0,
    max_direction_deviation_deg=150.0,
    ik_initial_q=None,
    needle_radius_mm=4.0,
    planning_safety_margin_mm=2.0,
):
    """Run inference, straight-line planning, and analytical FK/IK checks.

    ``patient_to_robot_base`` maps CT physical coordinates (mm) into the UR5 base
    frame (mm). Identity is accepted for a software demonstration only; it is not
    a patient-to-robot calibration.
    """
    target_patient = np.asarray(target_coords_mm, dtype=float)
    if target_patient.shape != (3,):
        raise ValueError("target_coords_mm must contain exactly three values")
    if approach_distance_mm <= 0:
        raise ValueError("approach_distance_mm must be positive")
    if patient_to_robot_transform_path is not None:
        patient_to_robot_base = _load_transform(patient_to_robot_transform_path)
        patient_to_robot_transform_confirmed = True
    elif patient_to_robot_base is None:
        patient_to_robot_base = identity_transform()
    else:
        patient_to_robot_base = np.asarray(patient_to_robot_base, dtype=float)
    if patient_to_robot_base.shape != (4, 4):
        raise ValueError("patient_to_robot_base must be a 4x4 matrix")

    print("=" * 72)
    print("BN5211 AI-assisted robotic biopsy planning")
    print("=" * 72)

    model, device = _load_attention_model(ckpt_path)
    malignant_probability = _infer_malignancy(model, device, roi_cube_path)
    print(f"[AI] ROI: {os.path.basename(roi_cube_path)}")
    print(f"[AI] Malignancy probability: {malignant_probability:.2%}")
    print(f"[AI] Decision threshold: {malignancy_threshold:.2f}")

    result = {
        "coordinate_system": {
            "patient": "CT physical coordinates in mm",
            "robot_base": "UR5 base coordinates in mm; CoppeliaSim receives metres",
            "patient_to_robot_base": patient_to_robot_base.tolist(),
        },
        "roi_cube_path": os.path.abspath(roi_cube_path),
        "checkpoint_path": os.path.abspath(ckpt_path),
        "patient_to_robot_transform_path": (
            os.path.abspath(patient_to_robot_transform_path)
            if patient_to_robot_transform_path else None
        ),
        "target_point_patient_mm": target_patient.tolist(),
        "planning_geometry": {
            "needle_length_mm": float(needle_tip_offset_mm),
            "needle_radius_mm": float(needle_radius_mm),
            "additional_aabb_clearance_mm": float(planning_safety_margin_mm),
            "approach_distance_mm": float(approach_distance_mm),
            "checked_segment": "full needle shaft swept from pre-approach through target",
        },
        "target_roi_match_confirmed": bool(target_roi_match_confirmed),
        "target_roi_binding": {
            "roi_path": os.path.abspath(roi_cube_path),
            "target_point_patient_mm": target_patient.tolist(),
            "confirmed_by_operator": bool(target_roi_match_confirmed),
        },
        "malignancy_probability": malignant_probability,
        "malignancy_threshold": malignancy_threshold,
        "robot_stage_forced": bool(force_robot_demo),
        "patient_to_robot_transform_confirmed": bool(patient_to_robot_transform_confirmed),
    }

    if malignant_probability < malignancy_threshold and not force_robot_demo:
        result["decision"] = "low_risk_no_robot_motion"
        print("[AI] Below threshold; robot planning was not started.")
        if output_plan_path:
            _write_json(output_plan_path, result)
        return result

    if force_robot_demo and malignant_probability < malignancy_threshold:
        print("[AI] Force-demo flag enabled; continuing for software integration testing.")
    else:
        print("[AI] Above threshold; continuing to planning.")

    if not os.path.isfile(obstacles_json_path):
        raise FileNotFoundError(
            "Obstacle JSON is required for planning. Generate it from the matching CT "
            f"before continuing: {obstacles_json_path}"
        )

    if not target_roi_match_confirmed:
        result["decision"] = "target_roi_match_unconfirmed"
        print(
            "[SAFETY] Robot planning stopped: explicitly confirm that this ROI "
            "belongs to the supplied target coordinates."
        )
        if output_plan_path:
            _write_json(output_plan_path, result)
        return result

    planner = BiopsyTrajectoryPlanner(
        obstacles_json_path,
        safety_margin_mm=planning_safety_margin_mm,
        needle_radius_mm=needle_radius_mm,
    )
    if planning_mode == "multi_direction":
        best_path = planner.plan_multidirectional(
            target_patient,
            entry_distances_mm=entry_distances_mm,
            angular_step_deg=angular_step_deg,
            preferred_direction=[0.0, -1.0, 0.0],
            max_direction_deviation_deg=max_direction_deviation_deg,
            approach_distance_mm=approach_distance_mm,
            needle_length_mm=needle_tip_offset_mm,
        )
    elif planning_mode == "fixed_y":
        best_path = planner.plan_trajectory(
            target_patient,
            approach_distance_mm=approach_distance_mm,
            needle_length_mm=needle_tip_offset_mm,
        )
    else:
        raise ValueError("planning_mode must be 'multi_direction' or 'fixed_y'")
    if best_path is None:
        result["decision"] = "no_collision_free_path"
        if output_plan_path:
            _write_json(output_plan_path, result)
        return result

    entry_patient = np.asarray(best_path["entry_point"], dtype=float)
    axis_patient = np.asarray(best_path["vector"], dtype=float)
    approach_patient = entry_patient - axis_patient * float(approach_distance_mm)
    target_robot = transform_point(patient_to_robot_base, target_patient)
    entry_robot = transform_point(patient_to_robot_base, entry_patient)
    approach_robot = transform_point(patient_to_robot_base, approach_patient)
    axis_robot = transform_direction(patient_to_robot_base, axis_patient)

    target_pose_robot = build_needle_pose(
        needle_tip=target_robot,
        direction_vector=axis_robot,
        needle_tip_offset_mm=needle_tip_offset_mm,
    )
    ur5 = UR5Kinematics()
    joint_angles, ik_converged, ik_position_error = ur5.solve_ik_numerical(
        target_pose_robot,
        initial_q=ik_initial_q,
    )
    fk_pose = ur5.forward_kinematics(joint_angles)
    fk_tip = needle_tip_from_tcp_pose(fk_pose, needle_tip_offset_mm)
    targeting_error = float(np.linalg.norm(fk_tip - target_robot))

    print(f"[PLAN] Entry point: {entry_patient.tolist()} mm (patient frame)")
    print(f"[PLAN] Target point: {target_patient.tolist()} mm (patient frame)")
    print(f"[PLAN] Needle depth: {best_path['path_length']:.2f} mm")
    print(f"[IK] Converged: {ik_converged}; position error: {ik_position_error:.3f} mm")
    print(f"[IK] Analytical needle-tip error: {targeting_error:.3f} mm")

    if not ik_converged or targeting_error > 0.5:
        result.update(
            {
                "decision": "analytical_ik_failed",
                "analytical_ik": {
                    "joint_angles_rad": joint_angles.tolist(),
                    "converged": bool(ik_converged),
                    "position_error_mm": float(ik_position_error),
                    "needle_tip_error_mm": targeting_error,
                },
            }
        )
        if output_plan_path:
            _write_json(output_plan_path, result)
        return result

    result.update(
        {
            "decision": "robot_plan_ready",
            "entry_point_patient_mm": entry_patient.tolist(),
            "approach_point_patient_mm": approach_patient.tolist(),
            "approach_distance_mm": float(approach_distance_mm),
            "needle_axis_patient": axis_patient.tolist(),
            "target_point_robot_base_mm": target_robot.tolist(),
            "entry_point_robot_base_mm": entry_robot.tolist(),
            "approach_point_robot_base_mm": approach_robot.tolist(),
            "needle_axis_robot_base": axis_robot.tolist(),
            "needle_tip_offset_mm_for_analytical_dh": float(needle_tip_offset_mm),
            "best_trajectory": best_path,
            "planning_mode": planning_mode,
            "analytical_ik": {
                "joint_angles_rad": joint_angles.tolist(),
                "joint_angles_deg": np.rad2deg(joint_angles).tolist(),
                "converged": bool(ik_converged),
                "position_error_mm": float(ik_position_error),
                "needle_tip_error_mm": targeting_error,
            },
            "coppeliasim_contract": {
                "ik_tip_must_be_at_physical_needle_tip": True,
                "ik_target_receives_robot_base_pose": True,
                "units_for_target_and_entry": "mm in this JSON; convert to m at API boundary",
                "execution_phases": [
                    "initial_tcp_to_approach",
                    "approach_to_entry",
                    "entry_to_target_insertion",
                ],
            },
        }
    )
    if output_plan_path:
        _write_json(output_plan_path, result)
        print(f"[PLAN] Saved: {output_plan_path}")
    return result


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="models/checkpoints_attention/3d_resnet_cbam-epoch=29-val_acc=0.84.ckpt")
    parser.add_argument("--roi", default="data/registered_rois/LIDC-IDRI-0037.npy")
    parser.add_argument("--obstacles", default="data/demo_nodules/rib_obstacles_0037.json")
    parser.add_argument("--output-plan", default="data/demo_nodules/planned_trajectory_0037.json")
    parser.add_argument(
        "--target-mm",
        nargs=3,
        type=float,
        default=[118.275, 20.625, -163.50],
        help="Target in CT physical millimetres; for 0037 this is the ROI centre from the matching DICOM series",
    )
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--needle-tip-offset-mm", type=float, default=0.0)
    parser.add_argument(
        "--approach-distance-mm",
        type=float,
        default=30.0,
        help="Distance outside the planned entry point for the pre-insertion waypoint",
    )
    parser.add_argument(
        "--planning-mode",
        choices=("multi_direction", "fixed_y"),
        default="multi_direction",
        help="Use generalized radial direction search or the legacy fixed-Y search",
    )
    parser.add_argument(
        "--entry-distances-mm",
        nargs="+",
        type=float,
        default=[60.0, 80.0, 100.0],
        help="Radial entry distances used by multi_direction planning",
    )
    parser.add_argument("--angular-step-deg", type=float, default=15.0)
    parser.add_argument("--force-robot-demo", action="store_true")
    parser.add_argument("--patient-to-robot-transform", help="JSON file containing a 4x4 CT-mm to UR5-base-mm transform")
    parser.add_argument(
        "--confirm-coincident-demo-frame",
        action="store_true",
        help="Explicitly allow identity CT-to-robot registration for a non-clinical demo scene",
    )
    parser.add_argument(
        "--confirm-target-roi-match",
        action="store_true",
        help="Explicitly confirm that the supplied ROI and target coordinates are the same case",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_end_to_end_biopsy_pipeline(
        ckpt_path=args.checkpoint,
        roi_cube_path=args.roi,
        target_coords_mm=args.target_mm,
        obstacles_json_path=args.obstacles,
        malignancy_threshold=args.threshold,
        output_plan_path=args.output_plan,
        needle_tip_offset_mm=args.needle_tip_offset_mm,
        force_robot_demo=args.force_robot_demo,
        patient_to_robot_transform_confirmed=args.confirm_coincident_demo_frame,
        patient_to_robot_transform_path=args.patient_to_robot_transform,
        target_roi_match_confirmed=args.confirm_target_roi_match,
        approach_distance_mm=args.approach_distance_mm,
        planning_mode=args.planning_mode,
        entry_distances_mm=tuple(args.entry_distances_mm),
        angular_step_deg=args.angular_step_deg,
    )
