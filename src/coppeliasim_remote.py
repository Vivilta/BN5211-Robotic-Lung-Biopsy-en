"""Execute a planned needle path in CoppeliaSim through the ZeroMQ Remote API.

The scene must contain a UR5, an IK tip dummy located at the physical needle tip,
an IK target dummy, and collision collections. The exact object paths are command
line arguments because CoppeliaSim model aliases vary between installations.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from robot_geometry import rotation_from_z_axis, rotation_to_quaternion_xyzw


DEFAULT_INITIAL_JOINT_ANGLES_RAD = (0.0, -math.pi / 2.0, 0.0, -math.pi / 2.0, 0.0, 0.0)


def _load_plan(path):
    with open(path, "r", encoding="utf-8") as handle:
        plan = json.load(handle)
    required = (
        "entry_point_robot_base_mm",
        "target_point_robot_base_mm",
        "needle_axis_robot_base",
        "target_roi_match_confirmed",
    )
    missing = [key for key in required if key not in plan]
    if missing:
        raise ValueError(f"Planning JSON is missing required keys: {', '.join(missing)}")
    if plan.get("decision") != "robot_plan_ready":
        raise ValueError(f"Planning JSON is not executable: decision={plan.get('decision')!r}")
    if not plan["target_roi_match_confirmed"]:
        raise ValueError(
            "Planning JSON does not confirm that the ROI and target belong to the same case"
        )
    return plan


def _object_handle(sim, path):
    try:
        return sim.getObject(path)
    except Exception as exc:
        raise RuntimeError(f"CoppeliaSim object not found: {path}") from exc


def _ensure_collection(sim, name, root_handle=None, object_handles=None):
    """Return a runtime collection, recreating it after a scene reload if needed."""
    collection_name = str(name).lstrip("/")
    try:
        return sim.getCollectionHandle(collection_name)
    except Exception:
        collection = sim.createCollection(0)
        sim.setCollectionName(collection, collection_name)
        if root_handle is not None:
            sim.addItemToCollection(collection, sim.handle_tree, root_handle, 0)
        for handle in object_handles or []:
            sim.addItemToCollection(collection, sim.handle_single, handle, 0)
        return collection


def _patient_obstacle_handles(sim):
    handles = []
    for handle in sim.getObjectsInTree(sim.handle_scene, sim.object_shape_type, 0):
        alias = sim.getObjectAlias(handle, 4)
        if alias.startswith("patientObstacle_"):
            handles.append(handle)
    return handles


def _joint_handles(sim, base_handle, paths):
    if paths:
        handles = [_object_handle(sim, path) for path in paths]
    else:
        # ``options`` is a bit mask, not an object handle.  With CoppeliaSim
        # 4.10, passing handle_all here returns only the root joint.
        handles = sim.getObjectsInTree(base_handle, sim.object_joint_type, 0)
        if len(handles) != 6:
            raise RuntimeError(
                "Automatic joint discovery did not return exactly six joints. "
                "Pass six --joint-path arguments in kinematic order."
            )
    if len(handles) != 6:
        raise RuntimeError(f"Expected six UR5 joints, got {len(handles)}")
    return handles


def _target_pose_mm(point_mm, axis):
    rotation = rotation_from_z_axis(axis)
    quaternion = rotation_to_quaternion_xyzw(rotation)
    point_m = np.asarray(point_mm, dtype=float) / 1000.0
    return point_m.tolist() + quaternion.tolist()


def _ik_status_ok(sim_ik, status):
    if isinstance(status, (list, tuple)):
        status = status[0]
    success = getattr(sim_ik, "result_success", 0)
    return status == success or status == 0


def _polyline_samples(points, total_steps):
    """Sample a waypoint polyline with exactly ``total_steps`` points.

    The first three waypoints are optional in practice: if two adjacent points
    coincide, that segment receives no intervals. This keeps the execution
    contract useful for old plans while making the phase boundaries explicit.
    """
    points = [np.asarray(point, dtype=float) for point in points]
    if any(point.shape != (3,) for point in points):
        raise ValueError("Every waypoint must contain exactly three coordinates")
    if total_steps < 2:
        raise ValueError("total_steps must be at least 2")

    segment_lengths = np.array(
        [np.linalg.norm(points[index + 1] - points[index]) for index in range(len(points) - 1)],
        dtype=float,
    )
    active = np.flatnonzero(segment_lengths > 1e-9)
    if len(active) == 0:
        return [points[0].copy()], ["stationary"]

    interval_budget = total_steps - 1
    if interval_budget < len(active):
        raise ValueError(
            f"total_steps={total_steps} is too small for {len(active)} non-zero waypoint segments"
        )

    intervals = np.zeros(len(segment_lengths), dtype=int)
    intervals[active] = 1
    remaining = interval_budget - len(active)
    proportions = segment_lengths[active] / segment_lengths[active].sum()
    extra = np.floor(remaining * proportions).astype(int)
    intervals[active] += extra
    remainder = remaining - int(extra.sum())
    order = active[np.argsort(-segment_lengths[active])]
    for index in range(remainder):
        intervals[order[index % len(order)]] += 1

    samples = [points[0].copy()]
    phases = ["initial"]
    phase_names = ("approach", "approach_to_entry", "insertion")
    for index, interval_count in enumerate(intervals):
        if interval_count == 0:
            continue
        segment = np.linspace(points[index], points[index + 1], interval_count + 1)[1:]
        phase = phase_names[index] if index < len(phase_names) else f"segment_{index}"
        samples.extend(segment)
        phases.extend([phase] * interval_count)
    return samples, phases


def run_remote_simulation(
    plan_path,
    scene_path=None,
    load_scene=False,
    host="localhost",
    port=23000,
    robot_path="/UR5",
    ik_tip_path="/UR5/ikTip",
    ik_target_path="/UR5/ikTarget",
    joint_paths=None,
    robot_collision_collection=None,
    environment_collision_collection=None,
    steps=100,
    allow_no_collision_check=False,
    kinematic_only=False,
    initial_joint_angles_rad=None,
    frame_delay_s=0.0,
    output_path="data/demo_nodules/coppeliasim_run_0037.json",
):
    """Run approach and insertion phases and return measured simulation metrics.

    ``kinematic_only`` is the supported mode for the stock UR5 model: it keeps
    the scene stopped, switches its joints to kinematic mode, and lets simIK
    write joint positions without a dynamic controller overwriting them.
    """
    plan = _load_plan(plan_path)
    if not plan.get("patient_to_robot_transform_confirmed", False):
        raise RuntimeError(
            "Patient CT coordinates have not been registered to the UR5 base. "
            "Do not execute an uncalibrated plan; provide a measured 4x4 transform "
            "or mark a deliberately coincident demo frame in a separate demo plan."
        )
    if scene_path and not os.path.isfile(scene_path):
        raise FileNotFoundError(f"CoppeliaSim scene not found: {scene_path}")
    if load_scene and not scene_path:
        raise ValueError("--load-scene requires --scene")
    if steps < 2:
        raise ValueError("steps must be at least 2")
    if frame_delay_s < 0:
        raise ValueError("frame_delay_s must be non-negative")

    try:
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import coppeliasim_zmqremoteapi_client. Install "
            "coppeliasim-zmqremoteapi-client or add CoppeliaSim's bundled Python client to PYTHONPATH."
        ) from exc

    client = RemoteAPIClient(host=host, port=port)
    sim = client.require("sim")
    sim_ik = client.require("simIK")

    if load_scene:
        sim.loadScene(os.path.abspath(scene_path))

    base_handle = _object_handle(sim, robot_path)
    tip_handle = _object_handle(sim, ik_tip_path)
    target_handle = _object_handle(sim, ik_target_path)
    joints = _joint_handles(sim, base_handle, joint_paths)

    collision_pair = None
    if robot_collision_collection and environment_collision_collection:
        collision_pair = (
            _ensure_collection(sim, robot_collision_collection, root_handle=base_handle),
            _ensure_collection(
                sim,
                environment_collision_collection,
                object_handles=_patient_obstacle_handles(sim),
            ),
        )
    elif not allow_no_collision_check:
        raise RuntimeError(
            "Collision collections are required. Pass --robot-collision-collection and "
            "--environment-collision-collection, or explicitly use --allow-no-collision-check."
        )

    if kinematic_only:
        initial_joint_angles_rad = (
            DEFAULT_INITIAL_JOINT_ANGLES_RAD
            if initial_joint_angles_rad is None
            else tuple(initial_joint_angles_rad)
        )
        if len(initial_joint_angles_rad) != 6:
            raise ValueError("initial_joint_angles_rad must contain exactly six values")
        for joint in joints:
            sim.setJointMode(joint, sim.jointmode_kinematic, 0)
        for joint, angle in zip(joints, initial_joint_angles_rad):
            sim.setJointPosition(joint, float(angle))

    ik_env = sim_ik.createEnvironment()
    ik_group = sim_ik.createGroup(ik_env)
    sim_ik.setGroupCalculation(ik_env, ik_group, sim_ik.method_damped_least_squares, 0.1, 100)
    _, sim_to_ik, _ = sim_ik.addElementFromScene(
        ik_env,
        ik_group,
        base_handle,
        tip_handle,
        target_handle,
        sim_ik.constraint_pose,
    )
    for joint in joints:
        ik_joint = sim_to_ik.get(joint)
        if ik_joint is not None:
            sim_ik.setJointMode(ik_env, ik_joint, sim_ik.jointmode_ik)

    entry = np.asarray(plan["entry_point_robot_base_mm"], dtype=float)
    target = np.asarray(plan["target_point_robot_base_mm"], dtype=float)
    axis = np.asarray(plan["needle_axis_robot_base"], dtype=float)
    approach = np.asarray(plan.get("approach_point_robot_base_mm", entry), dtype=float)
    for name, point in (
        ("entry_point_robot_base_mm", entry),
        ("target_point_robot_base_mm", target),
        ("approach_point_robot_base_mm", approach),
        ("needle_axis_robot_base", axis),
    ):
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError(f"Plan field {name} must contain three finite numbers")
    if np.linalg.norm(axis) <= 1e-9:
        raise ValueError("Plan field needle_axis_robot_base must be non-zero")
    errors_mm = []
    collision_frames = []
    ik_statuses = []
    result = {
        "plan_path": os.path.abspath(plan_path),
        "scene_path": os.path.abspath(scene_path) if scene_path else None,
        "coordinate_system": "UR5 base frame; input mm converted to metres at API boundary",
        "steps": steps,
        "executed_steps": None,
        "collision_check_available": collision_pair is not None,
        "collision_free": None,
        "kinematic_only": bool(kinematic_only),
        "max_tip_position_error_mm": None,
        "final_tip_position_error_mm": None,
    }

    sim.setStepping(not kinematic_only)
    started = False
    try:
        if not kinematic_only:
            sim.startSimulation()
            started = True
        initial_pose = sim.getObjectPose(tip_handle, -1)
        initial_tip = np.asarray(initial_pose[:3], dtype=float) * 1000.0
        if kinematic_only:
            # A stock UR5 may start far from the patient and with a different
            # tool orientation. Solving tiny Cartesian steps while changing
            # orientation can trap simIK in a local failure. The first IK
            # command therefore places the tool at the pre-approach point;
            # the reported trajectory still covers the clinically relevant
            # approach, entry, and insertion segments.
            waypoints, phases = _polyline_samples(
                [approach, entry, target],
                total_steps=steps,
            )
            # _polyline_samples labels its first point as the measured
            # initial state. Here the first point is an actual IK command.
            phase_remap = {
                "initial": "approach",
                "approach": "approach_to_entry",
                "approach_to_entry": "insertion",
            }
            phases = [phase_remap[phase] for phase in phases]
        else:
            waypoints, phases = _polyline_samples(
                [initial_tip, approach, entry, target],
                total_steps=steps,
            )
        result["initial_tip_robot_base_mm"] = initial_tip.tolist()
        result["approach_point_robot_base_mm"] = approach.tolist()
        result["entry_point_robot_base_mm"] = entry.tolist()
        result["target_point_robot_base_mm"] = target.tolist()
        result["phase_sequence"] = phases
        result["executed_steps"] = len(waypoints)
        result["phase_frame_ranges"] = {
            phase: [phases.index(phase), len(phases) - 1 - phases[::-1].index(phase)]
            for phase in dict.fromkeys(phases)
        }

        for frame, (desired_tip, phase) in enumerate(zip(waypoints, phases)):
            if phase == "initial":
                # The first point is the measured scene state, not an IK
                # command. Solving a 180-degree orientation change here can
                # fail even though the subsequent path is reachable.
                ik_statuses.append("initial_pose")
            else:
                sim.setObjectPose(target_handle, -1, _target_pose_mm(desired_tip, axis))
                status = sim_ik.handleGroup(ik_env, ik_group, {"syncWorlds": True})
                ik_statuses.append(status)
                if not _ik_status_ok(sim_ik, status):
                    raise RuntimeError(f"simIK failed during {phase} at frame {frame}: status={status}")

            if not kinematic_only:
                sim.step()
            actual_pose = sim.getObjectPose(tip_handle, -1)
            actual_tip_mm = np.asarray(actual_pose[:3], dtype=float) * 1000.0
            error_mm = float(np.linalg.norm(actual_tip_mm - desired_tip))
            errors_mm.append(error_mm)

            if collision_pair is not None:
                collision_result = sim.checkCollision(collision_pair[0], collision_pair[1])
                collision = bool(collision_result[0]) if isinstance(collision_result, (list, tuple)) else bool(collision_result)
                collision_frames.append(frame if collision else None)
                if collision:
                    raise RuntimeError(f"Collision detected during {phase} at frame {frame}")
            if frame_delay_s > 0:
                time.sleep(frame_delay_s)
    finally:
        if started:
            sim.stopSimulation()

    result["collision_free"] = not any(frame is not None for frame in collision_frames)
    result["max_tip_position_error_mm"] = max(errors_mm) if errors_mm else None
    result["final_tip_position_error_mm"] = errors_mm[-1] if errors_mm else None
    result["ik_statuses"] = [str(status) for status in ik_statuses]
    result["collision_frames"] = [frame for frame in collision_frames if frame is not None]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved CoppeliaSim result: {output}")
    return result


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="data/demo_nodules/planned_trajectory_0037.json")
    parser.add_argument("--scene")
    parser.add_argument("--load-scene", action="store_true")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=23000)
    parser.add_argument("--robot", default="/UR5")
    parser.add_argument("--ik-tip", default="/UR5/ikTip")
    parser.add_argument("--ik-target", default="/UR5/ikTarget")
    parser.add_argument("--joint-path", action="append", dest="joint_paths")
    parser.add_argument("--robot-collision-collection")
    parser.add_argument("--environment-collision-collection")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument(
        "--frame-delay",
        type=float,
        default=0.0,
        help="Seconds to pause after each frame so the GUI motion is visible.",
    )
    parser.add_argument("--allow-no-collision-check", action="store_true")
    parser.add_argument(
        "--initial-joint-angle-rad",
        action="append",
        type=float,
        dest="initial_joint_angles_rad",
        help="Override one of six deterministic kinematic-only start angles; repeat six times.",
    )
    parser.add_argument(
        "--kinematic-only",
        action="store_true",
        help="Keep simulation stopped and use simIK to pose the stock UR5 kinematically",
    )
    parser.add_argument("--output", default="data/demo_nodules/coppeliasim_run_0037.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_remote_simulation(
        plan_path=args.plan,
        scene_path=args.scene,
        load_scene=args.load_scene,
        host=args.host,
        port=args.port,
        robot_path=args.robot,
        ik_tip_path=args.ik_tip,
        ik_target_path=args.ik_target,
        joint_paths=args.joint_paths,
        robot_collision_collection=args.robot_collision_collection,
        environment_collision_collection=args.environment_collision_collection,
        steps=args.steps,
        allow_no_collision_check=args.allow_no_collision_check,
        kinematic_only=args.kinematic_only,
        initial_joint_angles_rad=args.initial_joint_angles_rad,
        frame_delay_s=args.frame_delay,
        output_path=args.output,
    )
