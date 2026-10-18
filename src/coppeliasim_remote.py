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

from robot_geometry import (
    quaternion_xyzw_to_rotation,
    rotation_from_z_axis,
    rotation_to_quaternion_xyzw,
)


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


def _ensure_self_collision_collections(sim, robot_handle, joints):
    """Recreate tool/arm collections when scene reloads drop runtime collections."""
    try:
        needle_collection = sim.getCollectionHandle("needleCollision")
    except Exception:
        needle_collection = None
    try:
        arm_collection = sim.getCollectionHandle("robotArmClearance")
    except Exception:
        arm_collection = None
    if needle_collection is not None and arm_collection is not None:
        return needle_collection, arm_collection

    shapes = sim.getObjectsInTree(robot_handle, sim.object_shape_type, 0)
    needle_shapes = [
        shape
        for shape in shapes
        if sim.getObjectAlias(shape, 4).split("__")[0] == "needle"
    ]
    if len(needle_shapes) != 1:
        raise RuntimeError(
            f"Expected one UR5 needle shape for self-collision checks, found {len(needle_shapes)}"
        )
    needle = needle_shapes[0]
    wrist_shapes = set(sim.getObjectsInTree(joints[-1], sim.object_shape_type, 0))

    if needle_collection is None:
        needle_collection = sim.createCollection(0)
        sim.setCollectionName(needle_collection, "needleCollision")
        sim.addItemToCollection(needle_collection, sim.handle_single, needle, 0)
    if arm_collection is None:
        arm_collection = sim.createCollection(0)
        sim.setCollectionName(arm_collection, "robotArmClearance")
        for shape in shapes:
            if shape != needle and shape not in wrist_shapes:
                sim.addItemToCollection(arm_collection, sim.handle_single, shape, 0)
    return needle_collection, arm_collection


def _target_pose_mm(point_mm, axis, orientation_quaternion=None):
    if orientation_quaternion is None:
        rotation = rotation_from_z_axis(axis)
        quaternion = rotation_to_quaternion_xyzw(rotation)
    else:
        quaternion = np.asarray(orientation_quaternion, dtype=float)
        rotation = quaternion_xyzw_to_rotation(quaternion)
        normalized_axis = np.asarray(axis, dtype=float)
        normalized_axis /= np.linalg.norm(normalized_axis)
        if not np.allclose(rotation[:, 2], normalized_axis, atol=1e-5):
            raise ValueError("Scene IK target axis does not match the planned needle axis")
    point_m = np.asarray(point_mm, dtype=float) / 1000.0
    return point_m.tolist() + quaternion.tolist()


def _ik_status_ok(sim_ik, status):
    if isinstance(status, (list, tuple)):
        status = status[0]
    success = getattr(sim_ik, "result_success", 1)
    return status == success


def _collision_object_names(sim, collision_result):
    """Return the colliding object aliases when checkCollision provides them."""
    if not isinstance(collision_result, (list, tuple)) or len(collision_result) < 2:
        return []
    handles = []

    def collect_handles(value):
        if isinstance(value, (list, tuple)):
            for item in value:
                collect_handles(item)
        elif isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            handles.append(int(value))

    for value in collision_result[1:]:
        collect_handles(value)
    if len(handles) < 2:
        return []
    names = []
    for handle in handles[:2]:
        try:
            names.append(sim.getObjectAlias(handle, 4))
        except Exception:
            names.append(str(handle))
    return names


def _colliding_patient_obstacles(sim, robot_collection, obstacle_handles):
    """Probe individual patient shapes when collection collision omits object IDs."""
    colliding = []
    for handle in obstacle_handles:
        result = sim.checkCollision(robot_collection, handle)
        hit = bool(result[0]) if isinstance(result, (list, tuple)) else bool(result)
        if hit:
            try:
                colliding.append(sim.getObjectAlias(handle, 4))
            except Exception:
                colliding.append(str(handle))
    return colliding


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
    needle_self_collision_pair = None
    if robot_collision_collection and environment_collision_collection:
        collision_pair = (
            _ensure_collection(sim, robot_collision_collection, root_handle=base_handle),
            _ensure_collection(
                sim,
                environment_collision_collection,
                object_handles=_patient_obstacle_handles(sim),
            ),
        )
        needle_self_collision_pair = _ensure_self_collision_collections(
            sim, base_handle, joints
        )
    elif not allow_no_collision_check:
        raise RuntimeError(
            "Collision collections are required. Pass --robot-collision-collection and "
            "--environment-collision-collection, or explicitly use --allow-no-collision-check."
        )

    if kinematic_only:
        # Scene generation can solve and save the planned approach pose. Keep
        # that pose by default; resetting to the generic home angles here
        # creates a large, non-trajectory swing before insertion begins.
        for joint in joints:
            sim.setJointMode(joint, sim.jointmode_kinematic, 0)
        if initial_joint_angles_rad is not None:
            initial_joint_angles_rad = tuple(initial_joint_angles_rad)
            if len(initial_joint_angles_rad) != 6:
                raise ValueError("initial_joint_angles_rad must contain exactly six values")
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
    scene_target_pose = sim.getObjectPose(target_handle, base_handle)
    scene_target_quaternion = np.asarray(scene_target_pose[3:7], dtype=float)
    scene_target_rotation = quaternion_xyzw_to_rotation(scene_target_quaternion)
    normalized_axis = axis / np.linalg.norm(axis)
    if not np.allclose(scene_target_rotation[:, 2], normalized_axis, atol=1e-5):
        scene_target_quaternion = rotation_to_quaternion_xyzw(
            rotation_from_z_axis(normalized_axis)
        )
    errors_mm = []
    collision_frames = []
    needle_self_collision_frames = []
    ik_statuses = []
    result = {
        "plan_path": os.path.abspath(plan_path),
        "scene_path": os.path.abspath(scene_path) if scene_path else None,
        "coordinate_system": "UR5 base frame; input mm converted to metres at API boundary",
        "steps": steps,
        "executed_steps": None,
        "collision_check_available": collision_pair is not None,
        "needle_self_collision_check_available": needle_self_collision_pair is not None,
        "collision_free": None,
        "needle_self_collision_free": None,
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
        initial_pose = sim.getObjectPose(tip_handle, base_handle)
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

        initial_target_pose = _target_pose_mm(waypoints[0], axis, scene_target_quaternion)
        initial_position_error_mm = float(
            np.linalg.norm(np.asarray(initial_pose[:3]) - np.asarray(initial_target_pose[:3]))
            * 1000.0
        )
        initial_quaternion = np.asarray(initial_pose[3:7], dtype=float)
        target_quaternion = np.asarray(initial_target_pose[3:7], dtype=float)
        quaternion_dot = float(
            np.clip(abs(np.dot(initial_quaternion, target_quaternion)), 0.0, 1.0)
        )
        initial_orientation_error_deg = math.degrees(2.0 * math.acos(quaternion_dot))
        result["initial_target_position_error_mm"] = initial_position_error_mm
        result["initial_target_orientation_error_deg"] = initial_orientation_error_deg
        if kinematic_only and (
            initial_position_error_mm > 5.0 or initial_orientation_error_deg > 5.0
        ):
            raise RuntimeError(
                "Saved scene posture is not at the planned pre-approach pose; refusing an "
                "unplanned initial swing. Regenerate the scene and verify its saved UR5 pose. "
                f"position_error_mm={initial_position_error_mm:.3f}, "
                f"orientation_error_deg={initial_orientation_error_deg:.3f}"
            )

        for frame, (desired_tip, phase) in enumerate(zip(waypoints, phases)):
            if frame == 0 and kinematic_only:
                # Scene generation already solved this exact pre-approach pose.
                # Re-solving can select another equivalent UR5 branch and cause
                # a large, unnecessary joint-space swing before the planned path.
                ik_statuses.append("saved_pre_approach_pose")
            elif phase == "initial":
                ik_statuses.append("initial_pose")
            else:
                sim.setObjectPose(
                    target_handle,
                    base_handle,
                    _target_pose_mm(desired_tip, axis, scene_target_quaternion),
                )
                status = sim_ik.handleGroup(ik_env, ik_group, {"syncWorlds": True})
                ik_statuses.append(status)
                if not _ik_status_ok(sim_ik, status):
                    raise RuntimeError(f"simIK failed during {phase} at frame {frame}: status={status}")

            if not kinematic_only:
                sim.step()
            actual_pose = sim.getObjectPose(tip_handle, base_handle)
            actual_tip_mm = np.asarray(actual_pose[:3], dtype=float) * 1000.0
            error_mm = float(np.linalg.norm(actual_tip_mm - desired_tip))
            errors_mm.append(error_mm)

            if collision_pair is not None:
                collision_result = sim.checkCollision(collision_pair[0], collision_pair[1])
                collision = bool(collision_result[0]) if isinstance(collision_result, (list, tuple)) else bool(collision_result)
                collision_frames.append(frame if collision else None)
                if collision:
                    colliding_objects = _collision_object_names(sim, collision_result)
                    if not colliding_objects:
                        colliding_objects = _colliding_patient_obstacles(
                            sim,
                            collision_pair[0],
                            _patient_obstacle_handles(sim),
                        )
                    object_detail = (
                        f"; objects={colliding_objects}" if colliding_objects else ""
                    )
                    raise RuntimeError(
                        f"Collision detected during {phase} at frame {frame}"
                        f"{object_detail}; collision_result={collision_result!r}; "
                        f"desired_tip_mm={desired_tip.tolist()}, "
                        f"actual_tip_mm={actual_tip_mm.tolist()}, "
                        f"tip_error_mm={error_mm:.3f}"
                    )
            if needle_self_collision_pair is not None:
                self_result = sim.checkCollision(
                    needle_self_collision_pair[0], needle_self_collision_pair[1]
                )
                self_collision = (
                    bool(self_result[0])
                    if isinstance(self_result, (list, tuple))
                    else bool(self_result)
                )
                needle_self_collision_frames.append(frame if self_collision else None)
                if self_collision:
                    raise RuntimeError(
                        f"Needle intersects a non-wrist UR5 link during {phase} at frame {frame}"
                    )
            if frame_delay_s > 0:
                time.sleep(frame_delay_s)
    finally:
        if started:
            sim.stopSimulation()

    result["collision_free"] = not any(frame is not None for frame in collision_frames)
    result["needle_self_collision_free"] = (
        not any(frame is not None for frame in needle_self_collision_frames)
        if needle_self_collision_pair is not None
        else None
    )
    result["max_tip_position_error_mm"] = max(errors_mm) if errors_mm else None
    result["final_tip_position_error_mm"] = errors_mm[-1] if errors_mm else None
    result["ik_statuses"] = [str(status) for status in ik_statuses]
    result["collision_frames"] = [frame for frame in collision_frames if frame is not None]
    result["needle_self_collision_frames"] = [
        frame for frame in needle_self_collision_frames if frame is not None
    ]

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
