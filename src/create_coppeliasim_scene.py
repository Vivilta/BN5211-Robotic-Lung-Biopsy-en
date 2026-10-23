"""Create the reproducible CoppeliaSim scene used by the robot demo.

This scene is an integration fixture for UR5 + simIK + ZeroMQ Remote API. The
obstacle boxes are deliberately marked as a simplified patient model: the
current JSON contains diagnostic AABBs and must not be treated as a clinical
anatomical reconstruction.
"""

import argparse
import json
import math
import os
from collections import Counter
from contextlib import contextmanager
from itertools import product
from pathlib import Path

import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from robot_geometry import (
    overhead_mount_base_to_world,
    rotation_from_z_axis,
    rotation_to_quaternion_xyzw,
    rotation_with_z_axis_roll,
    transform_point,
)


UR5_MODEL_CANDIDATES = (
    "/Applications/coppeliaSim.app/Contents/Resources/models/robots/non-mobile/UR5.ttm",
)
DEFAULT_OUTPUT = "scenes/robotic_lung_biopsy.ttt"
DEFAULT_OBSTACLES = "data/demo_nodules/rib_obstacles_0037.json"
DEFAULT_INITIAL_JOINT_ANGLES_RAD = (0.0, -math.pi / 2.0, 0.0, -math.pi / 2.0, 0.0, 0.0)

# IK search defaults.  These are deliberately conservative scene-generation
# safeguards: the saved scene must start from a joint configuration that leaves
# room to continue along the Cartesian trajectory.
DEFAULT_SIMIK_DAMPING = 0.1
DEFAULT_SIMIK_MAX_ITERATIONS = 150
DEFAULT_JOINT_LIMIT_MARGIN_RAD = math.radians(5.0)
DEFAULT_MAX_JOINT_STEP_RAD = math.radians(20.0)


def _resolve_ur5_model(model_path=None):
    candidates = []
    if model_path:
        candidates.append(Path(model_path).expanduser())
    root = os.environ.get("COPPELIASIM_ROOT")
    if root:
        candidates.append(Path(root) / "Contents/Resources/models/robots/non-mobile/UR5.ttm")
    candidates.extend(Path(path) for path in UR5_MODEL_CANDIDATES)
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"UR5.ttm not found. Searched: {searched}")


def _transform_obstacle(obstacle, patient_to_robot_transform, mount_height_mm=0.0):
    minimum = np.asarray(obstacle["min_mm"], dtype=float)
    maximum = np.asarray(obstacle["max_mm"], dtype=float)
    patient_to_robot_transform = (
        np.eye(4) if patient_to_robot_transform is None else patient_to_robot_transform
    )
    robot_to_world = overhead_mount_base_to_world(mount_height_mm)
    corners = [
        transform_point(
            robot_to_world,
            transform_point(patient_to_robot_transform, [x, y, z]),
        )
        for x in (minimum[0], maximum[0])
        for y in (minimum[1], maximum[1])
        for z in (minimum[2], maximum[2])
    ]
    transformed = dict(obstacle)
    transformed["min_mm"] = np.min(corners, axis=0).tolist()
    transformed["max_mm"] = np.max(corners, axis=0).tolist()
    transformed["coordinate_system"] = "CoppeliaSim world coordinates in mm"
    return transformed


def _load_obstacles(path, max_obstacles, patient_to_robot_transform=None, mount_height_mm=0.0):
    if not path:
        return []
    with open(path, "r", encoding="utf-8") as handle:
        obstacles = json.load(handle)
    if not isinstance(obstacles, list):
        raise ValueError("Obstacle JSON must contain a list")
    if max_obstacles is not None and max_obstacles > 0:
        obstacles = obstacles[:max_obstacles]
    return [
        _transform_obstacle(item, patient_to_robot_transform, mount_height_mm)
        for item in obstacles
    ]


def _set_shape_color(sim, handle, rgb):
    sim.setShapeColor(handle, "", sim.colorcomponent_ambient_diffuse, list(rgb))


def _set_shape_static(sim, handle):
    """Keep a shape out of Bullet dynamics while retaining collision checks."""
    sim.setObjectInt32Param(handle, sim.shapeintparam_static, 1)


def _set_visible(sim, handle, visible):
    """Toggle rendering without removing the object from collision collections."""
    sim.setObjectInt32Param(handle, sim.objintparam_visibility_layer, 1 if visible else 0)


def _make_robot_non_dynamic(sim, robot_handle):
    """Prevent the stock UR5 model from falling when the GUI starts dynamics."""
    sim.setModelProperty(robot_handle, sim.modelproperty_not_dynamic)
    for shape in sim.getObjectsInTree(robot_handle, sim.object_shape_type, 0):
        _set_shape_static(sim, shape)


def _remove_default_floor(sim):
    """Remove CoppeliaSim's helper floor, which otherwise hides negative-Z models."""
    try:
        floor = sim.getObject("/Floor")
    except Exception:
        return False
    sim.removeObject(floor)
    return True


def _remove_default_box(sim):
    """Remove CoppeliaSim's standalone helper box, not patient obstacles."""
    try:
        box = sim.getObject("/box")
    except Exception:
        return False
    sim.removeObject(box)
    return True


def _configure_overview_camera(sim, view_target_robot_base_mm=None, mount_height_mm=0.0):
    """Set a stable overview view that contains the UR5 and needle path."""
    try:
        camera = sim.getObject("/DefaultCamera")
    except Exception:
        return False
    target_mm = np.asarray(
        [60.0, 0.0, 180.0]
        if view_target_robot_base_mm is None
        else view_target_robot_base_mm,
        dtype=float,
    )
    target_m = transform_point(
        overhead_mount_base_to_world(mount_height_mm), target_mm
    ) / 1000.0
    camera_position = target_m + np.array([1.15, -1.35, 0.90], dtype=float)
    forward = target_m - camera_position
    forward /= np.linalg.norm(forward)
    quaternion = rotation_to_quaternion_xyzw(rotation_from_z_axis(-forward))
    sim.setObjectPose(camera, camera_position.tolist() + quaternion.tolist(), sim.handle_world)
    return True


def _create_box(sim, name, minimum_mm, maximum_mm, rgb, visible=False):
    minimum = [float(value) / 1000.0 for value in minimum_mm]
    maximum = [float(value) / 1000.0 for value in maximum_mm]
    size = [maximum[index] - minimum[index] for index in range(3)]
    if any(value <= 0 for value in size):
        raise ValueError(f"Invalid obstacle bounds for {name}: {minimum_mm}, {maximum_mm}")
    handle = sim.createPrimitiveShape(sim.primitiveshape_cuboid, size, 0)
    sim.setObjectAlias(handle, name)
    sim.setObjectPose(
        handle,
        [
            (minimum[0] + maximum[0]) / 2.0,
            (minimum[1] + maximum[1]) / 2.0,
            (minimum[2] + maximum[2]) / 2.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        sim.handle_world,
    )
    _set_shape_color(sim, handle, rgb)
    _set_shape_static(sim, handle)
    _set_visible(sim, handle, visible)
    return handle


def _create_robot_base(sim, mount_height_mm=0.0):
    """Create the overhead plate that supports the inverted UR5."""
    base = _create_box(
        sim,
        "robotBasePlate",
        [-160.0, -160.0, float(mount_height_mm)],
        [160.0, 160.0, float(mount_height_mm) + 16.0],
        (0.18, 0.20, 0.24),
        visible=True,
    )
    _set_visible(sim, base, True)
    return base


def _initial_ik_seed_candidates(seed):
    """Generate deterministic UR5 wrist/elbow alternatives around the configured seed."""
    seed = np.asarray(seed, dtype=float)
    if seed.shape != (6,):
        raise ValueError("Initial IK seed must contain exactly six joint angles")
    offsets = product((0.0, math.pi, -math.pi), repeat=3)
    candidates = []
    for shoulder, elbow, wrist in offsets:
        candidate = seed.copy()
        candidate[[0, 2, 4]] += [shoulder, elbow, wrist]
        if not any(np.allclose(candidate, existing) for existing in candidates):
            candidates.append(candidate)
    return candidates


def _read_joint_limits(sim, joints, margin_rad=0.0):
    """Read CoppeliaSim joint intervals as ``(cyclic, lower, upper)``.

    CoppeliaSim reports a non-cyclic interval as ``(minimum, range)``.  The
    explicit conversion here keeps the same convention for the scene-side
    filtering and for the simIK environment.  Cyclic joints are left
    unconstrained because their equivalent angles are valid by definition.
    """
    margin_rad = float(margin_rad)
    if margin_rad < 0.0:
        raise ValueError("joint limit margin must be non-negative")

    limits = []
    for joint in joints:
        try:
            cyclic, interval = sim.getJointInterval(joint)
        except Exception as exc:
            raise RuntimeError(
                "Could not read a UR5 joint interval from CoppeliaSim; "
                "joint-limit validation is required for scene generation."
            ) from exc
        if cyclic:
            limits.append({"cyclic": True, "lower": None, "upper": None})
            continue

        interval = np.asarray(interval, dtype=float).reshape(-1)
        if interval.size < 2 or not np.isfinite(interval[:2]).all():
            raise RuntimeError(f"Invalid CoppeliaSim joint interval: {interval!r}")
        lower = float(interval[0])
        upper = lower + float(interval[1])
        if upper <= lower:
            raise RuntimeError(f"Invalid CoppeliaSim joint interval: {interval.tolist()!r}")
        effective_lower = lower + margin_rad
        effective_upper = upper - margin_rad
        if effective_upper <= effective_lower:
            raise ValueError(
                f"Joint interval is narrower than twice the configured margin: "
                f"[{lower:.6f}, {upper:.6f}] rad"
            )
        limits.append(
            {
                "cyclic": False,
                "lower": effective_lower,
                "upper": effective_upper,
            }
        )
    return limits


def _within_joint_limits(joint_angles, limits, atol=1e-9):
    """Return whether every non-cyclic joint is inside its safe interval."""
    joint_angles = np.asarray(joint_angles, dtype=float)
    if joint_angles.shape != (len(limits),):
        raise ValueError("Joint vector and limit list must have the same length")
    if not np.isfinite(joint_angles).all():
        return False
    for angle, limit in zip(joint_angles, limits):
        if limit["cyclic"]:
            continue
        if angle < limit["lower"] - atol or angle > limit["upper"] + atol:
            return False
    return True


def _minimum_joint_limit_margin(joint_angles, limits):
    """Return the smallest remaining safe margin in radians."""
    joint_angles = np.asarray(joint_angles, dtype=float)
    margins = [
        min(float(angle) - limit["lower"], limit["upper"] - float(angle))
        for angle, limit in zip(joint_angles, limits)
        if not limit["cyclic"]
    ]
    return min(margins) if margins else float("inf")


def _wrapped_joint_delta(current, previous):
    """Shortest signed angular difference for a pair of six-joint vectors."""
    current = np.asarray(current, dtype=float)
    previous = np.asarray(previous, dtype=float)
    return np.arctan2(np.sin(current - previous), np.cos(current - previous))


def _collision_result(sim, first_collection, second_collection):
    result = sim.checkCollision(first_collection, second_collection)
    if isinstance(result, (list, tuple)):
        return bool(result[0]), list(result[1]) if len(result) > 1 else []
    return bool(result), []


def _object_alias(sim, handle):
    try:
        return sim.getObjectAlias(int(handle), 4)
    except Exception:
        return str(handle)


def _sample_trajectory_points(points, total_steps, max_segment_mm=3.0, max_steps=1000):
    """Sample a Cartesian polyline with a spatial step bound.

    ``total_steps`` is retained as a lower bound for compatibility.  The
    adaptive bound is what prevents a long needle sweep from jumping over a
    thin AABB between two collision checks.
    """
    points = [np.asarray(point, dtype=float) for point in points]
    if len(points) < 2 or any(point.shape != (3,) or not np.isfinite(point).all() for point in points):
        raise ValueError("Trajectory points must contain at least two finite 3D points")
    if total_steps < 2:
        raise ValueError("trajectory_steps must be at least 2")
    if max_segment_mm <= 0 or max_steps < 2:
        raise ValueError("max_segment_mm must be positive and max_steps >= 2")

    lengths = np.asarray(
        [np.linalg.norm(points[index + 1] - points[index])
         for index in range(len(points) - 1)],
        dtype=float,
    )
    active = np.flatnonzero(lengths > 1e-9)
    if len(active) == 0:
        return [points[0].copy()]

    intervals = np.zeros(len(lengths), dtype=int)
    intervals[active] = np.maximum(
        1, np.ceil(lengths[active] / float(max_segment_mm)).astype(int)
    )
    requested_intervals = max(int(total_steps) - 1, int(intervals.sum()))
    remaining = requested_intervals - int(intervals.sum())
    if remaining:
        proportions = lengths[active] / lengths[active].sum()
        extra = np.floor(remaining * proportions).astype(int)
        intervals[active] += extra
        for index in range(remaining - int(extra.sum())):
            intervals[active[np.argmax(lengths[active])]] += 1

    if int(intervals.sum()) + 1 > int(max_steps):
        scale = (float(max_steps) - 1.0) / float(intervals.sum())
        intervals[active] = np.maximum(1, np.floor(intervals[active] * scale).astype(int))
        while int(intervals.sum()) + 1 > int(max_steps):
            longest = active[np.argmax(lengths[active] / intervals[active])]
            if intervals[longest] <= 1:
                break
            intervals[longest] -= 1

    samples = [points[0].copy()]
    for index, count in enumerate(intervals):
        if count:
            samples.extend(np.linspace(points[index], points[index + 1], count + 1)[1:])
    return samples


@contextmanager
def _suppress_display(sim, suppress):
    """Hide intermediate IK candidates while preserving collision checks."""
    if not suppress:
        yield
        return

    display_param = getattr(sim, "boolparam_display_enabled", None)
    if display_param is None:
        yield
        return

    previous = None
    try:
        previous = sim.getBoolParam(display_param)
        sim.setBoolParam(display_param, False)
    except Exception:
        yield
        return

    try:
        yield
    finally:
        try:
            sim.setBoolParam(display_param, bool(previous))
        except Exception:
            pass


def _set_initial_ik_pose_impl(
    sim,
    sim_ik,
    robot_handle,
    tip_handle,
    target_handle,
    joints,
    point_mm,
    axis,
    initial_joint_angles_rad,
    tool_mount_handle,
    needle_handle,
    needle_length_mm,
    tip_mount_signs,
    needle_collection,
    arm_collection,
    robot_collection,
    patient_collection,
    mount_collection,
    mount_arm_collection,
    trajectory_points_mm=None,
    trajectory_steps=100,
    max_cartesian_step_mm=3.0,
    max_trajectory_steps=1000,
    joint_limit_margin_rad=DEFAULT_JOINT_LIMIT_MARGIN_RAD,
    max_joint_step_rad=DEFAULT_MAX_JOINT_STEP_RAD,
    simik_damping=DEFAULT_SIMIK_DAMPING,
    simik_max_iterations=DEFAULT_SIMIK_MAX_ITERATIONS,
):
    """Choose a pose whose complete planned trajectory is collision-free."""
    if max_joint_step_rad <= 0:
        raise ValueError("max_joint_step_rad must be positive")
    if max_cartesian_step_mm <= 0 or max_trajectory_steps < 2:
        raise ValueError("max_cartesian_step_mm must be positive and max_trajectory_steps >= 2")
    if simik_damping <= 0:
        raise ValueError("simik_damping must be positive")
    if simik_max_iterations < 1:
        raise ValueError("simik_max_iterations must be at least 1")

    joint_limits = _read_joint_limits(sim, joints, joint_limit_margin_rad)
    target_pose = (
        np.asarray(point_mm, dtype=float) / 1000.0
    ).tolist() + rotation_to_quaternion_xyzw(
        rotation_with_z_axis_roll(np.asarray(axis, dtype=float), 0.0)
    ).tolist()
    last_status = None
    ik_success_count = 0
    colliding_solution_count = 0
    colliding_pairs = Counter()
    trajectory_rejection_count = 0
    candidate_solutions = []
    sampled_trajectory = None
    if trajectory_points_mm is not None:
        sampled_trajectory = _sample_trajectory_points(
            trajectory_points_mm, trajectory_steps, max_cartesian_step_mm, max_trajectory_steps
        )

    def _create_ik_context():
        env = sim_ik.createEnvironment()
        group = sim_ik.createGroup(env)
        sim_ik.setGroupCalculation(
            env,
            group,
            sim_ik.method_damped_least_squares,
            float(simik_damping),
            int(simik_max_iterations),
        )
        _, sim_to_ik, _ = sim_ik.addElementFromScene(
            env,
            group,
            robot_handle,
            tip_handle,
            target_handle,
            sim_ik.constraint_pose,
        )
        flags = int(getattr(sim_ik, "group_enabled", 1))
        for name in (
            "group_avoidlimits",
            "group_stoponlimithit",
            "group_restoreonbadlintol",
            "group_restoreonbadangtol",
        ):
            value = getattr(sim_ik, name, None)
            if value is not None:
                flags |= int(value)
        set_group_flags = getattr(sim_ik, "setGroupFlags", None)
        if set_group_flags is not None and flags:
            try:
                set_group_flags(env, group, flags)
            except Exception:
                pass
        for joint_index, joint in enumerate(joints):
            ik_joint = sim_to_ik.get(joint)
            if ik_joint is not None:
                sim_ik.setJointMode(env, ik_joint, sim_ik.jointmode_ik)
                set_step = getattr(sim_ik, "setJointMaxStepSize", None)
                if set_step is not None:
                    try:
                        set_step(env, ik_joint, float(max_joint_step_rad))
                    except Exception:
                        pass
                set_margin = getattr(sim_ik, "setJointLimitMargin", None)
                if set_margin is not None:
                    try:
                        set_margin(env, ik_joint, float(joint_limit_margin_rad))
                    except Exception:
                        pass
                limit = joint_limits[joint_index]
                if not limit["cyclic"]:
                    # simIK uses the same (minimum, range) interval convention
                    # as sim.getJointInterval().  Keep a margin at both ends.
                    set_interval = getattr(sim_ik, "setJointInterval", None)
                    if set_interval is not None:
                        try:
                            set_interval(
                                env,
                                ik_joint,
                                False,
                                [
                                    float(limit["lower"]),
                                    float(limit["upper"] - limit["lower"]),
                                ],
                            )
                        except Exception:
                            pass
        return env, group

    for mount_sign in tip_mount_signs:
        sim.setObjectPose(
            needle_handle,
            [0.0, 0.0, mount_sign * needle_length_mm / 2000.0, 0.0, 0.0, 0.0, 1.0],
            tool_mount_handle,
        )
        sim.setObjectPose(
            tip_handle,
            [
                0.0,
                0.0,
                mount_sign * needle_length_mm / 2000.0,
                1.0 if mount_sign < 0 else 0.0,
                0.0,
                0.0,
                0.0 if mount_sign < 0 else 1.0,
            ],
            needle_handle,
        )
        for roll_deg in range(0, 360, 30):
            target_pose[3:] = rotation_to_quaternion_xyzw(
                rotation_with_z_axis_roll(np.asarray(axis, dtype=float), math.radians(roll_deg))
            ).tolist()
            sim.setObjectPose(target_handle, robot_handle, target_pose)
            for seed in _initial_ik_seed_candidates(initial_joint_angles_rad):
                # Do not ask simIK to start from a seed already outside the
                # safe interval.  This avoids equivalent branches that begin
                # at a hard stop and are then difficult to continue smoothly.
                if not _within_joint_limits(seed, joint_limits):
                    continue
                for joint, angle in zip(joints, seed):
                    sim.setJointPosition(joint, float(angle))

                env = None
                try:
                    env, group = _create_ik_context()

                    status = sim_ik.handleGroup(env, group, {"syncWorlds": True})
                    if isinstance(status, (list, tuple)):
                        status = status[0]
                    last_status = status
                    if status != getattr(sim_ik, "result_success", 1):
                        continue

                    ik_success_count += 1
                    collided, object_handles = _collision_result(
                        sim, needle_collection, arm_collection
                    )
                    if collided:
                        colliding_solution_count += 1
                        pair_names = tuple(sorted(_object_alias(sim, handle) for handle in object_handles))
                        colliding_pairs[pair_names or ("unknown pair",)] += 1
                        continue
                    patient_collision, patient_pairs = _collision_result(
                        sim, robot_collection, patient_collection
                    )
                    if patient_collision:
                        colliding_solution_count += 1
                        pair_names = tuple(
                            sorted(_object_alias(sim, handle) for handle in patient_pairs)
                        )
                        colliding_pairs[pair_names or ("robot/patient",)] += 1
                        continue

                    # Reject IK configurations intersecting the mounting plate.
                    mount_collision, mount_pairs = _collision_result(
                        sim,
                        mount_arm_collection,
                        mount_collection,
                    )

                    if mount_collision:
                        colliding_solution_count += 1

                        pair_names = tuple(
                            sorted(
                                _object_alias(sim, handle)
                                for handle in mount_pairs
                            )
                        )

                        colliding_pairs[
                            pair_names or ("robot/mount",)
                        ] += 1

                        continue

                    joint_angles = np.asarray(
                        [sim.getJointPosition(joint) for joint in joints], dtype=float
                    )
                    if not _within_joint_limits(joint_angles, joint_limits):
                        colliding_solution_count += 1
                        colliding_pairs[("joint_limit_margin",)] += 1
                        continue
                    # Preserve the analytical IK branch whenever possible. A
                    # weak seed penalty can switch q4 by pi for a tiny q3
                    # posture improvement, which is a visually large and
                    # mechanically different wrist configuration. The seed
                    # distance is therefore primary; elbow posture is only a
                    # secondary tie-breaker.
                    posture_cost = abs(
                        math.atan2(math.sin(joint_angles[2]), math.cos(joint_angles[2]))
                    )
                    seed_cost = float(
                        np.linalg.norm(
                            np.arctan2(
                                np.sin(joint_angles - np.asarray(seed, dtype=float)),
                                np.cos(joint_angles - np.asarray(seed, dtype=float)),
                            )
                        )
                    )
                    limit_margin = _minimum_joint_limit_margin(joint_angles, joint_limits)
                    # Keep the seed distance primary, while preferring a
                    # branch with more room to move away from joint stops.
                    limit_cost = 0.02 / max(limit_margin, 1e-3)
                    score = seed_cost + 0.05 * posture_cost + limit_cost
                    candidate_solutions.append(
                        {
                            "score": score,
                            "limit_margin_rad": float(limit_margin),
                            "mount_sign": mount_sign,
                            "roll_deg": roll_deg,
                            "joint_angles": joint_angles.copy(),
                            "seed": seed.copy(),
                            "target_pose": list(target_pose),
                        }
                    )
                finally:
                    if env is not None:
                        sim_ik.eraseEnvironment(env)

    # Initial collision checks are cheap enough to run for all IK branches.
    # Run the complete trajectory preflight for every initially safe branch so
    # the selected posture is based on the whole path, rather than whichever
    # candidate happens to pass first.
    candidate_solutions.sort(key=lambda candidate: candidate["score"])
    trajectory_candidates = []
    for candidate in candidate_solutions:
        mount_sign = candidate["mount_sign"]
        roll_deg = candidate["roll_deg"]
        joint_angles = candidate["joint_angles"]
        candidate_target_pose = candidate["target_pose"]
        sim.setObjectPose(
            needle_handle,
            [0.0, 0.0, mount_sign * needle_length_mm / 2000.0, 0.0, 0.0, 0.0, 1.0],
            tool_mount_handle,
        )
        sim.setObjectPose(
            tip_handle,
            [
                0.0,
                0.0,
                mount_sign * needle_length_mm / 2000.0,
                1.0 if mount_sign < 0 else 0.0,
                0.0,
                0.0,
                0.0 if mount_sign < 0 else 1.0,
            ],
            needle_handle,
        )
        sim.setObjectPose(target_handle, robot_handle, candidate_target_pose)
        for joint, angle in zip(joints, joint_angles):
            sim.setJointPosition(joint, float(angle))

        trajectory_collision = False
        trajectory_joint_cost = 0.0
        trajectory_max_joint_step = 0.0
        previous_joint_angles = joint_angles.copy()
        env = None
        try:
            env, group = _create_ik_context()
            if sampled_trajectory is not None:
                for path_point in sampled_trajectory[1:]:
                    path_pose = (
                        (np.asarray(path_point, dtype=float) / 1000.0).tolist()
                        + candidate_target_pose[3:]
                    )
                    sim.setObjectPose(target_handle, robot_handle, path_pose)
                    path_status = sim_ik.handleGroup(env, group, {"syncWorlds": True})
                    if isinstance(path_status, (list, tuple)):
                        path_status = path_status[0]
                    if path_status != getattr(sim_ik, "result_success", 1):
                        trajectory_collision = True
                        colliding_pairs[("trajectory_ik_failure",)] += 1
                        break

                    path_joint_angles = np.asarray(
                        [sim.getJointPosition(joint) for joint in joints], dtype=float
                    )
                    if not _within_joint_limits(path_joint_angles, joint_limits):
                        trajectory_collision = True
                        colliding_pairs[("trajectory_joint_limit_margin",)] += 1
                        break
                    joint_delta = _wrapped_joint_delta(
                        path_joint_angles, previous_joint_angles
                    )
                    max_joint_step = float(np.max(np.abs(joint_delta)))
                    if max_joint_step > float(max_joint_step_rad):
                        trajectory_collision = True
                        colliding_pairs[("trajectory_joint_step",)] += 1
                        break
                    trajectory_joint_cost += float(np.linalg.norm(joint_delta))
                    trajectory_max_joint_step = max(
                        trajectory_max_joint_step, max_joint_step
                    )
                    previous_joint_angles = path_joint_angles

                    path_collided, path_pairs = _collision_result(
                        sim, robot_collection, patient_collection
                    )
                    if path_collided:
                        trajectory_collision = True
                        pair_names = tuple(
                            sorted(_object_alias(sim, handle) for handle in path_pairs)
                        )
                        colliding_pairs[pair_names or ("trajectory robot/patient",)] += 1
                        break
                    self_collided, self_pairs = _collision_result(
                        sim, needle_collection, arm_collection
                    )
                    if self_collided:
                        trajectory_collision = True
                        pair_names = tuple(
                            sorted(_object_alias(sim, handle) for handle in self_pairs)
                        )
                        colliding_pairs[pair_names or ("trajectory needle/arm",)] += 1
                        break
                    # Check mounting-plate collision at every sampled trajectory pose.
                    mount_collided, mount_pairs = _collision_result(
                        sim,
                        mount_arm_collection,
                        mount_collection,
                    )

                    if mount_collided:
                        trajectory_collision = True

                        pair_names = tuple(
                            sorted(
                                _object_alias(sim, handle)
                                for handle in mount_pairs
                            )
                        )

                        colliding_pairs[
                            pair_names or ("trajectory robot/mount",)
                        ] += 1

                        break
        finally:
            if env is not None:
                sim_ik.eraseEnvironment(env)

        if trajectory_collision:
            trajectory_rejection_count += 1
            continue
        candidate = dict(candidate)
        candidate["trajectory_joint_cost"] = float(trajectory_joint_cost)
        candidate["trajectory_max_joint_step_rad"] = float(trajectory_max_joint_step)
        trajectory_candidates.append(candidate)

    best_solution = None
    if trajectory_candidates:
        # Prefer the seed/limit score, then smooth continuation, then the
        # remaining limit margin.  All terms are computed only for complete
        # collision-free trajectories.
        best_solution = min(
            trajectory_candidates,
            key=lambda candidate: (
                candidate["score"] + 0.05 * candidate["trajectory_joint_cost"],
                candidate["trajectory_max_joint_step_rad"],
                -candidate["limit_margin_rad"],
            ),
        )

    if best_solution is not None:
        mount_sign = best_solution["mount_sign"]
        roll_deg = best_solution["roll_deg"]
        joint_angles = best_solution["joint_angles"]
        seed = best_solution["seed"]
        sim.setObjectPose(
            needle_handle,
            [0.0, 0.0, mount_sign * needle_length_mm / 2000.0, 0.0, 0.0, 0.0, 1.0],
            tool_mount_handle,
        )
        sim.setObjectPose(
            tip_handle,
            [
                0.0,
                0.0,
                mount_sign * needle_length_mm / 2000.0,
                1.0 if mount_sign < 0 else 0.0,
                0.0,
                0.0,
                0.0 if mount_sign < 0 else 1.0,
            ],
            needle_handle,
        )
        target_pose[3:] = rotation_to_quaternion_xyzw(
            rotation_with_z_axis_roll(np.asarray(axis, dtype=float), math.radians(roll_deg))
        ).tolist()
        sim.setObjectPose(target_handle, robot_handle, target_pose)
        for joint, angle in zip(joints, joint_angles):
            sim.setJointPosition(joint, float(angle))
        print(
            "Selected collision-free UR5 posture (full-trajectory preflight) "
            f"(mount sign={mount_sign:+d}, tool roll={roll_deg} deg, "
            f"q3={joint_angles[2]:.3f} rad, "
            f"limit_margin={best_solution['limit_margin_rad']:.3f} rad, "
            f"max_step={best_solution.get('trajectory_max_joint_step_rad', 0.0):.3f} rad, "
            f"seed={np.round(seed, 3).tolist()})"
        )
        return

    raise RuntimeError(
        "Could not create a collision-free initial biopsy pose; the scene was not saved. "
        f"IK successes={ik_success_count}, initially safe candidates={len(candidate_solutions)}, "
        f"colliding solutions={colliding_solution_count}, "
        f"trajectory rejections={trajectory_rejection_count}, "
        f"complete trajectory candidates={len(trajectory_candidates)}, "
        f"last simIK status={last_status}, most frequent colliding pairs="
        f"{colliding_pairs.most_common(5)}. Check the mount, target, and patient obstacle geometry."
    )


def _set_initial_ik_pose(*args, **kwargs):
    """Run initial-pose search without displaying intermediate candidate motions."""
    sim = args[0]
    suppress = kwargs.get("trajectory_points_mm") is not None
    with _suppress_display(sim, suppress):
        return _set_initial_ik_pose_impl(*args, **kwargs)


def _create_needle_and_ik(
    sim,
    tool_mount_handle,
    robot_handle,
    needle_length_mm=220.0,
    needle_radius_mm=4.0,
):
    needle_length = float(needle_length_mm) / 1000.0
    if needle_length <= 0:
        raise ValueError("needle_length_mm must be positive")
    needle_radius = float(needle_radius_mm) / 1000.0
    if needle_radius <= 0:
        raise ValueError("needle_radius_mm must be positive")
    needle = sim.createPrimitiveShape(
        sim.primitiveshape_cylinder,
        [needle_radius * 2.0, needle_radius * 2.0, needle_length],
        0,
    )
    sim.setObjectAlias(needle, "needle")
    _set_shape_static(sim, needle)
    sim.setObjectParent(needle, tool_mount_handle, False)
    sim.setObjectPose(
        needle,
        [0.0, 0.0, needle_length / 2.0, 0.0, 0.0, 0.0, 1.0],
        tool_mount_handle,
    )
    _set_shape_color(sim, needle, (0.82, 0.82, 0.86))

    tip = sim.createDummy(0.018)
    sim.setObjectAlias(tip, "ikTip")
    sim.setObjectParent(tip, needle, False)
    # The dummy represents the physical needle tip. The cylinder is centered
    # at half-length, so the tip must be placed at the far end of the needle.
    sim.setObjectPose(tip, [0.0, 0.0, needle_length, 0.0, 0.0, 0.0, 1.0], needle)

    target = sim.createDummy(0.024)
    sim.setObjectAlias(target, "ikTarget")
    sim.setObjectPose(target, [0.45, 0.0, 0.35, 0.0, 0.0, 0.0, 1.0], sim.handle_world)
    sim.setObjectParent(target, robot_handle, True)

    return needle, tip, target


def create_scene(
    output_path,
    obstacle_path=None,
    max_obstacles=0,
    host="127.0.0.1",
    port=23000,
    initial_joint_angles_rad=DEFAULT_INITIAL_JOINT_ANGLES_RAD,
    patient_to_robot_transform=None,
    view_target_robot_base_mm=None,
    show_obstacles=True,
    ur5_model_path=None,
    initial_tool_point_mm=None,
    initial_needle_axis=None,
    trajectory_points_mm=None,
    trajectory_steps=100,
    max_cartesian_step_mm=3.0,
    max_trajectory_steps=1000,
    needle_length_mm=220.0,
    mount_height_mm=0.0,
    needle_radius_mm=4.0,
    joint_limit_margin_rad=DEFAULT_JOINT_LIMIT_MARGIN_RAD,
    max_joint_step_rad=DEFAULT_MAX_JOINT_STEP_RAD,
    simik_damping=DEFAULT_SIMIK_DAMPING,
    simik_max_iterations=DEFAULT_SIMIK_MAX_ITERATIONS,
):
    output_path = os.path.abspath(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    mount_height_mm = float(mount_height_mm)
    print("\n[SCENE GEOMETRY]")
    print(f"Needle radius: {needle_radius_mm} mm")
    print(f"Needle diameter: {2 * needle_radius_mm} mm")
    print(f"Needle length: {needle_length_mm} mm")
    print(f"Mount height: {mount_height_mm} mm")
    obstacles = _load_obstacles(
        obstacle_path, max_obstacles, patient_to_robot_transform, mount_height_mm
    )

    client = RemoteAPIClient(host, port)
    sim = client.require("sim")
    sim_ik = client.require("simIK")
    if sim.getSimulationState() != sim.simulation_stopped:
        sim.stopSimulation()
    sim.closeScene()
    _remove_default_floor(sim)
    _remove_default_box(sim)

    ur5 = sim.loadModel(_resolve_ur5_model(ur5_model_path))
    sim.setObjectAlias(ur5, "UR5")
    sim.setObjectPose(
        ur5,
        sim.handle_world,
        [0.0, 0.0, mount_height_mm / 1000.0, 1.0, 0.0, 0.0, 0.0],
    )
    _make_robot_non_dynamic(sim, ur5)

    base_plate = _create_robot_base(
        sim, mount_height_mm
    )

    mount_collection = sim.createCollection(0)
    sim.setCollectionName(
        mount_collection, "mountCollision"
    )
    sim.addItemToCollection(
        mount_collection,
        sim.handle_single,
        base_plate,
        0,
    )
    joints = sim.getObjectsInTree(ur5, sim.object_joint_type, 0)
    if len(joints) != 6:
        raise RuntimeError(f"Expected 6 UR5 joints, found {len(joints)}")
    if len(initial_joint_angles_rad) != 6:
        raise ValueError("initial_joint_angles_rad must contain exactly six values")
    for joint in joints:
        sim.setJointMode(joint, sim.jointmode_kinematic, 0)
    for joint, angle in zip(joints, initial_joint_angles_rad):
        sim.setJointPosition(joint, float(angle))

    # Attach the tool to joint 6's coordinate frame, not an arbitrary mesh origin.
    final_joint = joints[-1]
    descendants = sim.getObjectsInTree(final_joint, sim.object_shape_type, 0)
    if not descendants:
        raise RuntimeError("Could not locate the UR5 wrist/flange shape")
    needle, tip, target = _create_needle_and_ik(
        sim, final_joint, ur5, needle_length_mm, needle_radius_mm
    )
    robot_collection = sim.createCollection(0)
    sim.setCollectionName(robot_collection, "robotCollision")
    sim.addItemToCollection(robot_collection, sim.handle_tree, ur5, 0)

    needle_collection = sim.createCollection(0)
    sim.setCollectionName(needle_collection, "needleCollision")
    sim.addItemToCollection(needle_collection, sim.handle_single, needle, 0)
    arm_collection = sim.createCollection(0)
    sim.setCollectionName(arm_collection, "robotArmClearance")
    wrist_shapes = set(sim.getObjectsInTree(final_joint, sim.object_shape_type, 0))
    for shape in sim.getObjectsInTree(ur5, sim.object_shape_type, 0):
        if shape != needle and shape not in wrist_shapes:
            sim.addItemToCollection(arm_collection, sim.handle_single, shape, 0)

    # Moving robot links vs overhead mounting plate.
    # Exclude the fixed-base shapes and include the wrist and needle.
    mount_arm_collection = sim.createCollection(0)
    sim.setCollectionName(mount_arm_collection, "mountArmCollision")

    all_shapes = sim.getObjectsInTree(
        ur5, sim.object_shape_type, 0
    )

    base_joint = joints[0]
    moving_shapes = set(
        sim.getObjectsInTree(base_joint, sim.object_shape_type, 0)
    )

    # The needle is also a moving collision object.
    moving_shapes.add(needle)

    for shape in moving_shapes:
        sim.addItemToCollection(
            mount_arm_collection,
            sim.handle_single,
            shape,
            0,
        )

    print("[MOUNT] Moving collision shapes:")

    for shape in sorted(moving_shapes):
        print(
            "  ",
            shape,
            _object_alias(sim, shape)
        )
    patient_collection = sim.createCollection(0)
    sim.setCollectionName(patient_collection, "patientCollision")
    for index, obstacle in enumerate(obstacles):
        box = _create_box(
            sim,
            f"patientObstacle_{index + 1}",
            obstacle["min_mm"],
            obstacle["max_mm"],
            (0.85, 0.28, 0.18),
            visible=show_obstacles,
        )
        sim.addItemToCollection(patient_collection, sim.handle_single, box, 0)

    if initial_tool_point_mm is not None and initial_needle_axis is not None:
        _set_initial_ik_pose(
            sim,
            sim_ik,
            ur5,
            tip,
            target,
            joints,
            initial_tool_point_mm,
            initial_needle_axis,
            initial_joint_angles_rad,
            final_joint,
            needle,
            needle_length_mm,
            (1, -1),
            needle_collection,
            arm_collection,
            robot_collection,
            patient_collection,
            mount_collection,
            mount_arm_collection,
            trajectory_points_mm=trajectory_points_mm,
            trajectory_steps=trajectory_steps,
            max_cartesian_step_mm=max_cartesian_step_mm,
            max_trajectory_steps=max_trajectory_steps,
            joint_limit_margin_rad=joint_limit_margin_rad,
            max_joint_step_rad=max_joint_step_rad,
            simik_damping=simik_damping,
            simik_max_iterations=simik_max_iterations,
        )

    _configure_overview_camera(sim, view_target_robot_base_mm, mount_height_mm)
    sim.saveScene(output_path)
    print(f"Saved scene: {output_path}")
    print(f"UR5 joints: {len(joints)}")
    print(f"IK tip: /UR5/ikTip; IK target: /UR5/ikTarget")
    print(
        f"Collections: /robotCollision and /patientCollision ({len(obstacles)} obstacles; "
        f"visible={show_obstacles})"
    )
    return output_path


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--obstacles", default=DEFAULT_OBSTACLES)
    parser.add_argument(
        "--max-obstacles",
        type=int,
        default=0,
        help="Maximum obstacles to instantiate; 0 means all entries",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=23000)
    parser.add_argument("--ur5-model", help="Path to CoppeliaSim's UR5.ttm model")
    parser.add_argument("--max-cartesian-step-mm", type=float, default=3.0)
    parser.add_argument("--max-trajectory-steps", type=int, default=1000)
    parser.add_argument("--joint-limit-margin-rad", type=float, default=DEFAULT_JOINT_LIMIT_MARGIN_RAD)
    parser.add_argument("--max-joint-step-rad", type=float, default=DEFAULT_MAX_JOINT_STEP_RAD)
    parser.add_argument("--simik-damping", type=float, default=DEFAULT_SIMIK_DAMPING)
    parser.add_argument("--simik-max-iterations", type=int, default=DEFAULT_SIMIK_MAX_ITERATIONS)
    parser.add_argument("--initial-tool-point-mm", nargs=3, type=float)
    parser.add_argument("--initial-needle-axis", nargs=3, type=float)
    parser.add_argument(
        "--view-target-mm",
        nargs=3,
        type=float,
        help="Robot-base point in mm for the generated overview camera",
    )
    parser.set_defaults(show_obstacles=True)
    parser.add_argument(
        "--show-obstacles",
        action="store_true",
        help="Render patientObstacle_* boxes (default)",
    )
    parser.add_argument(
        "--hide-obstacles",
        action="store_false",
        dest="show_obstacles",
        help="Hide patientObstacle_* boxes while retaining collision checks",
    )
    parser.add_argument(
        "--initial-joint-angle-rad",
        action="append",
        type=float,
        dest="initial_joint_angles_rad",
        help="Override one of six scene start angles; repeat six times.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    create_scene(
        output_path=args.output,
        obstacle_path=args.obstacles,
        max_obstacles=args.max_obstacles,
        host=args.host,
        port=args.port,
        initial_joint_angles_rad=(
            args.initial_joint_angles_rad
            if args.initial_joint_angles_rad is not None
            else DEFAULT_INITIAL_JOINT_ANGLES_RAD
        ),
        view_target_robot_base_mm=args.view_target_mm,
        show_obstacles=args.show_obstacles,
        ur5_model_path=args.ur5_model,
        initial_tool_point_mm=args.initial_tool_point_mm,
        initial_needle_axis=args.initial_needle_axis,
        max_cartesian_step_mm=args.max_cartesian_step_mm,
        max_trajectory_steps=args.max_trajectory_steps,
        joint_limit_margin_rad=args.joint_limit_margin_rad,
        max_joint_step_rad=args.max_joint_step_rad,
        simik_damping=args.simik_damping,
        simik_max_iterations=args.simik_max_iterations,
    )
