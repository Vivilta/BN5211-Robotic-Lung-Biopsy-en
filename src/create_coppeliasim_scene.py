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
from pathlib import Path

import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from robot_geometry import transform_point


UR5_MODEL = "/Users/vivi/Downloads/coppeliaSim.app/Contents/Resources/models/robots/non-mobile/UR5.ttm"
DEFAULT_OUTPUT = "scenes/robotic_lung_biopsy.ttt"
DEFAULT_OBSTACLES = "data/demo_nodules/rib_obstacles_0037.json"
DEFAULT_INITIAL_JOINT_ANGLES_RAD = (0.0, -math.pi / 2.0, 0.0, -math.pi / 2.0, 0.0, 0.0)


def _transform_obstacle(obstacle, patient_to_robot_transform):
    if patient_to_robot_transform is None:
        return obstacle
    minimum = np.asarray(obstacle["min_mm"], dtype=float)
    maximum = np.asarray(obstacle["max_mm"], dtype=float)
    corners = [
        transform_point(
            patient_to_robot_transform,
            [x, y, z],
        )
        for x in (minimum[0], maximum[0])
        for y in (minimum[1], maximum[1])
        for z in (minimum[2], maximum[2])
    ]
    transformed = dict(obstacle)
    transformed["min_mm"] = np.min(corners, axis=0).tolist()
    transformed["max_mm"] = np.max(corners, axis=0).tolist()
    transformed["coordinate_system"] = "UR5 base coordinates in mm"
    return transformed


def _load_obstacles(path, max_obstacles, patient_to_robot_transform=None):
    if not path:
        return []
    with open(path, "r", encoding="utf-8") as handle:
        obstacles = json.load(handle)
    if not isinstance(obstacles, list):
        raise ValueError("Obstacle JSON must contain a list")
    if max_obstacles is not None and max_obstacles > 0:
        obstacles = obstacles[:max_obstacles]
    return [_transform_obstacle(item, patient_to_robot_transform) for item in obstacles]


def _set_shape_color(sim, handle, rgb):
    sim.setShapeColor(handle, "", sim.colorcomponent_ambient_diffuse, list(rgb))


def _create_box(sim, name, minimum_mm, maximum_mm, rgb):
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
    return handle


def _create_needle_and_ik(sim, flange_handle, robot_handle):
    needle_length = 0.30
    needle_radius = 0.004
    needle = sim.createPrimitiveShape(
        sim.primitiveshape_cylinder,
        [needle_radius * 2.0, needle_radius * 2.0, needle_length],
        0,
    )
    sim.setObjectAlias(needle, "needle")
    sim.setObjectParent(needle, flange_handle, False)
    sim.setObjectPose(needle, [0.0, 0.0, needle_length / 2.0, 0.0, 0.0, 0.0, 1.0], flange_handle)
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
):
    output_path = os.path.abspath(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    obstacles = _load_obstacles(obstacle_path, max_obstacles, patient_to_robot_transform)

    client = RemoteAPIClient(host, port)
    sim = client.require("sim")
    if sim.getSimulationState() != sim.simulation_stopped:
        sim.stopSimulation()
    sim.closeScene()

    ur5 = sim.loadModel(UR5_MODEL)
    sim.setObjectAlias(ur5, "UR5")
    joints = sim.getObjectsInTree(ur5, sim.object_joint_type, 0)
    if len(joints) != 6:
        raise RuntimeError(f"Expected 6 UR5 joints, found {len(joints)}")
    if len(initial_joint_angles_rad) != 6:
        raise ValueError("initial_joint_angles_rad must contain exactly six values")
    for joint in joints:
        sim.setJointMode(joint, sim.jointmode_kinematic, 0)
    for joint, angle in zip(joints, initial_joint_angles_rad):
        sim.setJointPosition(joint, float(angle))

    # The final visible link is the child of the sixth joint in the stock model.
    final_joint = joints[-1]
    descendants = sim.getObjectsInTree(final_joint, sim.object_shape_type, 0)
    if not descendants:
        raise RuntimeError("Could not locate the UR5 wrist/flange shape")
    flange = descendants[-1]
    _, tip, target = _create_needle_and_ik(sim, flange, ur5)

    robot_collection = sim.createCollection(0)
    sim.setCollectionName(robot_collection, "robotCollision")
    sim.addItemToCollection(robot_collection, sim.handle_tree, ur5, 0)

    patient_collection = sim.createCollection(0)
    sim.setCollectionName(patient_collection, "patientCollision")
    for index, obstacle in enumerate(obstacles):
        box = _create_box(
            sim,
            f"patientObstacle_{index + 1}",
            obstacle["min_mm"],
            obstacle["max_mm"],
            (0.85, 0.28, 0.18),
        )
        sim.addItemToCollection(patient_collection, sim.handle_single, box, 0)

    sim.saveScene(output_path)
    print(f"Saved scene: {output_path}")
    print(f"UR5 joints: {len(joints)}")
    print(f"IK tip: /UR5/ikTip; IK target: /UR5/ikTarget")
    print(f"Collections: /robotCollision and /patientCollision ({len(obstacles)} obstacles)")
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
    )
