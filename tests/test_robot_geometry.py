import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robot_geometry import (
    overhead_mount_base_to_world,
    patient_to_robot_base_for_mount_height,
    quaternion_xyzw_to_rotation,
    rotation_from_z_axis,
    rotation_to_quaternion_xyzw,
    rotation_with_z_axis_roll,
    transform_direction,
    transform_point,
)
from robot_kinematics import UR5Kinematics, build_needle_pose, needle_tip_from_tcp_pose
from trajectory_planner import BiopsyTrajectoryPlanner, ray_intersects_aabb
from coppeliasim_remote import (
    _colliding_patient_obstacles,
    _collision_object_names,
    _ik_status_ok,
    _polyline_samples,
    _target_pose_mm,
)


class RobotGeometryTests(unittest.TestCase):
    def test_collision_result_reports_object_aliases(self):
        class FakeSim:
            @staticmethod
            def getObjectAlias(handle, options):
                return {11: "needle", 22: "patientObstacle_17"}[handle]

        self.assertEqual(
            _collision_object_names(FakeSim(), [1, 11, 22]),
            ["needle", "patientObstacle_17"],
        )
        self.assertEqual(
            _collision_object_names(FakeSim(), (1, [11, 22])),
            ["needle", "patientObstacle_17"],
        )
        self.assertEqual(_collision_object_names(FakeSim(), [0]), [])

    def test_collision_fallback_identifies_patient_obstacle(self):
        class FakeSim:
            @staticmethod
            def checkCollision(robot_collection, obstacle_handle):
                return obstacle_handle == 22

            @staticmethod
            def getObjectAlias(handle, options):
                return f"patientObstacle_{handle}"

        self.assertEqual(
            _colliding_patient_obstacles(FakeSim(), 1, [11, 22, 33]),
            ["patientObstacle_22"],
        )

    def test_rotation_and_quaternion(self):
        rotation = rotation_from_z_axis([1.0, 2.0, 3.0])
        self.assertTrue(np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8))
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=8)
        quaternion = rotation_to_quaternion_xyzw(rotation)
        self.assertAlmostEqual(float(np.linalg.norm(quaternion)), 1.0, places=8)
        np.testing.assert_allclose(quaternion_xyzw_to_rotation(quaternion), rotation, atol=1e-8)

    def test_tool_roll_preserves_needle_axis(self):
        axis = np.array([0.2, -0.3, 0.9])
        axis /= np.linalg.norm(axis)
        for roll in np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False):
            rotation = rotation_with_z_axis_roll(axis, roll)
            np.testing.assert_allclose(rotation[:, 2], axis, atol=1e-8)
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
            self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=8)

    def test_remote_target_preserves_scene_roll_and_rejects_wrong_axis(self):
        axis = np.array([0.0, 0.0, 1.0])
        rotation = rotation_with_z_axis_roll(axis, np.deg2rad(60.0))
        quaternion = rotation_to_quaternion_xyzw(rotation)
        pose = _target_pose_mm([10.0, 20.0, 30.0], axis, quaternion)
        np.testing.assert_allclose(pose[:3], [0.01, 0.02, 0.03])
        np.testing.assert_allclose(pose[3:], quaternion)
        with self.assertRaises(ValueError):
            _target_pose_mm([0.0, 0.0, 0.0], [0.0, 0.0, -1.0], quaternion)

    def test_transform_point(self):
        transform = np.eye(4)
        transform[:3, 3] = [10.0, -2.0, 4.0]
        np.testing.assert_allclose(transform_point(transform, [1.0, 2.0, 3.0]), [11.0, 0.0, 7.0])

    def test_prone_demo_transform_points_planned_inward_axis_downward(self):
        rotation = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        patient_target = np.array([118.274994, 20.625, -163.5])
        robot_target = np.array([118.274994, 20.625, 400.0])
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = robot_target - rotation @ patient_target

        np.testing.assert_allclose(transform_point(transform, patient_target), robot_target)
        np.testing.assert_allclose(rotation @ [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
        world_transform = overhead_mount_base_to_world()
        np.testing.assert_allclose(transform_direction(world_transform, [0.0, 0.0, 1.0]), [0.0, 0.0, -1.0])
        np.testing.assert_allclose(transform_point(world_transform, robot_target), [118.274994, -20.625, -400.0])

        planner_seed = [2.173, -1.094, 2.794, -0.129, -1.571, 0.969]
        kinematics = UR5Kinematics()
        for z in (310.0, 340.0, 400.0, 500.0):
            pose = build_needle_pose([*robot_target[:2], z], [0.0, 0.0, 1.0], 150.0)
            planner_seed, converged, error = kinematics.solve_ik_numerical(
                pose,
                initial_q=planner_seed,
                max_iters=1000,
                orientation_tol=0.005,
            )
            self.assertTrue(converged)
            self.assertLess(error, 0.5)

    def test_raised_demo_mount_preserves_patient_world_geometry(self):
        patient_to_robot = np.eye(4)
        patient_to_robot[:3, 3] = [40.0, -20.0, 400.0]
        raised_transform = patient_to_robot_base_for_mount_height(patient_to_robot, 100.0)
        point_patient = np.array([-74.0, 80.0, -128.0])
        world_before = transform_point(
            overhead_mount_base_to_world(),
            transform_point(patient_to_robot, point_patient),
        )
        world_after = transform_point(
            overhead_mount_base_to_world(100.0),
            transform_point(raised_transform, point_patient),
        )
        np.testing.assert_allclose(world_after, world_before, atol=1e-9)

    def test_needle_tip_offset(self):
        pose = build_needle_pose([10.0, 20.0, 30.0], [0.0, 0.0, 1.0], needle_tip_offset_mm=5.0)
        np.testing.assert_allclose(needle_tip_from_tcp_pose(pose, 5.0), [10.0, 20.0, 30.0])

    def test_segment_aabb_intersection(self):
        self.assertTrue(ray_intersects_aabb(np.array([-1.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0]), np.zeros(3), np.ones(3)))
        self.assertFalse(ray_intersects_aabb(np.array([-1.0, 2.0, 0.0]), np.array([2.0, 2.0, 0.0]), np.zeros(3), np.ones(3)))

    def test_planner_requires_aabb_list(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "obstacles.json"
            path.write_text(json.dumps([{"id": "rib", "min_mm": [0, 0, 0], "max_mm": [1, 1, 1]}]), encoding="utf-8")
            planner = BiopsyTrajectoryPlanner(str(path))
            collision, obstacle_id = planner.check_collision([-1, 0.5, 0.5], [2, 0.5, 0.5])
            self.assertTrue(collision)
            self.assertEqual(obstacle_id, "rib")

    def test_planner_inflates_obstacles_by_needle_radius_and_clearance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "obstacles.json"
            path.write_text(
                json.dumps([{"id": "rib", "min_mm": [0, 0, 0], "max_mm": [1, 1, 1]}]),
                encoding="utf-8",
            )
            planner = BiopsyTrajectoryPlanner(
                str(path), safety_margin_mm=2.0, needle_radius_mm=4.0
            )
            collision, _ = planner.check_collision([-5.9, 0.5, 0.5], [-5.9, 0.5, 0.5])
            self.assertTrue(collision)
            collision, _ = planner.check_collision([-6.1, 0.5, 0.5], [-6.1, 0.5, 0.5])
            self.assertFalse(collision)

    def test_multidirectional_planner_checks_full_needle_sweep(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "obstacles.json"
            path.write_text(
                json.dumps([{"id": "rib", "min_mm": [-1, -2, -1], "max_mm": [1, -1, 1]}]),
                encoding="utf-8",
            )
            planner = BiopsyTrajectoryPlanner(str(path), safety_margin_mm=0.0)
            result = planner.plan_multidirectional(
                [0.0, 10.0, 0.0],
                entry_distances_mm=(10.0,),
                angular_step_deg=90.0,
                preferred_direction=[0.0, -1.0, 0.0],
                max_direction_deviation_deg=1.0,
                approach_distance_mm=1.0,
                needle_length_mm=20.0,
            )
            self.assertIsNone(result)

    def test_multidirectional_planner_finds_a_clear_direction(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "obstacles.json"
            path.write_text(
                json.dumps([{"id": "front_wall", "min_mm": [-5, -5, -5], "max_mm": [5, 5, 5]}]),
                encoding="utf-8",
            )
            planner = BiopsyTrajectoryPlanner(str(path), safety_margin_mm=0.0)
            result = planner.plan_multidirectional(
                [0.0, 0.0, 20.0],
                entry_distances_mm=(30.0,),
                angular_step_deg=30.0,
            )
            self.assertIsNotNone(result)
            self.assertEqual(result["planning_mode"], "multi_direction_radial")

    def test_multidirectional_planner_rejects_target_inside_obstacle(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "obstacles.json"
            path.write_text(
                json.dumps([{"id": "rib", "min_mm": [-1, -1, -1], "max_mm": [1, 1, 1]}]),
                encoding="utf-8",
            )
            planner = BiopsyTrajectoryPlanner(str(path), safety_margin_mm=0.0)
            self.assertIsNone(
                planner.plan_multidirectional([0.0, 0.0, 0.0], entry_distances_mm=(30.0,))
            )

    def test_polyline_samples_preserve_approach_and_insertion_phases(self):
        waypoints, phases = _polyline_samples(
            [
                np.array([0.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 3.0]),
                np.array([0.0, 0.0, 5.0]),
                np.array([0.0, 0.0, 10.0]),
            ],
            total_steps=10,
        )
        self.assertEqual(len(waypoints), 10)
        self.assertEqual(len(phases), 10)
        self.assertEqual(phases[0], "initial")
        self.assertIn("approach", phases)
        self.assertIn("approach_to_entry", phases)
        self.assertIn("insertion", phases)
        np.testing.assert_allclose(waypoints[0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(waypoints[-1], [0.0, 0.0, 10.0])

    def test_simik_status_only_accepts_explicit_success(self):
        class SimIK:
            result_success = 1

        self.assertTrue(_ik_status_ok(SimIK(), (1, 0, [0.0, 0.0])))
        self.assertFalse(_ik_status_ok(SimIK(), (0, 0, [0.0, 0.0])))
        self.assertFalse(_ik_status_ok(SimIK(), (2, 0, [0.0, 0.0])))


if __name__ == "__main__":
    unittest.main()
