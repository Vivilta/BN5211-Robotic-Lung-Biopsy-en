import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robot_geometry import rotation_from_z_axis, rotation_to_quaternion_xyzw, transform_point
from robot_kinematics import build_needle_pose, needle_tip_from_tcp_pose
from trajectory_planner import BiopsyTrajectoryPlanner, ray_intersects_aabb
from coppeliasim_remote import _polyline_samples


class RobotGeometryTests(unittest.TestCase):
    def test_rotation_and_quaternion(self):
        rotation = rotation_from_z_axis([1.0, 2.0, 3.0])
        self.assertTrue(np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8))
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=8)
        quaternion = rotation_to_quaternion_xyzw(rotation)
        self.assertAlmostEqual(float(np.linalg.norm(quaternion)), 1.0, places=8)

    def test_transform_point(self):
        transform = np.eye(4)
        transform[:3, 3] = [10.0, -2.0, 4.0]
        np.testing.assert_allclose(transform_point(transform, [1.0, 2.0, 3.0]), [11.0, 0.0, 7.0])

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


if __name__ == "__main__":
    unittest.main()
