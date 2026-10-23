import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robot_kinematics import (
    UR5Kinematics,
    build_needle_pose,
    needle_tip_from_tcp_pose,
)

ur5 = UR5Kinematics()

target_tip = np.array([300.0, 0.0, 300.0])
needle_axis = np.array([0.0, 0.0, 1.0])
needle_length = 220.0

target_pose = build_needle_pose(
    needle_tip=target_tip,
    direction_vector=needle_axis,
    needle_tip_offset_mm=needle_length,
)

print("Target TCP:", target_pose[:3, 3])
print("Needle axis:", needle_axis)

rng = np.random.default_rng(42)

seeds = [
    np.array([
        0.0,
        -np.pi / 4,
        np.pi / 2,
        -np.pi / 4,
        np.pi / 2,
        0.0,
    ])
]

for _ in range(60):
    seeds.append(rng.uniform(-np.pi, np.pi, 6))

best = None
best_error = float("inf")

for i, seed in enumerate(seeds):
    q, converged, pos_error = ur5.solve_ik_numerical(
        target_pose,
        initial_q=seed,
        max_iters=500,
    )

    fk_pose = ur5.forward_kinematics(q)
    fk_tip = needle_tip_from_tcp_pose(
        fk_pose, needle_length
    )

    tip_error = np.linalg.norm(fk_tip - target_tip)

    if tip_error < best_error:
        best_error = tip_error
        best = (i, q, converged, pos_error, tip_error)

    if converged and tip_error <= 0.5:
        print("\nIK SUCCESS")
        print("Seed index:", i)
        print("TCP position error:", pos_error)
        print("Needle-tip error:", tip_error)
        print("Joint angles (deg):", np.rad2deg(q))
        break
else:
    print("\nNo converged solution found.")

print("\nBest result:")
i, q, converged, pos_error, tip_error = best

print("Seed index:", i)
print("Converged:", converged)
print("TCP position error:", pos_error)
print("Needle-tip error:", tip_error)
print("Joint angles (deg):", np.rad2deg(q))