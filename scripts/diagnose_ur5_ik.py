import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robot_kinematics import (
    UR5Kinematics,
    build_needle_pose,
    needle_tip_from_tcp_pose,
)

ur5 = UR5Kinematics()

# Test A: FK -> IK -> FK
q_true = np.array([
    0.0,
    -np.pi / 3,
    np.pi / 2,
    -np.pi / 3,
    np.pi / 2,
    0.0,
])

reachable_pose = ur5.forward_kinematics(q_true)

print("\n=== Test A: FK -> IK -> FK ===")

for name, initial_q in [
    ("near_true", q_true + 0.1),
    ("default", None),
]:
    q, converged, error = ur5.solve_ik_numerical(
        reachable_pose,
        initial_q=initial_q,
    )

    print(
        name,
        "converged:", converged,
        "position_error_mm:", round(error, 4),
    )


# Test B: Actual 0037 target
print("\n=== Test B: Case 0037 Multi-start IK ===")

target_tip = np.array([300.0, 0.0, 300.0])
direction = np.array([0.0, 1.0, 0.0])
needle_length = 220.0

target_pose = build_needle_pose(
    target_tip,
    direction,
    needle_tip_offset_mm=needle_length,
)

print("Target TCP:", target_pose[:3, 3])

rng = np.random.default_rng(42)

seeds = [
    np.array([
        0.0, -np.pi / 4, np.pi / 2,
        -np.pi / 4, np.pi / 2, 0.0
    ])
]

for _ in range(30):
    seeds.append(rng.uniform(-np.pi, np.pi, 6))

best_error = float("inf")
best_result = None

for i, seed in enumerate(seeds):
    q, converged, pos_error = ur5.solve_ik_numerical(
        target_pose,
        initial_q=seed,
        max_iters=500,
    )

    fk_pose = ur5.forward_kinematics(q)

    tip = needle_tip_from_tcp_pose(
        fk_pose,
        needle_length,
    )

    tip_error = np.linalg.norm(tip - target_tip)

    if tip_error < best_error:
        best_error = tip_error
        best_result = (i, converged, pos_error, tip_error, q)

    if converged and tip_error <= 0.5:
        print(
            f"SUCCESS seed={i}, "
            f"position_error={pos_error:.4f} mm, "
            f"tip_error={tip_error:.4f} mm"
        )
        break

print("\nBest result:")
i, converged, pos_error, tip_error, q = best_result

print("Seed index:", i)
print("Converged:", converged)
print("TCP position error:", pos_error)
print("Needle-tip error:", tip_error)
print("Joint angles (deg):", np.rad2deg(q))


# Test C: Needle geometry consistency
print("\n=== Test C: Needle Geometry ===")

recovered_tip = needle_tip_from_tcp_pose(
    target_pose,
    needle_length,
)

print("Original tip:", target_tip)
print("Recovered tip:", recovered_tip)
print(
    "Geometry error:",
    np.linalg.norm(recovered_tip - target_tip),
)