import json
import numpy as np
from pathlib import Path

config_path = Path(
    "data/registration_candidates/cases_supine_demo.json"
)

with open(config_path, encoding="utf-8") as f:
    config = json.load(f)

# Patient LPS -> Robot Base rotation
R = np.array([
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
])

assert np.allclose(R.T @ R, np.eye(3))
assert np.isclose(np.linalg.det(R), 1.0)

# Desired target position in robot base coordinates
desired_target = np.array([300.0, 0.0, 300.0])

mount_height = float(
    config["defaults"].get(
        "overhead_mount_height_mm", 200.0
    )
)

for case in config["cases"]:
    if case["case_id"] != "LIDC-IDRI-0037":
        continue

    target_patient = np.array(
        case["target_mm"], dtype=float
    )

    # run_case_pipeline applies +mount_height to base Z
    translation = (
        desired_target
        - R @ target_patient
        - np.array([0.0, 0.0, mount_height])
    )

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = translation

    case["patient_to_robot_base"] = T.tolist()

    # The old IK seed was found for a different orientation.
    case.pop("initial_ik_seed_rad", None)

    print("Case:", case["case_id"])
    print("Rotation:\n", R)
    print("Translation:", translation)
    print("Configured transform:\n", T)

with open(config_path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print("Saved:", config_path)