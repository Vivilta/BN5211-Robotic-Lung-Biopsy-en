import json
import numpy as np
from pathlib import Path

config_path = Path(
    "data/registration_candidates/cases_candidates.json"
)

with open(config_path, encoding="utf-8") as f:
    config = json.load(f)

# AI gate
config["defaults"]["force_robot_demo"] = False
config["defaults"]["malignancy_threshold"] = 0.5

# Require explicit transform
config["defaults"]["allow_coincident_demo_frame"] = False

mount_height = float(
    config["defaults"].get("overhead_mount_height_mm", 200.0)
)

# Initial demonstration target position in UR5 base coordinates
desired_target_robot = np.array(
    [300.0, 0.0, 300.0],
    dtype=float
)

for case in config["cases"]:
    case_id = case["case_id"]

    if case_id not in [
        "LIDC-IDRI-0036",
        "LIDC-IDRI-0037"
    ]:
        continue

    patient_target = np.asarray(
        case["target_mm"],
        dtype=float
    )

    translation = (
        desired_target_robot
        - patient_target
        - np.array([0.0, 0.0, mount_height])
    )

    transform = np.eye(4, dtype=float)
    transform[:3, 3] = translation

    case["patient_to_robot_base"] = transform.tolist()

    print(f"\n{case_id}")
    print("Patient target:", patient_target)
    print("Configured translation:", translation)
    print("Expected robot target:", desired_target_robot)

with open(config_path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print("\nUpdated:", config_path)