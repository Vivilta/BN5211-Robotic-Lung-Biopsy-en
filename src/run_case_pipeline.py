"""Run the reusable per-case CT-to-UR5 demonstration pipeline.

The code and execution order are fixed. Case-specific inputs live in a JSON
configuration file: ROI cube, exact DICOM series, SeriesInstanceUID, and the
ROI centre voxel. By default this script prepares offline artifacts. Add
``--run-simulation`` to generate a CoppeliaSim scene and execute it remotely.

This is a software demonstration pipeline. Unless a measured 4x4
patient-to-robot transform is supplied, the identity transform is allowed only
with the explicit ``allow_coincident_demo_frame`` configuration flag.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from environment_modeling import VirtualChestEnvironment
from main_pipeline import run_end_to_end_biopsy_pipeline


DEFAULT_CHECKPOINT = (
    "/Users/vivi/Documents/BN5211/Final Project/训练好的/"
    "3d_resnet_cbam-epoch=29-val_acc=0.84.ckpt"
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be a JSON object: {path}")
    return value


def _select_case_config(raw: dict[str, Any], case_id: str | None) -> dict[str, Any]:
    """Merge a shared defaults block with one case entry."""
    cases = raw.get("cases")
    if cases is None:
        return raw
    if not isinstance(cases, list) or not all(isinstance(item, dict) for item in cases):
        raise ValueError("cases must be a JSON list of objects")
    if not case_id:
        if len(cases) != 1:
            raise ValueError("This configuration contains multiple cases; pass --case-id")
        selected = cases[0]
    else:
        selected = next((item for item in cases if item.get("case_id") == case_id), None)
        if selected is None:
            available = ", ".join(str(item.get("case_id")) for item in cases)
            raise ValueError(f"Unknown case_id {case_id!r}; available: {available}")
    merged = dict(raw.get("defaults", {}))
    merged.update(selected)
    return merged


def _resolve_path(value: str | None, config_path: Path, repo_root: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    # Relative paths are convenient in a shared repository config. Resolve
    # them relative to the repository root, with config-relative paths as a
    # fallback for configs kept outside the repository.
    repo_candidate = (repo_root / path).resolve()
    if repo_candidate.exists():
        return repo_candidate
    return (config_path.parent / path).resolve()


def _case_suffix(case_id: str) -> str:
    match = re.search(r"(\d+)$", case_id)
    return match.group(1) if match else re.sub(r"[^A-Za-z0-9]+", "_", case_id).strip("_")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _validate_roi(roi_path: Path) -> None:
    if not roi_path.is_file():
        raise FileNotFoundError(f"ROI cube not found: {roi_path}")
    cube = np.load(roi_path, mmap_mode="r")
    if cube.shape != (32, 32, 32):
        raise ValueError(f"Expected ROI shape (32, 32, 32), got {cube.shape}: {roi_path}")
    if not np.isfinite(cube).all():
        raise ValueError(f"ROI contains NaN or infinite values: {roi_path}")


def _validate_transform(value: Any, label: str) -> np.ndarray:
    transform = np.asarray(value, dtype=float)
    if transform.shape != (4, 4):
        raise ValueError(f"{label} must be a 4x4 matrix, got {transform.shape}")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError(f"{label} must have homogeneous last row [0, 0, 0, 1]")
    return transform


def _load_or_extract_obstacles(
    env: VirtualChestEnvironment,
    config: dict[str, Any],
    case_id: str,
    target_voxel: np.ndarray,
    obstacle_path: Path,
    preview_path: Path,
    skip_existing: bool,
) -> list[dict[str, Any]]:
    if skip_existing and obstacle_path.is_file():
        with obstacle_path.open("r", encoding="utf-8") as handle:
            obstacles = json.load(handle)
        if not isinstance(obstacles, list) or not obstacles:
            raise ValueError(f"Existing obstacle JSON is empty or invalid: {obstacle_path}")
        uid_values = {item.get("series_instance_uid") for item in obstacles if isinstance(item, dict)}
        if uid_values != {env.series_instance_uid}:
            raise ValueError(
                f"Existing obstacle JSON does not belong to the configured SeriesInstanceUID: "
                f"{obstacle_path}"
            )
        print(f"[CASE] Reusing obstacle JSON: {obstacle_path}")
        return obstacles

    obstacle_options = config.get("obstacle_extraction", {})
    if not isinstance(obstacle_options, dict):
        raise ValueError("obstacle_extraction must be a JSON object")
    boxes = env.extract_local_rib_bounding_boxes(
        target_voxel=target_voxel,
        bone_hu_threshold=float(obstacle_options.get("bone_hu_threshold", 500.0)),
        min_volume_voxels=int(obstacle_options.get("min_volume_voxels", 15)),
        x_min_offset_mm=float(obstacle_options.get("x_min_offset_mm", -35.0)),
        x_max_offset_mm=float(obstacle_options.get("x_max_offset_mm", 140.0)),
        y_half_range_mm=float(obstacle_options.get("y_half_range_mm", 150.0)),
        z_half_range_mm=float(obstacle_options.get("z_half_range_mm", 80.0)),
        segment_slices=int(obstacle_options.get("segment_slices", 1)),
        exclude_target_roi_half_size_voxels=int(
            obstacle_options.get("exclude_target_roi_half_size_voxels", 16)
        ),
        case_id=case_id,
    )
    if not boxes:
        raise RuntimeError("No local rib obstacles were extracted; refusing to continue")
    _write_json(obstacle_path, boxes)
    env.save_local_extraction_preview(preview_path, target_voxel, boxes)
    return boxes


def run_case(
    config_path: str | os.PathLike[str],
    case_id: str | None = None,
    run_simulation: bool = False,
    skip_existing: bool = False,
    frame_delay: float = 0.08,
    confirm_coincident_demo_frame: bool = False,
) -> dict[str, Any]:
    config_file = Path(config_path).expanduser().resolve()
    config = _select_case_config(_load_json(config_file), case_id)
    repo_root = config_file.parent.parent if config_file.parent.name == "configs" else config_file.parent

    case_id = str(config.get("case_id", "")).strip()
    if not case_id:
        raise ValueError("case_id is required")
    suffix = _case_suffix(case_id)
    roi_path = _resolve_path(config.get("roi_path"), config_file, repo_root)
    ct_path = _resolve_path(config.get("ct_series_path"), config_file, repo_root)
    checkpoint_path = _resolve_path(config.get("checkpoint_path", DEFAULT_CHECKPOINT), config_file, repo_root)
    if roi_path is None or ct_path is None or checkpoint_path is None:
        raise ValueError("roi_path, ct_series_path, and checkpoint_path are required")
    if not ct_path.exists():
        raise FileNotFoundError(f"CT series path not found: {ct_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    _validate_roi(roi_path)

    target_voxel = np.asarray(config.get("target_voxel"), dtype=int)
    if target_voxel.shape != (3,):
        raise ValueError("target_voxel must contain [x, y, z]")
    expected_uid = str(config.get("series_instance_uid", "")).strip()
    if not expected_uid:
        raise ValueError("series_instance_uid is required")

    output_dir = _resolve_path(config.get("output_dir", "data/demo_nodules"), config_file, repo_root)
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)
    obstacle_path = output_dir / f"rib_obstacles_{suffix}_segmented.json"
    preview_path = output_dir / f"rib_obstacles_{suffix}_segmented_preview.png"
    plan_path = output_dir / f"planned_trajectory_{suffix}.json"
    run_path = output_dir / f"coppeliasim_run_{suffix}_segmented.json"
    scene_path = repo_root / "scenes" / f"robotic_lung_biopsy_{suffix}_segmented.ttt"
    summary_path = output_dir / f"pipeline_summary_{suffix}.json"

    print(f"\n{'=' * 72}\n[CASE] {case_id}\n{'=' * 72}")
    print(f"[CASE] ROI: {roi_path}")
    print(f"[CASE] CT series: {ct_path}")

    env = VirtualChestEnvironment(str(ct_path))
    if env.series_instance_uid != expected_uid:
        raise ValueError(
            "SeriesInstanceUID mismatch: "
            f"selected={env.series_instance_uid}, expected={expected_uid}"
        )
    target_mm = np.asarray(env.voxel_to_world(target_voxel), dtype=float)
    configured_target_mm = config.get("target_mm")
    if configured_target_mm is not None:
        configured = np.asarray(configured_target_mm, dtype=float)
        if configured.shape != (3,) or not np.allclose(configured, target_mm, atol=1e-2):
            raise ValueError(
                "Configured target_mm disagrees with target_voxel in the selected DICOM series: "
                f"configured={configured.tolist()}, derived={target_mm.tolist()}"
            )

    obstacles = _load_or_extract_obstacles(
        env=env,
        config=config,
        case_id=case_id,
        target_voxel=target_voxel,
        obstacle_path=obstacle_path,
        preview_path=preview_path,
        skip_existing=skip_existing,
    )

    demo_frame_allowed = bool(config.get("allow_coincident_demo_frame", False)) or confirm_coincident_demo_frame
    transform_path = _resolve_path(config.get("patient_to_robot_transform_path"), config_file, repo_root)
    configured_transform = config.get("patient_to_robot_base")
    if transform_path is not None and configured_transform is not None:
        raise ValueError("Use either patient_to_robot_transform_path or patient_to_robot_base, not both")
    if transform_path is None and configured_transform is None and not demo_frame_allowed:
        raise RuntimeError(
            "No measured patient-to-robot transform is configured. For this software demo, "
            "set allow_coincident_demo_frame=true or pass --confirm-coincident-demo-frame."
        )

    if transform_path is not None:
        with transform_path.open("r", encoding="utf-8") as handle:
            patient_to_robot_base = _validate_transform(json.load(handle), "patient-to-robot transform")
        transform_mode = "measured_4x4"
    elif configured_transform is not None:
        patient_to_robot_base = _validate_transform(configured_transform, "patient_to_robot_base")
        transform_mode = "configured_demo_transform"
    else:
        patient_to_robot_base = np.eye(4, dtype=float)
        transform_mode = "coincident_demo_identity"

    if skip_existing and plan_path.is_file():
        with plan_path.open("r", encoding="utf-8") as handle:
            plan = json.load(handle)
        if plan.get("decision") != "robot_plan_ready":
            raise ValueError(f"Existing plan is not executable: {plan_path}")
        print(f"[CASE] Reusing plan: {plan_path}")
    else:
        plan = run_end_to_end_biopsy_pipeline(
            ckpt_path=str(checkpoint_path),
            roi_cube_path=str(roi_path),
            target_coords_mm=target_mm,
            obstacles_json_path=str(obstacle_path),
            malignancy_threshold=float(config.get("malignancy_threshold", 0.45)),
            output_plan_path=str(plan_path),
            patient_to_robot_base=patient_to_robot_base,
            needle_tip_offset_mm=float(config.get("needle_tip_offset_mm", 0.0)),
            force_robot_demo=bool(config.get("force_robot_demo", True)),
            patient_to_robot_transform_confirmed=True,
            patient_to_robot_transform_path=str(transform_path) if transform_path else None,
            target_roi_match_confirmed=True,
            approach_distance_mm=float(config.get("approach_distance_mm", 30.0)),
            planning_mode=str(config.get("planning_mode", "multi_direction")),
            entry_distances_mm=tuple(config.get("entry_distances_mm", [60.0, 80.0, 100.0])),
            angular_step_deg=float(config.get("angular_step_deg", 15.0)),
        )
    if plan.get("decision") != "robot_plan_ready":
        raise RuntimeError(f"Planning did not produce an executable plan: {plan.get('decision')}")

    simulation = None
    if run_simulation:
        # Import only when requested so offline preparation does not require
        # CoppeliaSim's bundled ZeroMQ client on PYTHONPATH.
        from create_coppeliasim_scene import create_scene
        from coppeliasim_remote import run_remote_simulation

        create_scene(
            output_path=str(scene_path),
            obstacle_path=str(obstacle_path),
            max_obstacles=int(config.get("max_scene_obstacles", 0)),
            host=str(config.get("coppeliasim_host", "127.0.0.1")),
            port=int(config.get("coppeliasim_port", 23000)),
            patient_to_robot_transform=patient_to_robot_base,
        )
        simulation = run_remote_simulation(
            plan_path=str(plan_path),
            scene_path=str(scene_path),
            load_scene=True,
            host=str(config.get("coppeliasim_host", "localhost")),
            port=int(config.get("coppeliasim_port", 23000)),
            robot_path=str(config.get("robot_path", "/UR5")),
            ik_tip_path=str(config.get("ik_tip_path", "/UR5/ikTip")),
            ik_target_path=str(config.get("ik_target_path", "/UR5/ikTarget")),
            robot_collision_collection=str(config.get("robot_collision_collection", "/robotCollision")),
            environment_collision_collection=str(config.get("environment_collision_collection", "/patientCollision")),
            steps=int(config.get("simulation_steps", 100)),
            kinematic_only=True,
            frame_delay_s=frame_delay,
            output_path=str(run_path),
        )

    summary = {
        "case_id": case_id,
        "series_instance_uid": env.series_instance_uid,
        "roi_path": str(roi_path),
        "ct_series_path": str(ct_path),
        "checkpoint_path": str(checkpoint_path),
        "target_voxel_xyz": target_voxel.tolist(),
        "target_point_patient_mm": target_mm.tolist(),
        "obstacle_count": len(obstacles),
        "obstacle_json": str(obstacle_path),
        "obstacle_preview_png": str(preview_path),
        "plan_json": str(plan_path),
        "scene_ttt": str(scene_path) if run_simulation else None,
        "simulation_json": str(run_path) if simulation is not None else None,
        "patient_to_robot_transform_mode": transform_mode,
        "planning_decision": plan.get("decision"),
        "malignancy_probability": plan.get("malignancy_probability"),
        "analytical_ik": plan.get("analytical_ik"),
        "simulation": simulation,
    }
    _write_json(summary_path, summary)
    print(f"[CASE] Summary: {summary_path}")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Case JSON configuration")
    parser.add_argument("--case-id", help="Case ID when --config contains a cases list")
    parser.add_argument("--run-simulation", action="store_true", help="Generate .ttt and execute CoppeliaSim")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing obstacle JSON and plan JSON")
    parser.add_argument("--frame-delay", type=float, default=0.08, help="Seconds between visible simulation frames")
    parser.add_argument(
        "--confirm-coincident-demo-frame",
        action="store_true",
        help="Allow identity CT-to-UR5 registration for this non-clinical software demo",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_case(
        config_path=args.config,
        case_id=args.case_id,
        run_simulation=args.run_simulation,
        skip_existing=args.skip_existing,
        frame_delay=args.frame_delay,
        confirm_coincident_demo_frame=args.confirm_coincident_demo_frame,
    )
