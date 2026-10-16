# Per-case Robotic Lung Biopsy Pipeline

This document describes the reusable robot-side workflow. The pipeline is
case-configured, but the execution order and source code are fixed.

## 1. What changes for every case?

The following inputs are case-specific and must be supplied again:

1. The trained-model ROI cube, usually a `32 x 32 x 32` `.npy` file.
2. The exact DICOM Series folder containing the CT used for that ROI.
3. The `SeriesInstanceUID` from that DICOM series.
4. The ROI centre voxel in `[x, y, z]` order for that exact series.
5. A new rib-obstacle JSON and its QC preview image.
6. A new planned trajectory JSON.
7. A new CoppeliaSim scene and simulation result, when simulation is run.

The raw CT does not need to be committed to Git or copied to every teammate's
computer. However, the computer running the pipeline must have access to the
exact DICOM series until obstacle extraction is complete. The generated
obstacle JSON can then be retained as the lightweight case artifact.

The measured patient-to-UR5 calibration is also case/setup-specific. The
current project has no measured matrix. The included examples therefore use
explicitly marked demo transforms that place each target in the same reachable
UR5 demonstration frame. These transforms preserve the relative CT geometry
for software integration only; they must not be described as clinical robot
registration. Replace them with a measured 4x4 matrix before making any claim
about physical robot placement.

## 2. What stays fixed?

These parts are shared by every case:

- The checkpoint architecture and checkpoint file.
- ROI validation and model inference code.
- HU thresholding, local rib AABB extraction, and preview generation code.
- Target-ROI exclusion during obstacle extraction, so a calcified nodule is not
  mistaken for a rib obstacle.
- Multi-direction straight-line trajectory planner.
- UR5 DH/IK validation.
- CoppeliaSim UR5 model, needle, simIK dummies, and collision-checking code.
- The order of execution in `src/run_case_pipeline.py`.

Do not copy 0037 coordinates, obstacles, or plan files to another case. The
script derives the target physical coordinate from the selected CT and target
voxel, then refuses to continue if the SeriesInstanceUID or configured target
metadata do not agree.

## 3. Prepare a case configuration

The provided configuration contains the three currently available examples:

```text
configs/cases_0036_0037_0039.json
```

Each case entry must provide:

```json
{
  "case_id": "LIDC-IDRI-0036",
  "roi_path": "/absolute/path/to/roi.npy",
  "ct_series_path": "/absolute/path/to/exact/series/folder",
  "series_instance_uid": "1.2.840...",
  "target_voxel": [142, 312, 199]
}
```

`ct_series_path` must point to one DICOM Series folder, not a patient folder
containing multiple series. `target_voxel` is `[x, y, z]`, while NumPy image
arrays are displayed in `[z, y, x]` order internally. The optional
`patient_to_robot_base` field is a 4x4 demo transform in millimetres; a real
setup should use `patient_to_robot_transform_path` for a measured matrix.

For a new case, add one object to the `cases` list. Update absolute paths on
the computer where the pipeline is run. The checkpoint, output directory,
planning parameters, and demo-frame flag can remain in `defaults`.

## 4. Offline per-case run

Run from the repository root:

```bash
cd /Users/vivi/Documents/BN5211/BN5211-Robotic-Lung-Biopsy-en-main
python src/run_case_pipeline.py \
  --config configs/cases_0036_0037_0039.json \
  --case-id LIDC-IDRI-0036 \
  --confirm-coincident-demo-frame
```

Replace only `--case-id` for 0037 or 0039. The offline run performs:

1. ROI shape and numerical-value validation.
2. DICOM loading and SeriesInstanceUID validation.
3. Target voxel to CT physical-coordinate conversion.
4. Local HU-based rib AABB extraction.
5. QC preview image generation.
6. Checkpoint inference.
7. Multi-direction collision-free trajectory search.
8. Analytical UR5 IK and needle-tip error validation.
9. A machine-readable case summary.

Expected files for case `0036` are:

```text
data/demo_nodules/rib_obstacles_0036_segmented.json
data/demo_nodules/rib_obstacles_0036_segmented_preview.png
data/demo_nodules/planned_trajectory_0036.json
data/demo_nodules/pipeline_summary_0036.json
```

Use `--skip-existing` only after manually checking the existing obstacle
preview. The script still validates the existing obstacle JSON's
SeriesInstanceUID.

## 5. Run the CoppeliaSim simulation

Start CoppeliaSim with the ZeroMQ server enabled:

```bash
/Users/vivi/Downloads/coppeliaSim.app/Contents/MacOS/coppeliaSim \
  -GzmqRemoteApi.rpcPort=23000
```

In another terminal, run the same case with `--run-simulation` and the bundled
Python Remote API client on `PYTHONPATH`:

```bash
cd /Users/vivi/Documents/BN5211/BN5211-Robotic-Lung-Biopsy-en-main
PYTHONPATH="/Users/vivi/Downloads/coppeliaSim.app/Contents/Resources/programming/zmqRemoteApi/clients/python/src:$PWD/src" \
python src/run_case_pipeline.py \
  --config configs/cases_0036_0037_0039.json \
  --case-id LIDC-IDRI-0036 \
  --confirm-coincident-demo-frame \
  --run-simulation \
  --frame-delay 0.08
```

The script generates the case scene, loads it into CoppeliaSim, and executes
the approach, entry, and insertion phases with `simIK`. The GUI should remain
open while the command runs. A nonzero frame delay makes the motion visible;
it does not change the planned path.

The generated files additionally include:

```text
scenes/robotic_lung_biopsy_0036_segmented.ttt
data/demo_nodules/coppeliasim_run_0036_segmented.json
```

Repeat the command with `LIDC-IDRI-0037` and `LIDC-IDRI-0039` to reproduce the
other cases.

## 6. Acceptance checks

### Before simulation

Open the `*_segmented_preview.png` image and confirm:

- The cyan target marker is on the intended lesion/ROI centre.
- Lime boxes correspond to nearby high-density bone candidates.
- The preview belongs to the requested case and exact SeriesInstanceUID.
- The boxes are a conservative simulation approximation, not a medical-grade
  rib segmentation.
- The target ROI is excluded from the bone candidate mask so a calcified
  lesion is not turned into its own obstacle. This does not replace clinical
  anatomy segmentation.

Open the plan JSON and confirm:

- `decision` is `robot_plan_ready`.
- `target_roi_match_confirmed` is `true`.
- `analytical_ik.converged` is `true`.
- `analytical_ik.needle_tip_error_mm` is below the project threshold, currently
  `0.5 mm`.
- `patient_to_robot_transform_mode` in the summary is explicitly understood:
  `coincident_demo_identity` is not a real calibration.

### After simulation

Open `coppeliasim_run_<case>_segmented.json` and confirm:

- `collision_free` is `true`.
- `collision_frames` is an empty list.
- `phase_sequence` contains approach, approach-to-entry, and insertion.
- `max_tip_position_error_mm` and `final_tip_position_error_mm` are small and
  consistent with the selected IK tolerance.
- The CoppeliaSim GUI shows the needle attached to the UR5 wrist and the tip
  dummy at the physical needle tip.

## 7. Which files are reusable?

Reusable across cases:

- `src/run_case_pipeline.py`
- `src/environment_modeling.py`
- `src/trajectory_planner.py`
- `src/main_pipeline.py`
- `src/create_coppeliasim_scene.py`
- `src/coppeliasim_remote.py`
- the checkpoint
- the CoppeliaSim installation and UR5 model

Not reusable across different cases:

- ROI `.npy`
- DICOM Series folder and UID
- target voxel and derived target coordinate
- obstacle JSON and preview
- trajectory JSON
- generated `.ttt` scene and simulation result

## 8. Common failures

**SeriesInstanceUID mismatch**: the DICOM folder is not the same series used
to produce the ROI. Find the correct nested series folder and update the case
configuration.

**Target mismatch**: the supplied target voxel is wrong for the selected
series. Do not bypass this check; obtain the ROI centre in the original CT
index space.

**No safe path**: inspect the preview and adjust the local extraction window
or planner sampling only after checking that the obstacle model is reasonable.

**Remote API unavailable**: start CoppeliaSim with port `23000`, keep it open,
and run the client with the bundled ZeroMQ client directory on `PYTHONPATH`.

**simIK failure**: inspect the scene object names and start configuration. The
stock UR5 demo uses `/UR5`, `/UR5/ikTip`, `/UR5/ikTarget`,
`/robotCollision`, and `/patientCollision`.

**Collision detected**: inspect the case preview and the generated scene. A
collision result is a failed run even if the tip reaches the target.

## 9. Short answer: what must be rerun?

For every new case, rerun the case-specific data preparation, inference,
trajectory planning, scene generation, and simulation. You do not retrain the
model, rebuild the UR5 logic, or manually edit the robot code. The intended
command boundary is the single `run_case_pipeline.py` entry point.
