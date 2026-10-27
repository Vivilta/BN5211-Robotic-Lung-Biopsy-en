# BN5211 Lung Nodule Image Analysis and Robotic Biopsy Simulation

This project covers lung nodule CT data processing, a binary malignancy-risk model, model evaluation and visualization, interactive inference, and a subsequent UR5 biopsy path-planning and CoppeliaSim simulation extension. The overall workflow is:

```text
LIDC-IDRI / LUNA16 data
  -> XML/CSV annotation curation
  -> CT preprocessing and nodule ROI construction
  -> Patient dataset manifest and train/validation split
  -> 2D/3D model training and comparison
  -> Evaluation on separate data, subgroup/ensemble/threshold analysis, Grad-CAM
  -> Streamlit ROI inference demo
  -> [Robot extension] CT geometry -> collision-aware planning -> UR5 IK -> CoppeliaSim simulation
```

The model outputs and robot workflow are course research prototypes, not clinical diagnosis or treatment systems. They must not be used for real patient care or to control a real surgical robot. The Streamlit page currently contains fixed risk ranges and recommendation text; these are not clinically validated results or medical advice.

## 1. Project modules

### 1.1 Data and model pipeline

- **LIDC-IDRI**: DICOM CT and radiologist annotations in XML; curates reader scores and extracts local 3D nodule ROIs.
- **LUNA16**: MHD CT and `annotations.csv`; used for whole-lung preprocessing, ROI extraction, and cross-dataset evaluation.
- **Models**: 2D ResNet/DenseNet/CBAM, 3D ResNet/CBAM/VNet, and an ablation experiment without augmentation.
- **Analysis**: Classification metrics, LUNA16 evaluation, clinical subgroup analysis, model ensembling, ROC/Youden operating points, and Grad-CAM visualizations.
- **Interactive demo**: The Streamlit page accepts pre-normalized `.npy` ROIs and displays probabilities and risk strata.

### 1.2 Robot extension

The robot pipeline reuses a trained 3D CBAM checkpoint. It uses the ROI, CT, and target for a registered case to generate local bone-candidate AABBs, plans a straight path, computes analytical UR5 IK, and runs a kinematic simulation using the CoppeliaSim UR5, simIK, and ZeroMQ Remote API. This is a later robot-demo extension to the main project; it does not replace the preceding data and model pipeline. The current case configuration enables `force_robot_demo`, so software integration planning continues even when the malignancy probability is below its threshold. The model score does not automatically select a nodule and is not a safety gate for robot planning.

## 2. Environment and local data

Python 3.9 is recommended. Install the dependencies:

```bash
conda create -n bn5211 python=3.9 -y
conda activate bn5211
python -m pip install -r requirements.txt
```

The default data roots in [config/config.yaml](config/config.yaml) are:

```yaml
data:
  raw_lidc_dir: "./data/LIDC-IDRI"
  raw_luna_dir: "./data/LUNA16"
  processed_dir: "./data/processed"
```

Change the paths to your local data locations, or place the data in the directory structure shown above. Original medical images, training checkpoints, and complete processed datasets are generally not included in Git; each user must prepare these local assets.

Basic dependency check:

```bash
python -c "import torch, SimpleITK, pydicom, numpy, pandas; print(torch.__version__)"
```

## 3. Directory structure and main outputs

```text
config/config.yaml                 # Roots for raw and processed data
configs/cases.json                 # General case registry (currently includes example cases)
src/preprocess.py                  # LUNA16 MHD whole-lung resampling/masking/normalization
src/explore_data.py                # LIDC XML annotation curation and label manifest
src/extract_roi.py                 # LIDC 3D ROI extraction
src/extract_luna16.py              # LUNA16 nodule ROIs and manifest
src/split_dataset.py               # Patient-level split manifest generation
src/train_*.py                     # 2D/3D model training
src/evaluate_*.py                  # Evaluation, interpretability, and application analyses
src/run_case_pipeline.py           # Single-case robot workflow orchestration
data/processed/                    # Locally generated ROIs, CSV manifests, and preprocessed images
models/                             # Local checkpoints, logs, and analysis plots
data/demo_nodules/                  # Robot workflow JSON, previews, and run records
scenes/                             # Canonical CoppeliaSim scene files
docs/                               # Path-planning references
scripts/                            # Analysis and diagnostic helper scripts
```

Original images, preprocessing matrices, checkpoints, and run outputs under `data/` are not distributed with the repository. The repository retains source code, tests, planning-reference documentation, and a small number of canonical scenes. CT/DICOM data, ROIs, checkpoints, case-registration configuration under `data/`, run outputs, and locally generated experimental scenes are ignored. Generate each new case's scene and run outputs locally. `configs/cases.json` is a configuration template without machine-specific absolute paths; fill in local data paths before running. Real case-registration configurations belong under local `data/` and must not be committed to a public repository.

## 4. End-to-end run guide (data to model)

Run all commands from the repository root. First check the paths in `config/config.yaml` and prepare the required LIDC-IDRI and LUNA16 input assets.

### Recommended order starting from raw data

To prepare the project's LIDC-IDRI and LUNA16 outputs from scratch, run the following steps in order. This sequence puts preparation for both datasets in one main workflow:

```bash
python src/preprocess.py
python src/explore_data.py
python src/extract_roi.py
python src/extract_luna16.py
python src/split_dataset.py
```

Prepare the following before running:

- LIDC-IDRI raw DICOM and XML files in the directory specified by `raw_lidc_dir` in `config/config.yaml`.
- LUNA16 raw `.mhd` files and their associated image files, plus `annotations.csv`. `preprocess.py` reads `raw_luna_dir` and `processed_dir` from the configuration, but `extract_luna16.py` uses the relative input/output paths `data/LUNA16/` and `data/processed/`. Therefore, run it from the repository root and put LUNA16 data in the corresponding directory, or update the script paths for your machine first.
- Sufficient disk space. `preprocess.py` writes a preprocessed `.npy` for each full LUNA16 volume; it does not generate only small nodule ROIs.

The inputs, outputs, and dataset notes for these five steps are described below. If processing only one dataset, you can skip steps specific to the other dataset. However, `split_dataset.py` depends on LIDC's `final_dataset_manifest.csv`, so it cannot be run to generate this project's training split when only LUNA16 has been prepared.

### Step 1: LUNA16 whole-lung preprocessing

Make sure `raw_luna_dir` in `config/config.yaml` points to the LUNA16 directory containing `.mhd` files and associated images, then run:

```bash
python src/preprocess.py
```

For each whole-lung volume, the script performs 1 mm isotropic resampling, threshold/morphological mask processing, and HU normalization. It writes:

```text
data/processed/preprocessed_cubes/*_preprocessed.npy
```

Processing the full dataset takes considerable time and disk space. The script can skip cases whose output files already exist.

### Step 2: Curate LIDC annotations

```bash
python src/explore_data.py
```

Expected outputs:

```text
data/processed/master_data_exploration.csv
data/processed/cleaned_nodules_labels.csv
```

The script reads LIDC XML, aggregates nodule spatial coordinates and scores, then filters labels by reader count, diameter, and mean malignancy score. It is a project data-curation script; its output should not be described as clinical ground truth. The current code uses simplified coordinate clustering and diameter approximations, so sample-check the results before using them.

### Step 3: Generate LIDC nodule ROIs and dataset manifest

```bash
python src/extract_roi.py
```

Expected outputs:

```text
data/processed/all_roi_cubes/*.npy
data/processed/final_dataset_manifest.csv
```

Each `.npy` is a single-channel `32 x 32 x 32` CT patch generated using the script's windowing, cropping, and normalization conventions. Check the correspondence among ROIs, labels, and source DICOM data before and after running. **The current LIDC ROI script traverses each patient directory and uses the first readable DICOM series it finds.** If a patient has multiple series, do not assume that this is the series corresponding to the XML annotation. For data requiring strict case traceability, confirm the association between each series and ROI first.

### Step 4: Generate LUNA16 nodule ROIs

Confirm that `data/LUNA16` contains the raw `.mhd` data, associated image files, and the official `annotations.csv`, and that Step 1 is complete. Then run:

```bash
python src/extract_luna16.py
```

The script generates:

```text
data/processed/luna16_roi_cubes/*.npy
data/processed/luna16_dataset_manifest.csv
```

The extraction script crops ROIs using physical coordinates from `annotations.csv`. LUNA16 annotation and label definitions differ from LIDC's; do not interpret labels from the two datasets as the same clinical outcome without explanation.

### Step 5: Generate the data-split manifest

```bash
python src/split_dataset.py
```

This generates `data/processed/final_split_train_manifest.csv`. The script creates split manifests stratified by patient and risk. However, most current `train_*.py` and `evaluate_*.py` scripts split data themselves with `random_split` and do not read this split manifest. Therefore, **running `split_dataset.py` alone does not establish that all training and evaluation use patient-level independent splits**. Before formally reporting model performance, verify and standardize the train, validation, and test indices to avoid patient/nodule leakage.

### Stage 5: Train models (optional)

Skip this stage if you only need to reuse the provided CBAM weights for inference or the robot demo. To retrain, first generate the ROIs and manifests described above, then run the model scripts for your experiment.

```bash
# 2D baselines and DenseNet
python src/train_2D.py
python src/train_2D_densenet.py
python src/train_2D_attention.py

# 3D baselines, CBAM, VNet, and ablation
python src/train_3D.py
python src/train_3D_attention.py
python src/train_3D_vnet.py
python src/train_3D_ablation.py
```

The training scripts use PyTorch Lightning, save the best checkpoint according to `val_acc` in their respective `models/checkpoints*` subdirectories, and write TensorBoard logs to `models/logs/`. Hyperparameters, data splits, and output directories are defined by the code; check the data split and random-seed settings before running. The robot currently uses the 3D CBAM architecture in `src/train_3D_attention.py`; a standard 3D ResNet or 2D checkpoint cannot replace it.

### Stage 6: Model evaluation and analysis (optional)

```bash
# Compare 2D / 3D models
python src/evaluate_2D_models.py
python src/evaluate_3D_models.py

# Cross-dataset LUNA16 evaluation (requires a prepared LUNA16 manifest, ROIs, and matching checkpoints)
python src/evaluate_luna16_generalization.py

# Subgroup, ensemble, and threshold analyses
python src/evaluate_clinical_subgroups.py
python src/evaluate_ensemble.py
python src/evaluate_clinical_value.py

# 3D CBAM Grad-CAM
python src/generate_3D_gradcam.py
```

Evaluation plots are mainly written to `models/evaluation_results/`. These scripts are experimental analysis tools for a course project. Their outputs depend on the data split, label definitions, sample size, and implementation, and do not constitute external clinical validation or clinical thresholds.

### Stage 7: Streamlit inference demo (optional)

`src/app.py` accepts `.npy` ROIs already generated according to this project's preprocessing conventions, not raw DICOM data. By default, the page loads 3D CBAM weights from `models/checkpoints_attention/`. Install the dependencies in `requirements.txt`, place a checkpoint with a compatible architecture in that directory, and run:

```bash
streamlit run src/app.py
```

After uploading an ROI, you can view the model probability and the demo risk stratum defined by the page. The risk boundaries and action recommendations are fixed display logic in the page code and have not been clinically validated. The app does not automatically detect, locate, or crop nodules from CT and must not be used to make medical decisions.

## 5. Robotic biopsy simulation extension

The robot component takes a confirmed ROI/target/CT combination as input. LIDC XML contains multiple readers and nodules, so the script does not infer the simulation target from the patient ID alone. First select a specific XML annotation and its reader/nodule IDs.

### 5.1 Case inputs and registration

Each case is registered in the `cases` array of the general file `configs/cases.json`. The filename refers to the case registry; it does not limit the number or IDs of cases:

```json
{
  "case_id": "LIDC-IDRI-00XX",
  "roi_path": "/absolute/path/to/case_roi.npy",
  "ct_series_path": "/absolute/path/to/one/exact/DICOM/series",
  "series_instance_uid": "UID_READ_FROM_THAT_SERIES",
  "target_voxel": [X, Y, Z]
}
```

Field sources and current automation capability:

| Field | How to obtain it | Can it currently be generated reliably? |
| --- | --- | --- |
| `case_id` | LIDC case directory name, such as `LIDC-IDRI-0035` | Yes; read from the input data directory name |
| `roi_path` | Find the corresponding `.npy` from `cube_file_path` in `data/processed/final_dataset_manifest.csv`, or provide a prepared checkpoint input ROI | Candidate ROIs can be listed; verify that the selected ROI is the nodule for this simulation |
| `ct_series_path` | Path to the one confirmed DICOM Series folder for this case | Candidate series can be listed offline; the correct series cannot be selected from patient ID alone |
| `series_instance_uid` | Read `SeriesInstanceUID` from the selected series' DICOM headers | Can be read automatically after selecting a series and validated by the pipeline |
| `target_voxel` | `[x, y, z]` voxel indices of the ROI center in the selected CT series; the array's internal order is `[z, y, x]` | Cannot currently be recovered reliably from existing manifests; verify the relationship between ROI and CT |

An added tool can generate a `32³` ROI and registration metadata from a selected LIDC XML annotation in an offline environment. First list the nodules, readers, and scores in the XML:

```bash
python src/register_lidc_case.py \
  --annotation-xml /path/to/one/LIDC-annotation.xml \
  --list-annotations
```

Then explicitly specify the exact DICOM series for the same case, the `noduleID`, and, if that ID occurs more than once in the XML, its corresponding `servicingRadiologistID`:

```bash
python src/register_lidc_case.py \
  --case-id LIDC-IDRI-00XX \
  --ct-series /path/to/exact/DICOM-series \
  --annotation-xml /path/to/one/LIDC-annotation.xml \
  --nodule-id "Nodule 001" \
  --reader-id "READER_ID" \
  --output-dir data/registered_rois \
  --cases-config configs/cases.json
```

The script cross-checks the series and slices using the XML `SeriesInstanceUid`, each contour's `imageSOP_UID`, and the DICOM headers. It then maps the contour center to `[x,y,z]` voxels using DICOM orientation, pixel spacing, and slice positions. Outputs include the checkpoint-compatible `32 x 32 x 32` HU-window-normalized ROI, a registration JSON with source and coordinate information, three-view QC PNGs, and optionally a case entry appended to `cases.json`. Existing outputs are not silently overwritten. Use `--overwrite` to replace an ROI file or `--replace-case` to replace an existing case in the configuration.

The registration script automates “one explicitly selected reader annotation -> corresponding DICOM -> ROI/target/config record”; it does not automatically select a clinical-consensus nodule from a patient ID. Nodule IDs from different LIDC readers must not be treated as the same nodule based only on matching IDs. After generation, inspect the QC images to confirm the ROI covers the intended lesion. The raw DICOM/XML and checkpoint must already be downloaded locally; the script does not access online services. The older `extract_roi.py` remains intended for slicing the course training dataset. Its “first readable series in the patient directory” logic must not be used for strict case registration.

`patient_to_robot_base` is the 4x4 transform from CT patient coordinates to the UR5 base. The matrix in `configs/cases.json` is a simulation positioning parameter, not a measured calibration, and cannot be derived from CT/XML. A real robot application requires obtaining the matrix through an external calibration procedure and registering it as `patient_to_robot_transform_path`; this project is only for explicitly labeled demonstrations in a teaching coordinate frame. The checkpoint, output directory, and planning parameters are in shared `defaults`; case-specific paths, UID, target voxels, and demo transforms are in the corresponding case entry. Update absolute paths for each computer. Relative ROI paths are resolved from the repository root.

`src/run_case_pipeline.py` checks the ROI shape and values, the DICOM Series UID, and the conversion from target voxel to physical coordinates. These consistency checks do not prove that the operator selected the correct nodule or ROI. Add a verified case record for each new case; obstacles, plans, and simulation results are written separately by case ID. Do not copy a target or result from another case.

### 5.2 Generate obstacles and a trajectory offline

```bash
python src/run_case_pipeline.py \
  --config configs/cases.json \
  --case-id LIDC-IDRI-0031 \
  --output-dir data/case_runs/LIDC-IDRI-0031
```

This step reads the CT, extracts local high-density bone-candidate AABBs, generates a QC preview, runs inference with the CBAM checkpoint, searches for a straight path, computes analytical UR5 IK, and generates:

```text
data/case_runs/LIDC-IDRI-0031/rib_obstacles_0031_segmented.json
data/case_runs/LIDC-IDRI-0031/rib_obstacles_0031_segmented_preview.png
data/case_runs/LIDC-IDRI-0031/planned_trajectory_0031.json
data/case_runs/LIDC-IDRI-0031/pipeline_summary_0031.json
```

0031 is an example case with a complete kinematic simulation pass record in the repository configuration. Under the current 30° entry-angle range, 0036 has no safe straight path after the full needle-sweep check. 0039 had an offline planning result, but that does not establish a simulation pass with the current version. Current patient obstacles are segmented AABB approximations generated using a local HU threshold; they are not a complete thorax/rib segmentation and do not include the actual skin surface, vessels, soft tissue, or respiratory motion.

### 5.3 Generate a scene and run the CoppeliaSim simulation

Start the CoppeliaSim ZeroMQ Remote API service on port `23000`. macOS example:

```bash
export COPPELIASIM_ROOT="/Applications/coppeliaSim.app"
"$COPPELIASIM_ROOT/Contents/MacOS/coppeliaSim" \
  -GzmqRemoteApi.rpcPort=23000
```

Keep CoppeliaSim running, then run the following command from another terminal in the repository:

```bash
PYTHONPATH="$COPPELIASIM_ROOT/Contents/Resources/programming/zmqRemoteApi/clients/python/src:$PWD/src" \
python src/run_case_pipeline.py \
  --config configs/cases.json \
  --case-id LIDC-IDRI-0031 \
  --run-simulation \
  --frame-delay 0.08 \
  --output-dir data/case_runs/LIDC-IDRI-0031 \
  --scene-output scenes/robotic_lung_biopsy_0031_latest.ttt
```



#### 5.3.1 Regenerate the 0037 scene from current inputs (manual retest)

To verify that the current code can generate a usable scene, rerun the pipeline from the CT, registered ROI, case configuration, and checkpoint, using a new output directory and a new `.ttt` file. Do not use previously generated run JSON or `.ttt` files as inputs for this retest. The command below uses LIDC-IDRI-0037 as an example, with a 500 mm mount, a 220 mm needle, and the current segmented-AABB obstacle extraction parameters. `data/registration_candidates/cases_supine_mount500.json` is a local registration config and is not included in the public repository; replace it with your own case configuration after cloning.

From the repository root, first run `cd /path/to/BN5211-Robotic-Lung-Biopsy-en`. Terminal 1: start the CoppeliaSim ZeroMQ Remote API service:

```bash
export PROJECT_ROOT="$(pwd)"
export COPPELIASIM_ROOT="/Applications/coppeliaSim.app"
"$COPPELIASIM_ROOT/Contents/MacOS/coppeliaSim" \
  -GzmqRemoteApi.rpcPort=23000
```

Terminal 2: re-extract obstacles from the inputs, plan a trajectory, generate a scene, and run a 100-step kinematic simulation:

```bash
export PROJECT_ROOT="$(pwd)"
export COPPELIASIM_ROOT="/Applications/coppeliaSim.app"
export PYTHONPATH="$COPPELIASIM_ROOT/Contents/Resources/programming/zmqRemoteApi/clients/python/src:$PROJECT_ROOT/src"

PYTHONUNBUFFERED=1 python src/run_case_pipeline.py \
  --config data/registration_candidates/cases_supine_mount500.json \
  --case-id LIDC-IDRI-0037 \
  --run-simulation \
  --coppeliasim-port 23000 \
  --frame-delay 0.08 \
  --needle-length-mm 220 \
  --mount-height-mm 500 \
  --segment-slices 5 \
  --min-volume-voxels 40 \
  --output-dir data/case_runs/LIDC-IDRI-0037_manual_fresh \
  --scene-output scenes/robotic_lung_biopsy_0037_manual_fresh.ttt
```

If you use Conda, activate the environment containing the project dependencies first. You can also replace `python` with that environment's interpreter. On success, a new `scenes/robotic_lung_biopsy_0037_manual_fresh.ttt` should be generated, along with `rib_obstacles_0037_segmented.json`, `planned_trajectory_0037.json`, `coppeliasim_run_0037_segmented.json`, and `pipeline_summary_0037.json` in the same output directory.

Terminal 3: after closing CoppeliaSim or while it remains running, inspect this run's JSON. The field is named `phase_sequence`; do not read a nonexistent `phases` field or use the old output path `data/case_runs/manual_test_0037/coppeliasim_run_0037.json`:

```bash
python - <<'PY2'
import json
from pathlib import Path

path = Path("data/case_runs/LIDC-IDRI-0037_manual_fresh/coppeliasim_run_0037_segmented.json")
d = json.loads(path.read_text())
required = {"approach", "approach_to_entry", "insertion"}
phases = set(d.get("phase_sequence", []))
assert d.get("executed_steps", 0) >= 100
assert required <= phases, phases
assert d.get("collision_check_available") is True
assert d.get("needle_self_collision_check_available") is True
assert d.get("mount_collision_check_available") is True
assert d.get("collision_free") is True
assert d.get("needle_self_collision_free") is True
assert d.get("mount_collision_free") is True
assert not d.get("collision_frames")
assert not d.get("needle_self_collision_frames")
assert not d.get("mount_collision_frames")
assert d.get("max_tip_position_error_mm") is not None
assert d["max_tip_position_error_mm"] < 1.0
assert d.get("max_joint_step_rad") is not None
assert d.get("total_joint_motion_rad") is not None
margin = d.get("min_joint_limit_margin_rad")
threshold = d.get("joint_limit_margin_threshold_rad")
if margin is not None and threshold is not None:
    assert margin + 1e-6 >= threshold, (margin, threshold)
print("PASS")
for key in (
    "executed_steps", "max_tip_position_error_mm", "max_joint_step_rad",
    "total_joint_motion_rad", "min_joint_limit_margin_rad",
    "joint_limit_margin_threshold_rad",
):
    print(f"{key}: {d.get(key)}")
PY2
```

When this retest passed, `executed_steps=100` and the maximum needle-tip error was approximately 0.387 mm. Patient, needle self-collision, and mount collision checks were all available, with no collision frames. This result only indicates a kinematic simulation pass in the current software scene with its approximate obstacle model.

The default teaching scene uses a `220 mm` needle and raises the overhead mount by `200 mm`, while compensating for the demo coordinate transform so the CT patient geometry does not move in world coordinates. This changes the robot's extension posture but not the needle path relative to the patient. Local obstacle extraction defaults to `segment_slices=5` and `min_volume_voxels=40`. The planner expands bone-candidate AABBs by the 4 mm needle radius plus a 2 mm extra margin and checks the needle sweep from pre-approach to target. During scene generation, simIK is also run at every frame along the full Cartesian `approach -> entry -> target` trajectory, with checks for UR5/needle collisions with patient obstacles and needle self-collision with the robot. A scene is saved only if the full preflight passes. These values are simulation assumptions, not clinical device specifications. Adjust the ZeroMQ client path for your local CoppeliaSim installation.

Needle length, overhead-mount height, and obstacle segmentation parameters can be overridden for a single validation run without changing the configuration file. For example:

```bash
python src/run_case_pipeline.py \
  --config configs/cases.json \
  --case-id LIDC-IDRI-0031 \
  --needle-length-mm 220 \
  --mount-height-mm 200 \
  --segment-slices 5 \
  --min-volume-voxels 40
```

`segment_slices` is the number of CT Z-axis slices (axial slice direction) grouped together when dividing a connected bone candidate and generating one AABB obstacle box per group. Its unit is CT slices, not millimeters. For example, a candidate spanning about 30 slices will usually produce about six longitudinal obstacle boxes when `segment_slices=5`. A smaller value creates thinner and more numerous boxes, preserving more local shape but potentially retaining more segmentation noise. A larger value creates fewer boxes, but with coarser bounds that may extend beyond the ribs. This parameter affects only the obstacle approximation; it does not change the ROI, target, or actual needle insertion depth.

The default obstacles are not generated by a trained segmentation checkpoint. `src/environment_modeling.py` reads the DICOM CT matching the case, extracts local high-density bone candidates using an HU threshold (500 by default), applies 3D connected components, voxel-count filtering, and slice-based segmentation, then converts each candidate into an axis-aligned bounding box (AABB). `3d_resnet_cbam-epoch=29-val_acc=0.84.ckpt` is used only to classify ROI malignancy probability; it is not used for rib segmentation, target localization, or obstacle generation. The current obstacles are simulation approximations and must not be described as a complete rib reconstruction.

The locally saved run records for 0031 and 0040 each completed 100/100 kinematic steps. Collision checks were available, no patient or needle self-collisions were reported, and the maximum needle-tip position error was approximately `0.455 mm`. These are kinematic simulation records for the specified software scene and current approximate obstacle model; they are not dynamic validation, real-robot validation, or proof of clinical safety. Under the current configuration, 0036 has no feasible path; 0039 has no complete pass record for the current version. `data/` is Git-ignored, so these run JSON files are not distributed with the repository. Other computers must regenerate the results with the commands and inspect them.

Record a software-scene simulation as passed only when collision checks are available, both `collision_free` and `needle_self_collision_free` are true, there are no collision frames, all motion phases are complete, and needle-tip error is within the project's tolerance. A collision or IK failure is a failed result. Do not force a pass by arbitrarily changing needle length, disabling collision checks, or relaxing unsupported constraints. The current needle-length parameter is a geometric placeholder and has not been selected according to a specific clinical device IFU.

### 5.4 Manual acceptance checklist

Repeat acceptance checks after changing the case, ROI, DICOM series, coordinate transform, or simulation parameters. Do not decide that a run passed based only on the planner's “number of safe trajectories” or the process exit code.

1. **Inputs and coordinates**: Verify the case ID, ROI, DICOM `SeriesInstanceUID`, three-view ROI QC image, and target position. Confirm that the target voxel in `cases.json` matches the selected series. The patient-to-UR5 transform in the case registration is currently a teaching setup, not a measured calibration.
2. **Obstacle preview**: Open `rib_obstacles_<case>_segmented_preview.png`. Confirm that the bone candidates near the target roughly correspond to high-density CT structures and check for obvious isolated segmentation noise. The local AABB model is simplified; the preview does not prove complete or clinically accurate rib segmentation.
3. **Trajectory and scene**: Check that the approach, entry, target, direction, and insertion depth in `planned_trajectory_<case>.json` match the intended target. Open the generated `.ttt` and verify the overhead UR5, mounting plate, needle direction, target point, and obstacle positions. Check for initial self-collision or the needle passing through the arm. During execution, watch the full approach, entry, and insertion sequence in CoppeliaSim for continuity and unintended large movements. The current script does not automatically generate a GIF or video.
4. **Simulation result JSON**: Check that `executed_steps` in `coppeliasim_run_<case>_segmented.json` equals the configured step count; `collision_check_available` and `needle_self_collision_check_available` are true; `collision_free` and `needle_self_collision_free` are true; and `collision_frames` and `needle_self_collision_frames` are empty. Also inspect the maximum and final needle-tip errors.
5. **Summary results**: Check that the inputs, obstacles, plan, scene, and simulation result referenced by `pipeline_summary_<case>.json` all belong to the same case and current output directory. Any collision, IK failure, series mismatch, or unclear input provenance is a failure. Determine the cause before rerunning; do not describe a failed result as a successful biopsy.

Manual acceptance is only for checking software-simulation consistency and interpretability. The current setup lacks a complete patient thorax, soft tissue, organ motion, dynamics, and a real-robot closed loop. Therefore, a displayed needle tip reaching the planned target does not mean that a clinical needle biopsy has been performed.


### 5.5 Export video with a synchronized target (2026-10-24)

The new `scripts/record_verified_replay.py` implements the recording method used locally for 0031, 0037, and 0040. Each original sample updates both `/UR5/ikTarget` and the six verified joint angles, and the actual needle-tip-to-target distance is checked frame by frame. It replays an existing run record that passed validation; it does not replan or turn a failed run into a passing one.

Install the optional video dependencies and start CoppeliaSim first:

```bash
python -m pip install -r requirements-video.txt
python scripts/record_verified_replay.py --case-id LIDC-IDRI-0037 --run-json data/case_runs/LIDC-IDRI-0037_manual_fresh/coppeliasim_run_0037_segmented.json --scene scenes/robotic_lung_biopsy_0037_manual_fresh.ttt --output data/videos/LIDC-IDRI-0037_synchronized_1080p.mp4 --port 23000
```

Replace the example paths with the JSON and `.ttt` generated by the same successful local run. The script opens that scene; save any unsaved edits in the simulator first. Existing output files are not overwritten. Output is a silent 1920×1080, 30 fps, H.264 MP4. The current version requires 100 original samples and retains 5 frames per sample, for a video of about 16.7 seconds. The accompanying `.replay.json` records the maximum error per frame and the full video-decode check results. The vision sensor does not render dummies, so small spheres show the actual needle-tip position and current navigation-target position. The navigation target moves along the trajectory and is not an additional nodule. This script uses the validated model and joint order and cannot be used with arbitrary robot scenes that have a different structure.

The newly registered local case 0037 on 2026-10-24 (XML `138.xml`, nodule `IL057_127447`, reader `-822877843`) uses target voxel `[407, 288, 82]`, which differs from the old template target `[407, 289, 79]`. The supine teaching setup, 500 mm mount height, and 220 mm needle, together with the teaching coordinate transform, place the target at robot-base coordinates `[300, 0, 600]` mm. This run completed 100 steps; all three collision checks were available, with no collision frames, and the maximum needle-tip error was approximately 0.150 mm. The actual ROI, coordinate transform, initial joint pose, run record, and scene are kept locally under `data/` or the run-output directory. For a rerun, use a matching set of inputs; do not use the template directly as the run configuration.

The UR5 model path on Windows differs from the repository's default macOS search path. You can override the model path as in the command used for this run without changing the core planner:

```powershell
python scripts/run_local_case.py --ur5-model "C:/Program Files/CoppeliaRobotics/CoppeliaSimEdu/models/robots/non-mobile/UR5.ttm" --config data/registration_candidates/cases_supine_mount500.json --case-id LIDC-IDRI-0037 --run-simulation --coppeliasim-port 23000 --output-dir data/case_runs/LIDC-IDRI-0037_manual_fresh --scene-output scenes/robotic_lung_biopsy_0037_manual_fresh.ttt
```

Each computer must still prepare these configuration files based on its case registration. The public repository does not upload CT data, ROIs, model weights, videos, personal absolute paths, or complete local run configurations.


## 6. Current limitations and reproducibility notes

- **Data paths**: `config/config.yaml`, robot configuration, and several older scripts contain local relative or absolute paths; check and adjust them when moving to another computer.
- **Case-to-ROI association**: `src/register_lidc_case.py` packages an explicitly selected LIDC XML annotation, exact DICOM series, ROI, and provenance into a rerunnable workflow. It does not select a clinical-consensus nodule on behalf of the reader; QC images and annotation selection still require manual confirmation.
- **Model validation**: Training and evaluation scripts are available, but most training/evaluation entry points do not consistently use the split manifest. Interpret metrics with caution; standardize the patient-level split before rigorous comparisons.
- **LUNA16 labels**: LUNA16 annotations provide nodule centers and diameters, not LIDC radiologist malignancy scores. Label construction in the current LUNA16 extraction script is a project implementation choice and must not be treated as pathology ground truth.
- **Robot geometry**: Local AABBs support only software-level collision approximations. Complete ribs, body surface, soft tissue, motion, and real device geometry have not yet been modeled.
- **Coordinate calibration**: The CT-to-UR5 transform in the example case is a teaching setup; measured calibration from a real patient to the robot is not available.
- **Needle parameters**: Geometric length in the simulation does not represent clinical device length. Select a device and model it according to its IFU first.
- **Clinical scope**: All model probabilities, risk strata, and robot trajectories are research demonstrations. They do not provide clinical decisions or an executable surgical plan.

## 7. Frequently asked questions

**Input file/manifest not found**: Check `config/config.yaml` and the data directories. Generate the required CSV, ROIs, and preprocessed files in the stages described above.

**Model checkpoint architecture mismatch**: Confirm that the 2D, 3D ResNet, 3D CBAM, or VNet checkpoint matches the model class used to load it. The robot pipeline uses a 3D CBAM checkpoint.

**Robot case ID is not registered**: Add a verified ROI, exact DICOM series, Series UID, and target voxel to the `cases` array in the robot JSON.

**SeriesInstanceUID mismatch or multiple series found**: Specify the one correct DICOM series directory and verify the configuration. Do not bypass the validation.

**No collision-free path / IK failure**: Check the target, transform, preview, and UR5 configuration. Do not change clinical device parameters or disable collision checks just to get a successful log.

**Remote API unavailable**: Confirm that CoppeliaSim is running on port `23000` and that `PYTHONPATH` uses the Python ZeroMQ client bundled with that installation.

**Handling old scenes and results**: Historical scenes such as `scenes/robotic_lung_biopsy_0037_fresh.ttt` and `scenes/robotic_lung_biopsy_0037_supine.ttt` do not include the mounting-plate collision collection used by the current executor and should not be used for a new-version retest. `0037_needle15` belongs to a historical needle-length experiment. Keep `robotic_lung_biopsy_0037_manual_fresh.ttt`, `data/case_runs/LIDC-IDRI-0037_manual_fresh/`, and `data/registration_candidates/cases_supine_mount500.json` as reproducibility records for now. Once historical comparisons are no longer needed, archive old `.ttt` files and their corresponding run directories; do not delete them without checking their provenance.


## 8. Validation

Run the unit tests in the repository:

```bash
python -m unittest discover -s tests -v
```

An end-to-end robot run also requires local LIDC DICOM data, an ROI, a checkpoint, and a running CoppeliaSim ZeroMQ service. Passing the unit tests does not mean that a case simulation has passed.

## 9. Main script index

| Function | Entry point |
| --- | --- |
| LUNA16 whole-lung preprocessing | `src/preprocess.py` |
| LIDC annotation curation | `src/explore_data.py` |
| LIDC ROIs and training manifest | `src/extract_roi.py` |
| Single-case XML-DICOM ROI registration | `src/register_lidc_case.py` |
| LUNA16 ROIs and manifest | `src/extract_luna16.py` |
| Patient-level split manifest | `src/split_dataset.py` |
| 2D/3D training | `src/train_*.py` |
| Model evaluation and interpretability | `src/evaluate_*.py`, `src/generate_3D_gradcam.py` |
| Streamlit inference | `src/app.py` |
| Robot case orchestration | `src/run_case_pipeline.py` |
| CoppeliaSim scene generation/execution | `src/create_coppeliasim_scene.py`, `src/coppeliasim_remote.py` |
