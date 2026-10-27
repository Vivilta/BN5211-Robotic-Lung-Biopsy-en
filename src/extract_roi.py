import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import SimpleITK as sitk
import os

# ==========================================
# 🛠️ Configuration loading
# ==========================================
def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load the global YAML path configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ==========================================
# 🔍 Core utility: load 3D DICOM images from nested LIDC directories.
# ==========================================
def load_lidc_dicom_series(patient_dir: str):
    """
    Recursively scan nested directories and load the first directory
    containing DICOM files as a 3D image volume.
    """
    for root, dirs, files in os.walk(patient_dir):
        if any(f.endswith('.dcm') for f in files):
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(root)
            if dicom_names:
                reader.SetFileNames(dicom_names)
                try:
                    return reader.Execute()
                except:
                    continue
    return None

# ==========================================
# ✂️ Core utility: crop a local 3D image patch (ROI).
# ==========================================
def extract_3d_cube(itk_image, pos_x, pos_y, pos_z, cube_size=32):
    """
    Crop a 32x32x32 local nodule cube from the volume using fused physical
    coordinates, then normalize it with a standard medical HU window.
    """
    if cube_size <= 0 or cube_size % 2 != 0:
        raise ValueError("cube_size must be a positive even number")

    img_array = sitk.GetArrayFromImage(itk_image)  # Array order: (Z, Y, X)

    # SimpleITK is authoritative here: it applies origin, spacing, and direction.
    # Treating physical X/Y values as voxel indices silently extracts the wrong lesion
    # whenever the scan origin or spacing is not zero/one.
    physical_point = (float(pos_x), float(pos_y), float(pos_z))
    x_index, y_index, z_index = itk_image.TransformPhysicalPointToIndex(physical_point)

    half_size = cube_size // 2
    z_start, z_end = z_index - half_size, z_index + half_size
    y_start, y_end = y_index - half_size, y_index + half_size
    x_start, x_end = x_index - half_size, x_index + half_size

    max_z, max_y, max_x = img_array.shape
    if z_start < 0 or y_start < 0 or x_start < 0 or z_end > max_z or y_end > max_y or x_end > max_x:
        return None

    cube = img_array[z_start:z_end, y_start:y_end, x_start:x_end]

    # Lung-window normalization used by the supplied 32^3 checkpoints.
    cube = np.clip(cube, -1000.0, 400.0)
    cube = (cube + 1000.0) / 1400.0
    return cube.astype(np.float32)

# ==========================================
# 🚀 Attach binary and multiclass labels and crop 3D ROIs.
# ==========================================
if __name__ == "__main__":
    cfg = load_config()
    lidc_root = Path(cfg['data']['raw_lidc_dir'])
    processed_dir = Path(cfg['data']['processed_dir'])
    
    # Load the cleaned reference nodule list from the upstream step.
    labels_path = processed_dir / "cleaned_nodules_labels.csv"
    if not labels_path.exists():
        print("Error: Cleaned nodule labels were not found. Run explore_data.py first.")
        exit()
    df_labels = pd.read_csv(labels_path)
    
    # Create a directory for local 3D arrays (.npy files).
    cube_store_dir = processed_dir / "all_roi_cubes"
    cube_store_dir.mkdir(parents=True, exist_ok=True)
    
    # Scan all raw patient case directories.
    all_lidc_patients = [f for f in lidc_root.glob("LIDC-IDRI-*") if f.is_dir()]
    
    print("==================================================")
    print("🔪 Cropping 3D ROIs and attaching binary and multiclass labels...")
    print(f"-> Found {len(all_lidc_patients)} raw LIDC patient cases.")
    print("==================================================")
    
    final_dataset_records = []
    roi_success_count = 0
    
    for patient_folder in all_lidc_patients:
        patient_id = patient_folder.name
        
        # Check whether the patient has eligible nodules in the reference list.
        associated_nodules = df_labels[df_labels['patient_id'] == patient_id]
        
        # Case A: Load and crop the scan when high-quality nodules are available.
        if not associated_nodules.empty:
            itk_image = load_lidc_dicom_series(str(patient_folder))
            
            for _, n_row in associated_nodules.iterrows():
                nodule_id = n_row['nodule_id']
                mean_score = n_row['mean_malignancy']
                
                # Read the clustered 3D physical coordinates.
                pos_x = n_row['pos_x']
                pos_y = n_row['pos_y']
                pos_z = n_row['pos_z']
                
                # Attach the binary classification label.
                if mean_score > 3.0: binary_label, binary_desc = 1, "Malignant lesion"
                elif mean_score < 3.0: binary_label, binary_desc = 0, "Benign lesion"
                else: binary_label, binary_desc = -1, "Indeterminate"
                
                # Attach the multiclass risk label.
                if mean_score <= 2.5: multi_label, multi_desc = 0, "Low risk"
                elif mean_score <= 3.5: multi_label, multi_desc = 1, "Intermediate risk"
                else: multi_label, multi_desc = 2, "High risk"
                
                cube_path_str = "Extraction failed"
                if itk_image is not None:
                    roi_cube = extract_3d_cube(itk_image, pos_x, pos_y, pos_z)
                    if roi_cube is not None:
                        cube_filename = f"{patient_id}_{str(nodule_id)[:8]}.npy"
                        save_path = cube_store_dir / cube_filename
                        np.save(save_path, roi_cube)
                        cube_path_str = str(save_path)
                        roi_success_count += 1
                        print(f"✅ Extracted 3D ROI: {patient_id} - nodule {nodule_id}")
                        
                final_dataset_records.append({
                    'patient_id': patient_id,
                    'data_type': 'High-quality nodule',
                    'nodule_id': str(nodule_id),
                    'cube_file_path': cube_path_str,
                    'mean_score': mean_score,
                    'binary_label': binary_label,
                    'binary_desc': binary_desc,
                    'multi_label': multi_label,
                    'multi_desc': multi_desc
                })
                
        # Case B: Add a healthy negative control when no eligible nodules exist.
        else:
            final_dataset_records.append({
                'patient_id': patient_id,
                'data_type': 'Healthy negative control',
                'nodule_id': 'Healthy lung (no nodule)',
                'cube_file_path': 'No 3D cube required',
                'mean_score': 0.0,
                'binary_label': 0, 
                    'binary_desc': 'Healthy negative control (benign)',
                'multi_label': 0,   
                'multi_desc': 'No lesion; very low risk'
            })
            
    # Save the standard manifest for downstream steps.
    df_final_manifest = pd.DataFrame(final_dataset_records)
    manifest_csv_path = processed_dir / "final_dataset_manifest.csv"
    df_final_manifest.to_csv(manifest_csv_path, index=False, encoding="utf-8-sig")
    

    print("==================================================")
    print(f"-> Manifest contains {len(df_final_manifest)} records.")
    print(f"-> Successfully extracted and saved {roi_success_count} reference nodule cubes.")
    print(f"✅ Results saved to: {manifest_csv_path}")
    print("==================================================")
