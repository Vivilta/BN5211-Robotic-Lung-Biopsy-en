import yaml
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import os

# Suppress low-level ITK warnings to keep output concise.
sitk.ProcessObject_SetGlobalWarningDisplay(False)

# ==========================================
# 🛠️ Step 0: Load configuration.
# ==========================================
def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ==========================================
# 🛠️ Step 1: De-identification.
# ==========================================
def read_3d_image(file_path: str):
    """
    Read a LUNA16 .mhd image with SimpleITK.

    The image's spatial geometry and voxel data are returned for processing.
    """
    return sitk.ReadImage(file_path)

# ==========================================
# 🛠️ Step 2: 3D spatial resampling.
# ==========================================
def resample_image(itk_image, new_spacing=[1.0, 1.0, 1.0]):
    """
    Use 3D linear interpolation to resample the CT volume to isotropic spacing,
    such as 1 mm x 1 mm x 1 mm, while preserving its physical geometry.
    """
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    new_size = [
        int(np.round(original_size[0] * (original_spacing[0] / new_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / new_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / new_spacing[2])))
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(-1000) # Fill the outside background with an air-like value.
    resample.SetInterpolator(sitk.sitkLinear)

    return resample.Execute(itk_image)

# ==========================================
# 🛠️ Step 3: 3D lung parenchyma segmentation.
# ==========================================
def segment_lung_mask(itk_image):
    """
    Use mathematical morphology to remove the chest wall and scan table and
    extract the closed lung regions.
    1. Threshold the image; lung air typically has HU values below -400.
    2. Apply 3D morphological closing with a ball-shaped structuring element
       to fill cavities around large vessels while retaining peripheral nodules.
    """
    # Threshold the image to create a coarse lung air mask.
    binary_mask = sitk.BinaryThreshold(itk_image, lowerThreshold=-1000, upperThreshold=-400, insideValue=1, outsideValue=0)

    # Configure a 3D ball kernel for morphological closing.
    closing_filter = sitk.BinaryMorphologicalClosingImageFilter()
    closing_filter.SetKernelRadius([3, 3, 3]) # 3-voxel radius.
    closing_filter.SetKernelType(sitk.sitkBall)

    cleaned_mask = closing_filter.Execute(binary_mask)
    return cleaned_mask

# ==========================================
# 🛠️ Step 4: HU normalization.
# ==========================================
def normalize_hu(image_array: np.ndarray, min_bound: float = -1000.0, max_bound: float = 400.0) -> np.ndarray:
    """
    Clip Hounsfield units (HU) to the range [-1000, 400] and linearly scale
    the values to the [0.0, 1.0] float32 range.
    """
    image_array = np.clip(image_array, min_bound, max_bound)
    image_array = (image_array - min_bound) / (max_bound - min_bound)
    return image_array.astype(np.float32)

# ==========================================
# 🚀 Automated preprocessing pipeline.
# ==========================================
if __name__ == "__main__":
    cfg = load_config()
    luna_root = Path(cfg['data']['raw_luna_dir'])

    # Set the output directory and create it if needed.
    output_dir = Path(cfg['data']['processed_dir']) / "preprocessed_cubes"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Scanning and loading the LUNA16 dataset...")
    mhd_files = list(luna_root.rglob("*.mhd"))

    if not mhd_files:
        print(f"[Error] No .mhd files found in {luna_root}.")
    else:
        print("==================================================")
        print(f"🎬 Found {len(mhd_files)} source images.")
        print(f"💾 Output directory: {output_dir}")
        print("==================================================")

        success_count = 0

        for idx, file_path in enumerate(mhd_files):
            # Use the case's unique identifier as the output filename.
            case_uid = file_path.stem
            save_path = output_dir / f"{case_uid}_preprocessed.npy"

            # Skip existing outputs to support resuming an interrupted run.
            if save_path.exists():
                print(f"⏩ Progress [{idx+1}/{len(mhd_files)}] Existing output found; skipping: {case_uid[:20]}...")
                success_count += 1
                continue

            print(f"⚙️ Progress [{idx+1}/{len(mhd_files)}] Processing: {case_uid[:20]}...")

            try:
                # 1. Read the source image.
                original_image = read_3d_image(str(file_path))

                # 2. Resample the image spatially.
                resampled_image = resample_image(original_image, new_spacing=[1.0, 1.0, 1.0])

                # 3. Segment the lung parenchyma.
                lung_mask_image = segment_lung_mask(resampled_image)

                # 4. Convert to arrays and apply the mask for denoising.
                img_array = sitk.GetArrayFromImage(resampled_image)
                mask_array = sitk.GetArrayFromImage(lung_mask_image)

                # Fill non-lung regions with air (-1000 HU).
                segmented_lung_array = np.where(mask_array == 1, img_array, -1000)

                # 5. Normalize HU values.
                final_array = normalize_hu(segmented_lung_array)

                # 6. Save the NumPy array.
                np.save(save_path, final_array)
                success_count += 1

            except Exception as e:
                print(f"  -> ⚠️ Failed to process case {case_uid[:15]}; skipping. Reason: {e}")

        print("\n==================================================")
        print(f"Successfully processed {success_count}/{len(mhd_files)} 3D images.")
        print(f"Results saved to: {output_dir}")
        print("==================================================")
