import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class LungNodule3DDataset(Dataset):
    """
    3D lung nodule dataset for PyTorch.
    """
    def __init__(self, manifest_path, cube_dir, transform=None):
        """
        Args:
            manifest_path (str): Path to the manifest CSV file.
            cube_dir (str): Directory containing 3D ROI .npy cubes.
            transform (callable, optional): Optional data transforms.
        """
        self.cube_dir = cube_dir
        self.transform = transform
        
        # 1. Read the source manifest CSV.
        raw_manifest = pd.read_csv(manifest_path)
        
        # 2. Keep only rows whose cube files were successfully extracted.
        self.manifest = raw_manifest[raw_manifest['cube_file_path'].str.endswith('.npy', na=False)].reset_index(drop=True)
        
    def __len__(self):
        """
        Returns the number of usable samples after filtering.
        """
        return len(self.manifest)

    def __getitem__(self, idx):
        """
        Generate one sample and its label.
        """
        # 1. Get file name and label from the manifest
        row = self.manifest.iloc[idx]
        file_name = row['cube_file_path'] 
        
        # Map English or legacy Chinese descriptions to binary class IDs.
        label_str = str(row['binary_desc'])
        if 'binary_label' in row.index and pd.notna(row['binary_label']):
            # Preserve the legacy fallback behavior for indeterminate (-1) rows.
            label = 0 if int(row['binary_label']) == 0 else 1
        else:
            label = 0 if 'benign' in label_str.lower() else 1
        
        # 2. Load the local 32x32x32 3D numpy array.
        pure_file_name = os.path.basename(file_name)
        cube_path = os.path.join(self.cube_dir, pure_file_name)
        cube_array = np.load(cube_path)
        
        # 3. Add the channel dimension: (D, H, W) -> (C, D, H, W), with C=1.
        cube_array = np.expand_dims(cube_array, axis=0)
        
        # 4. Convert to PyTorch Tensor
        image_tensor = torch.tensor(cube_array, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        # 5. Apply transforms if any
        if self.transform:
            image_tensor = self.transform(image_tensor)
            
        return image_tensor, label_tensor

# ==========================================
# Local module test entry point.
# ==========================================
if __name__ == "__main__":
    # Adjust these paths for your environment.
    TEST_MANIFEST = "F:/Lung-Nodule/data/processed/final_dataset_manifest.csv"
    TEST_CUBE_DIR = "F:/Lung-Nodule/data/processed/all_roi_cubes/"
    
    try:
        # Initialize the dataset.
        test_dataset = LungNodule3DDataset(manifest_path=TEST_MANIFEST, cube_dir=TEST_CUBE_DIR)
        print(f"✅ Successfully loaded dataset with {len(test_dataset)} samples.")
        
        # Load the first sample and inspect its shape.
        img, lbl = test_dataset[0]
        print(f"📦 Image Tensor Shape: {img.shape}") # Expected: torch.Size([1, 32, 32, 32])
        print(f"🏷️ Label Tensor: {lbl}")
        
    except Exception as e:
        print(f"❌ Error during local testing: {e}")
