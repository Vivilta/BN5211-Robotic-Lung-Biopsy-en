import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# Import the 3D dataset and ResNet model
from dataset_3D import LungNodule3DDataset
from train_3D import LitLungNoduleModel

if __name__ == "__main__":
    MANIFEST_PATH = "data/processed/final_dataset_manifest.csv"
    CUBE_DIR = "data/processed/all_roi_cubes/"
    
    # ==========================================
    # ✂️ Core ablation point: disable data augmentation
    # ==========================================
    # The previously supplied transform=Transform3D() is removed here
    # Feed the model completely raw data without flips to test whether it severely overfits
    full_dataset = LungNodule3DDataset(manifest_path=MANIFEST_PATH, cube_dir=CUBE_DIR, transform=None)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # Fix the random seed so this split exactly matches the earlier augmented experiment, keeping the variable controlled
    torch.manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    # Initialize the model
    lit_model = LitLungNoduleModel(learning_rate=1e-3)
    
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=10, mode='min')
    
    # Save the model in its dedicated ablation-study folder
    checkpoint_callback = ModelCheckpoint(
        dirpath='models/checkpoints_ablation/',
        filename='3d_resnet18_no_aug-{epoch:02d}-{val_acc:.2f}',
        save_top_k=1, monitor='val_acc', mode='max'
    )
    
    # Use a separate log name to compare this run directly with the augmented version in TensorBoard
    logger = TensorBoardLogger("models/logs/", name="ablation_3d_resnet18_no_aug")
    
    trainer = pl.Trainer(
        max_epochs=30, accelerator='auto', devices=1, logger=logger,
        callbacks=[early_stop_callback, checkpoint_callback]
    )
    
    print("🚀 Starting the ablation study: training the 3D-ResNet18 model without data augmentation...")
    trainer.fit(lit_model, train_loader, val_loader)
