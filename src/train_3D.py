import os
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# Import previously defined components
from dataset_3D import LungNodule3DDataset
from model_3D import ResNet3D18

# ==========================================
# 🧩 1. 3D data augmentation (Data Augmentation)
# ==========================================
class Transform3D:
    """
    Random 3D flip augmentation for medical images
    """
    def __call__(self, tensor):
        # 50% probability of flipping along the Depth (D) axis
        if torch.rand(1) > 0.5:
            tensor = torch.flip(tensor, dims=[1])
        # 50% probability of flipping along the Height (H) axis
        if torch.rand(1) > 0.5:
            tensor = torch.flip(tensor, dims=[2])
        # 50% probability of flipping along the Width (W) axis
        if torch.rand(1) > 0.5:
            tensor = torch.flip(tensor, dims=[3])
        return tensor

# ==========================================
# 🧠 2. Core PyTorch Lightning model wrapper
# ==========================================
class LitLungNoduleModel(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        # Instantiate the 3D-ResNet18
        self.model = ResNet3D18(num_classes=2)
        # Cross-entropy loss
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Compute accuracy for the current batch
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        # Log metrics (prog_bar=True displays them directly in the terminal progress bar)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    # ==========================================
    # 📉 Learning-rate scheduler configuration (LR Scheduler)
    # ==========================================
    def configure_optimizers(self):
        # Use the AdamW optimizer
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-4)
        
        # Learning-rate decay: halve the learning rate (factor=0.5) if validation val_loss does not decrease for 5 consecutive epochs
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

# ==========================================
# 🚀 3. Automated training entry point
# ==========================================
if __name__ == "__main__":
    # Configure paths
    MANIFEST_PATH = "./data/processed/final_dataset_manifest.csv"
    CUBE_DIR = "./data/processed/all_roi_cubes/"
    
    # Initialize the dataset and apply 3D data augmentation
    full_dataset = LungNodule3DDataset(manifest_path=MANIFEST_PATH, cube_dir=CUBE_DIR, transform=Transform3D())
    
    # Split into training and validation sets (80% train, 20% validation)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Build the DataLoader
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    # Initialize the Lightning model
    lit_model = LitLungNoduleModel(learning_rate=1e-3)
    
    # 🛑 Early stopping: stop if validation loss does not decrease for 10 consecutive epochs
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=10, mode='min')
    
    # 💾 Automatically save the best weights (model checkpoint)
    checkpoint_callback = ModelCheckpoint(
        dirpath='models/checkpoints/',
        filename='3d_resnet18-{epoch:02d}-{val_acc:.2f}',
        save_top_k=1,
        monitor='val_acc',
        mode='max'
    )
    
    # 📊 TensorBoard experiment logger
    logger = TensorBoardLogger("models/logs/", name="baseline_3d_resnet18")
    
    # ⚙️ Instantiate the production-grade Trainer
    trainer = pl.Trainer(
        max_epochs=30,                  # Maximum of 30 epochs
        accelerator='auto',             # Automatically detect and use an available GPU (CUDA)
        devices=1,
        logger=logger,
        callbacks=[early_stop_callback, checkpoint_callback]
    )
    
    print("🚀 Starting the PyTorch Lightning 3D-ResNet18 baseline training pipeline...")
    trainer.fit(lit_model, train_loader, val_loader)
