import os
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# Import the 2D dataset and the DenseNet121 model defined above
from dataset_2D import LungNodule2DDataset
from model_2D_densenet import DenseNet2D121

# ==========================================
# 🧩 2D data augmentation
# ==========================================
class Transform2D:
    def __call__(self, tensor):
        if torch.rand(1) > 0.5:
            tensor = torch.flip(tensor, dims=[1])
        if torch.rand(1) > 0.5:
            tensor = torch.flip(tensor, dims=[2])
        return tensor

# ==========================================
# 🧠 PyTorch Lightning DenseNet wrapper
# ==========================================
class LitLungNoduleDenseNet2D(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        # 🌟 Hot-swap in the DenseNet2D121 model
        self.model = DenseNet2D121(num_classes=2)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
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

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}

# ==========================================
# 🚀 Automated training entry point
# ==========================================
if __name__ == "__main__":
    MANIFEST_PATH = "./data/processed/final_dataset_manifest.csv"
    CUBE_DIR = "./data/processed/all_roi_cubes/"
    
    full_dataset = LungNodule2DDataset(manifest_path=MANIFEST_PATH, cube_dir=CUBE_DIR, transform=Transform2D())
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    lit_model = LitLungNoduleDenseNet2D(learning_rate=1e-3)
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=10, mode='min')
    
    checkpoint_callback = ModelCheckpoint(
        dirpath='models/checkpoints_2d_densenet/',
        filename='2d_densenet-{epoch:02d}-{val_acc:.2f}',
        save_top_k=1, monitor='val_acc', mode='max'
    )
    
    logger = TensorBoardLogger("models/logs/", name="baseline_2d_densenet121")
    
    trainer = pl.Trainer(
        max_epochs=30, accelerator='auto', devices=1, logger=logger,
        callbacks=[early_stop_callback, checkpoint_callback]
    )
    
    print("🚀 Starting the 2D-DenseNet121 training pipeline...")
    trainer.fit(lit_model, train_loader, val_loader)
