import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from dataset_3D import LungNodule3DDataset
from train_3D import Transform3D # Reuse the previously defined data augmentation

# ==============================================================
# 🧠 1. Custom 3D CBAM attention mechanism (channel attention + spatial attention)
# ==============================================================

# 1.1 Channel attention module (Channel Attention)
class ChannelAttention3D(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
           
        self.fc = nn.Sequential(
            nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

# 1.2 Spatial attention module (Spatial Attention)
class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention3D, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        # Concatenate channels along dimension 1, then reduce to a single channel
        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

# 1.3 Complete CBAM module
class CBAM3D(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM3D, self).__init__()
        self.ca = ChannelAttention3D(in_planes, ratio)
        self.sa = SpatialAttention3D(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out

# ==============================================================
# 🏗️ 2. Build an advanced 3D ResNet18 with CBAM attention
# ==============================================================
class AttentionResNet3D(nn.Module):
    def __init__(self, num_classes=2):
        super(AttentionResNet3D, self).__init__()
        # Reuse the base 3D convolution layers
        self.stem = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )
        
        # CBAM attention is added to multi-layer feature extraction here
        self.layer1 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            CBAM3D(in_planes=32) # Inject the CBAM module
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            CBAM3D(in_planes=64) # Inject the CBAM module
        )
        
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        # Note that this uses num_classes (two-class output)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ==============================================================
# ⚡ 3. PyTorch Lightning training wrapper (ensure output dimensions match)
# ==============================================================
class LitAttentionModel(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = AttentionResNet3D(num_classes=2)
        # Switch back to the binary cross-entropy loss used previously
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
        
        self.log_dict({'val_loss': loss, 'val_acc': acc}, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }

if __name__ == "__main__":
    MANIFEST_PATH = "data/processed/final_dataset_manifest.csv"
    CUBE_DIR = "data/processed/all_roi_cubes/"
    
    # Fix the random seed so the data split exactly matches the baseline
    torch.manual_seed(42)
    
    full_dataset = LungNodule3DDataset(manifest_path=MANIFEST_PATH, cube_dir=CUBE_DIR, transform=Transform3D())
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    lit_model = LitAttentionModel(learning_rate=1e-3)
    
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=15, mode='min')
    checkpoint_callback = ModelCheckpoint(
        dirpath='models/checkpoints_attention/',
        filename='3d_resnet_cbam-{epoch:02d}-{val_acc:.2f}',
        save_top_k=1, monitor='val_acc', mode='max'
    )
    
    # Update the log name to distinguish this run
    logger = TensorBoardLogger("models/logs/", name="advanced_3d_resnet_cbam")
    
    trainer = pl.Trainer(
        max_epochs=40, accelerator='auto', devices=1, logger=logger,
        callbacks=[early_stop_callback, checkpoint_callback]
    )
    
    print("🚀 Training the advanced network with 3D CBAM attention...")
    trainer.fit(lit_model, train_loader, val_loader)
