import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# Import the 2D cross-sectional dataset
from dataset_2D import LungNodule2DDataset
from train_2D import Transform2D

# ==============================================================
# 🧠 1. Custom 2D CBAM attention mechanism (channel attention + spatial attention)
# ==============================================================

# 1.1 Channel attention module (Channel Attention 2D)
class ChannelAttention2D(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention2D, self).__init__()
        # Use 2D pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

# 1.2 Spatial attention module (Spatial Attention 2D)
class SpatialAttention2D(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention2D, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        # Input has 2 channels (concatenated average + max); output has 1 channel
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

# 1.3 Complete 2D CBAM module
class CBAM2D(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM2D, self).__init__()
        self.ca = ChannelAttention2D(in_planes, ratio)
        self.sa = SpatialAttention2D(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out

# ==============================================================
# 🏗️ 2. Build a custom 2D ResNet with CBAM attention
# ==============================================================
class AttentionResNet2D(nn.Module):
    def __init__(self, num_classes=2):
        super(AttentionResNet2D, self).__init__()
        # Match the network depth of the 3D version to keep the comparison controlled
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            CBAM2D(in_planes=32) # Inject the 2D CBAM module
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            CBAM2D(in_planes=64) # Inject the 2D CBAM module
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
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
# ⚡ 3. PyTorch Lightning training wrapper
# ==============================================================
class LitAttentionModel2D(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = AttentionResNet2D(num_classes=2)
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
    
    # Fix the random seed to align the data split
    torch.manual_seed(42)
    
    full_dataset = LungNodule2DDataset(manifest_path=MANIFEST_PATH, cube_dir=CUBE_DIR, transform=Transform2D())
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    lit_model = LitAttentionModel2D(learning_rate=1e-3)
    
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=15, mode='min')
    checkpoint_callback = ModelCheckpoint(
        dirpath='models/checkpoints_attention_2d/',
        filename='2d_resnet_cbam-{epoch:02d}-{val_acc:.2f}',
        save_top_k=1, monitor='val_acc', mode='max'
    )
    
    # Update the log name
    logger = TensorBoardLogger("models/logs/", name="advanced_2d_resnet_cbam")
    
    trainer = pl.Trainer(
        max_epochs=40, accelerator='auto', devices=1, logger=logger,
        callbacks=[early_stop_callback, checkpoint_callback]
    )
    
    print("🚀 Training the advanced network with 2D CBAM attention...")
    trainer.fit(lit_model, train_loader, val_loader)