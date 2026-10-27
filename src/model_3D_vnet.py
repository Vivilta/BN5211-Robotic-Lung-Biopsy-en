import torch
import torch.nn as nn
import torch.nn.functional as F

class VNetConvBlock3D(nn.Module):
    """
    Core 3D VNet feature extraction module
    Includes multiple sequential 3D convolution layers with residual shortcut connections and PReLU activation suited to medical images
    """
    def __init__(self, channels, num_convs):
        super(VNetConvBlock3D, self).__init__()
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv3d(channels, channels, kernel_size=5, padding=2, bias=False))
            layers.append(nn.BatchNorm3d(channels))
            layers.append(nn.PReLU(channels)) # VNet signature PReLU activation
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        # Classic local residual addition design
        return self.conv_block(x) + x


class VNet3DClassifier(nn.Module):
    """
    A VNet-Classifier tailored for 3D lung nodule classification.
    It retains the original VNet downsampling (encoder) path and large receptive-field extraction using 5x5x5 convolution kernels,
    and replaces the segmentation upsampling decoder with 3D global pooling and a classification head.
    """
    def __init__(self, num_classes=2):
        super(VNet3DClassifier, self).__init__()
        
        # 1. Stage 1 (input 1x32x32x32 -> output 16x32x32x32)
        self.in_conv = nn.Conv3d(1, 16, kernel_size=5, padding=2, bias=False)
        self.in_bn = nn.BatchNorm3d(16)
        self.in_prelu = nn.PReLU(16)
        
        # 2. Stage 2 (downsampling -> 32x16x16x16)
        self.down1 = nn.Conv3d(16, 32, kernel_size=2, stride=2, bias=False) # Use a stride-2 2x2x2 convolution instead of pooling
        self.down1_bn = nn.BatchNorm3d(32)
        self.down1_prelu = nn.PReLU(32)
        self.block1 = VNetConvBlock3D(32, num_convs=2)
        
        # 3. Stage 3 (downsampling -> 64x8x8x8)
        self.down2 = nn.Conv3d(32, 64, kernel_size=2, stride=2, bias=False)
        self.down2_bn = nn.BatchNorm3d(64)
        self.down2_prelu = nn.PReLU(64)
        self.block2 = VNetConvBlock3D(64, num_convs=3)
        
        # 4. Stage 4 (downsampling -> 128x4x4x4)
        self.down3 = nn.Conv3d(64, 128, kernel_size=2, stride=2, bias=False)
        self.down3_bn = nn.BatchNorm3d(128)
        self.down3_prelu = nn.PReLU(128)
        self.block3 = VNetConvBlock3D(128, num_convs=3)

        # 5. Stage 5 (downsampling -> 256x2x2x2)
        self.down4 = nn.Conv3d(128, 256, kernel_size=2, stride=2, bias=False)
        self.down4_bn = nn.BatchNorm3d(256)
        self.down4_prelu = nn.PReLU(256)
        self.block4 = VNetConvBlock3D(256, num_convs=3)
        
        # 6. 3D adaptive average pooling + classification output layer
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # Stage 1
        out = self.in_prelu(self.in_bn(self.in_conv(x)))
        
        # Stage 2 (downsampling + feature extraction)
        out = self.down1_prelu(self.down1_bn(self.down1(out)))
        out = self.block1(out)
        
        # Stage 3
        out = self.down2_prelu(self.down2_bn(self.down2(out)))
        out = self.block2(out)
        
        # Stage 4
        out = self.down3_prelu(self.down3_bn(self.down3(out)))
        out = self.block3(out)

        # Stage 5
        out = self.down4_prelu(self.down4_bn(self.down4(out)))
        out = self.block4(out)
        
        # Pooling and final flattening
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        
        # Classification
        out = self.fc(out)
        return out

# ==========================================
# Local model forward-pass test (local test entry point)
# ==========================================
if __name__ == "__main__":
    # Create a batch of 3D volume data: batch size 4, one channel, size 32x32x32
    mock_input = torch.randn(4, 1, 32, 32, 32)
    print("=== 3D VNet-Classifier architecture test ===")
    print(f"Input tensor shape: {mock_input.shape}")
    
    # Initialize the model
    model = VNet3DClassifier(num_classes=2)
    
    # Test the forward pass
    try:
        mock_output = model(mock_input)
        print(f"Output classification tensor shape: {mock_output.shape}") # Expected output: torch.Size([4, 2])
        print("✅ 3D-VNet architecture built successfully; forward-pass test passed")
    except Exception as e:
        print(f"❌ Architecture test failed: {e}")
