import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseLayer2D(nn.Module):
    """
    Core 2D DenseNet dense layer (bottleneck structure)
    Includes: 1x1 convolution for channel compression -> 3x3 convolution for feature extraction
    """
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer2D, self).__init__()
        # Classic DenseNet design: the internal bottleneck channel count is fixed at 4 * growth_rate
        inter_channels = 4 * growth_rate
        
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)
        
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        # x is the input formed by concatenating all previous feature maps
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        # Concatenate the original input and newly extracted features along the channel dimension (dense connection)
        return torch.cat([x, out], 1)


class DenseBlock2D(nn.Module):
    """
    2D DenseNet dense block
    Composed of multiple DenseLayers in sequence; each layer receives the combined outputs of all preceding layers
    """
    def __init__(self, num_layers, in_channels, growth_rate):
        super(DenseBlock2D, self).__init__()
        layers = []
        for i in range(num_layers):
            # The input channel count increases by growth_rate after each layer
            layers.append(DenseLayer2D(in_channels + i * growth_rate, growth_rate))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Transition2D(nn.Module):
    """
    2D transition layer
    Controls channel count between Dense Blocks and halves the spatial feature-map resolution
    """
    def __init__(self, in_channels, out_channels):
        super(Transition2D, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = self.pool(out)
        return out


class DenseNet2D121(nn.Module):
    """
    A lightweight DenseNet-121 tailored for 2D lung nodule classification
    """
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_classes=2):
        super(DenseNet2D121, self).__init__()
        self.growth_rate = growth_rate
        
        # 1. Initial 2D convolution (input channel C=1, 64 output features)
        # Strictly follows the single-channel medical image input convention
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Track the current channel count
        num_features = 64
        
        # 2. Build Dense Blocks and Transitions
        self.features = nn.Sequential()
        
        # Build the network layer by layer
        for i, num_layers in enumerate(block_config):
            # Add a Dense Block
            block = DenseBlock2D(num_layers=num_layers, in_channels=num_features, growth_rate=growth_rate)
            self.features.add_module(f'denseblock{i+1}', block)
            num_features = num_features + num_layers * growth_rate
            
            # If this is not the final Dense Block, append a Transition layer for downsampling
            if i != len(block_config) - 1:
                # Compress the channel count (compression rate 0.5)
                trans = Transition2D(in_channels=num_features, out_channels=num_features // 2)
                self.features.add_module(f'transition{i+1}', trans)
                num_features = num_features // 2
                
        # 3. Global average pooling and classification head
        self.final_bn = nn.BatchNorm2d(num_features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        # Initial feature extraction
        out = F.relu(self.bn1(self.conv1(x)))
        # Dense feature propagation
        out = self.features(out)
        out = F.relu(self.final_bn(out))
        # Pooling and flattening
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        # Classification output
        out = self.fc(out)
        return out

# ==========================================
# Local model forward-pass test (local test entry point)
# ==========================================
if __name__ == "__main__":
    # Create a batch of 2D cross-sectional slices: batch size 4, one channel, size 32x32
    mock_input = torch.randn(4, 1, 32, 32)
    print("=== 2D DenseNet-121 architecture test ===")
    print(f"Input tensor shape: {mock_input.shape}")
    
    # Initialize the model
    model = DenseNet2D121(num_classes=2)
    
    # Test the forward pass
    try:
        mock_output = model(mock_input)
        print(f"Output classification tensor shape: {mock_output.shape}") # Expected output: torch.Size([4, 2])
        print("✅ 2D-DenseNet121 architecture built successfully; forward-pass test passed")
    except Exception as e:
        print(f"❌ Architecture test failed: {e}")
