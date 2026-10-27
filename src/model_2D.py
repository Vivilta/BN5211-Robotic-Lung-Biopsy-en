import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock2D(nn.Module):
    """
    Core 2D ResNet residual block
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock2D, self).__init__()
        # First 2D convolution
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        # Second 2D convolution
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # Shortcut branch: use a 1x1 convolution to align dimensions when input and output shapes differ
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Residual connection: add the input directly to the convolution output
        out = F.relu(out)
        return out


class ResNet2D18(nn.Module):
    """
    A 2D-ResNet18 tailored for lung nodule classification
    """
    def __init__(self, num_classes=2): # two classes: benign vs. malignant
        super(ResNet2D18, self).__init__()
        self.in_planes = 64

        # 1. Initial 2D convolution (input channel C=1, 64 output features)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # 2. Four residual stages (resolution halves and channel count doubles as depth increases)
        self.layer1 = self._make_layer(BasicBlock2D, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(BasicBlock2D, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(BasicBlock2D, 256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(BasicBlock2D, 512, num_blocks=2, stride=2)
        
        # 3. 2D adaptive average pooling + fully connected classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock2D.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # Extract features
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        # Pooling and flattening
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        
        # Output classification logits
        out = self.fc(out)
        return out

# ==========================================
# Local model forward-pass test (local test entry point)
# ==========================================
if __name__ == "__main__":
    # Create a batch of lung nodule data: batch size 4, one channel, size 32x32 (the largest 2D cross-section, with depth omitted)
    mock_input = torch.randn(4, 1, 32, 32)
    print(f"Input tensor shape: {mock_input.shape}")
    
    # Initialize the model
    model = ResNet2D18(num_classes=2)
    
    # Test the forward pass
    mock_output = model(mock_input)
    print(f"Output classification tensor shape: {mock_output.shape}") # Expected output: torch.Size([4, 2])
    print("✅ 2D-ResNet18 architecture built successfully; forward-pass test passed")
