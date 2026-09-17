import torch
import torch.nn as nn
import torchvision.models as models
import torchcvnn.nn.modules as c_nn

from .blocks import DoubleConv, DoubleLinear, SingleLinear, SingleConv

class MnistClassifier(nn.Module):
    """
    ResNet-like for MNIST (CVNN).
    Input: (B, 1, 28, 28).
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            DoubleConv(1, 32, conv_mode='complex', normalization='group', activation='CReLU',
                       kernel_size=3, stride=2, padding=1),
            DoubleConv(32, 64, conv_mode='complex', normalization='group', activation='CReLU',
                       kernel_size=3, stride=2, padding=1),
        )
        self.classifier = nn.Sequential(
            DoubleLinear(in_ch=64, out_ch=num_classes, linear_mode='complex',
                         normalization='group', activation='CReLU'),
        )
        self.avg_pooling = c_nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        x = self.features(x)
        x = self.avg_pooling(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x, x.abs()
    
    def get_features(self, x):
        x = self.features(x)
        x = self.avg_pooling(x)
        x = torch.flatten(x, 1)
        for layer in self.classifier[:-1]:
            x = layer(x)
        return x
    
class MstarClassifier(nn.Module):
    """
    ResNet-like for MSTAR (CVNN).
    Input: (B, 1, 64, 64).
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            DoubleConv(1, 32, conv_mode='complex', normalization='group', activation='CReLU',
                       kernel_size=3, stride=2, padding=1, residual=True),
            DoubleConv(32, 64, conv_mode='complex', normalization='group', activation='CReLU',
                       kernel_size=3, stride=2, padding=1, residual=True),
            DoubleConv(64, 128, conv_mode='complex', normalization='group', activation='CReLU',
                       kernel_size=3, stride=2, padding=1, residual=True),
            DoubleConv(128, 256, conv_mode='complex', normalization='group', activation='CReLU',
                       kernel_size=3, stride=2, padding=1, residual=True),

        )
        self.classifier = nn.Sequential(
            DoubleLinear(in_ch=256, out_ch=64, linear_mode='complex',
                         normalization='group', activation='CReLU'),
            DoubleLinear(in_ch=64, out_ch=num_classes, linear_mode='complex',
                         normalization='group', activation='CReLU'),    
        )
        self.avg_pooling = c_nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pooling(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x, x.abs()
    
    def get_features(self, x):
        """Extract the penultimate layer for FID."""
        x = self.features(x)
        x = self.avg_pooling(x)
        x = torch.flatten(x, 1)
        # Replay the linear part up to the penultimate layer
        for layer in self.classifier[:-1]:
            x = layer(x)
        return x