import torch
import torch.nn as nn
import torch.nn.functional as F

class submission_20224258(nn.Module):
    """
    Ultra-Lightweight U-Net model designed for < 20,000 parameters.
    Uses nn.Upsample for parameter-free up-sampling.
    """
    def __init__(self, in_channels=3, num_classes=1):
        super(submission_20224258, self).__init__()
        
        self.num_classes = 1 if num_classes == 1 else num_classes

        # Encoder
        self.enc1 = ConvBlock(in_channels, 8)
        self.enc2 = ConvBlock(8, 16)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = ConvBlock(16, 32)

        # Decoder
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = ConvBlock(16 + 32, 16)  # Skip-connection (enc2 + upconv2)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = ConvBlock(8 + 16, 8)    # Skip-connection (enc1 + upconv1)

        # Output Layer
        self.out_conv = nn.Conv2d(8, self.num_classes, kernel_size=1)

    def forward(self, x):
        # --- Encoder ---
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool(e2)

        # --- Bottleneck ---
        b = self.bottleneck(p2)

        # --- Decoder ---
        d2 = self.up2(b)
        # Skip-connection from encoder
        d2 = torch.cat([e2, d2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        # Skip-connection from encoder
        d1 = torch.cat([e1, d1], dim=1)
        d1 = self.dec1(d1)
        
        # --- Output ---
        out = self.out_conv(d1)
        
        return out

class ConvBlock(nn.Module):
    """A simple convolutional block: Conv -> BN -> ReLU"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

