import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )
    def forward(self, x):
        return self.conv(x)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=False)
        
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.relu1(self.bn1(self.depthwise(x)))
        x = self.relu2(self.bn2(self.pointwise(x)))
        return x

class submission_20224258(nn.Module):
    """
    파라미터 20,000개 미만을 보장하는 최종 경량 U-Net
    """
    def __init__(self, in_channels=3, num_classes=1):
        super(submission_20224258, self).__init__()
        self.num_classes = 1 if num_classes == 2 else num_classes

        # Encoder
        self.enc_stem = ConvBNReLU(in_channels, 16, stride=2)
        self.enc1 = DepthwiseSeparableConv(16, 24, stride=2)
        self.enc2 = DepthwiseSeparableConv(24, 32, stride=2)
        self.bottleneck = DepthwiseSeparableConv(32, 48, stride=2)

        # Decoder
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = DepthwiseSeparableConv(48 + 32, 32) # Concat with enc2

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = DepthwiseSeparableConv(32 + 24, 24) # Concat with enc1

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = DepthwiseSeparableConv(24 + 16, 16) # Concat with stem
        
        self.up0 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec0 = DepthwiseSeparableConv(16, 8)     # No skip connection from original input

        # Final Classifier
        self.classifier = nn.Conv2d(8, self.num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e0 = self.enc_stem(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        
        # Bottleneck
        b = self.bottleneck(e2)
        
        # Decoder
        d3 = self.up3(b)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e0], dim=1)
        d1 = self.dec1(d1)
        
        d0 = self.up0(d1)
        d0 = self.dec0(d0)

        out = self.classifier(d0)
        return out