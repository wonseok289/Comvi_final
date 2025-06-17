import torch
import torch.nn as nn
import torch.nn.functional as F

class submission_20224258(nn.Module):
    """LCNet 원칙에 기반한 초경량(<20k) 이중 경로 모델"""
    def __init__(self, in_channels=3, num_classes=1):
        super(submission_20224258, self).__init__()
        self.num_classes = 1 if num_classes == 1 else num_classes

        # Stem Layer
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )

        # Encoder
        self.enc1 = DepthwiseSeparableConv(8, 16, stride=2)
        self.enc2 = nn.Sequential(
            DepthwiseSeparableConv(16, 32, stride=2),
            SEBlock(32)  # SE-Block 추가로 특징 강조
        )

        # Decoder
        self.dec1 = DepthwiseSeparableConv(32, 16)
        self.dec2 = DepthwiseSeparableConv(16, 8)

        # Final Classifier
        self.out_conv = nn.Conv2d(8, self.num_classes, kernel_size=1, bias=False)

    def forward(self, x):
        # Encoder
        s_out = self.stem(x)
        enc1_out = self.enc1(s_out)
        enc2_out = self.enc2(enc1_out)

        # Decoder
        dec1_out = F.interpolate(enc2_out, size=enc1_out.size()[2:], mode='bilinear', align_corners=True)
        dec1_out = self.dec1(dec1_out)

        dec2_out = F.interpolate(dec1_out, size=s_out.size()[2:], mode='bilinear', align_corners=True)
        dec2_out = self.dec2(dec2_out)

        # Final Output
        out = F.interpolate(dec2_out, size=x.size()[2:], mode='bilinear', align_corners=True)
        out = self.out_conv(out)
        
        return out

class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution: 파라미터 효율성을 위한 블록"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu2(self.bn2(self.pointwise(self.relu1(self.bn1(self.depthwise(x))))))

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block: 채널별 중요도 학습 모듈"""
    def __init__(self, in_channels, reduction=4):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

