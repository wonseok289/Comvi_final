import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super(DepthwiseSeparableConv, self).__init__()
        padding = dilation
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride, padding, dilation=dilation, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class MicroASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MicroASPP, self).__init__()
        inter_channels = in_channels // 4
        self.branch1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False), nn.BatchNorm2d(inter_channels), nn.ReLU())
        self.branch2 = DepthwiseSeparableConv(in_channels, inter_channels, dilation=2)
        self.branch3 = DepthwiseSeparableConv(in_channels, inter_channels, dilation=4)
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, inter_channels, 1, bias=False), nn.BatchNorm2d(inter_channels), nn.ReLU())
        self.conv_cat = nn.Sequential(nn.Conv2d(inter_channels * 4, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b_pool = F.interpolate(self.pool(x), size=(h, w), mode='bilinear', align_corners=True)
        out = torch.cat([b1, b2, b3, b_pool], dim=1)
        return self.conv_cat(out)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # *** 오류 수정: 입력 채널 수를 정확히 계산 ***
        self.conv = DepthwiseSeparableConv(in_channels + skip_channels, out_channels)

    def forward(self, x, skip_feature):
        x = self.up(x)
        x = torch.cat([x, skip_feature], dim=1)
        return self.conv(x)

class submission_20224258(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super(submission_20224258, self).__init__()
        self.num_classes = 1 if num_classes == 1 else num_classes

        # Encoder
        self.enc1 = DepthwiseSeparableConv(in_channels, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DepthwiseSeparableConv(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DepthwiseSeparableConv(32, 64)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck with ASPP
        self.bottleneck = MicroASPP(64, 64)

        # Decoder
        # *** 오류 수정: skip_channels 크기를 올바르게 지정 ***
        self.dec3 = DecoderBlock(64, 64, 32)
        self.dec2 = DecoderBlock(32, 32, 16)
        self.dec1 = DecoderBlock(16, 16, 8)
        
        # Final Classifier
        self.final_conv = nn.Conv2d(8, self.num_classes, kernel_size=1)

    def forward(self, x):
        H, W = x.size()[2:]

        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        # Bottleneck
        b = self.bottleneck(p3)

        # Decoder
        # *** 오류 수정: DecoderBlock을 통해 업샘플링과 특징 융합을 한번에 처리 ***
        d3 = self.dec3(b, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        
        # Final Output
        out = F.interpolate(d1, size=(H, W), mode='bilinear', align_corners=True)
        out = self.final_conv(out)
        
        return out