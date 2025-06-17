import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    """Dilation을 지원하는 경량 합성곱 블록"""
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super(DepthwiseSeparableConv, self).__init__()
        # Dilation에 맞는 패딩 계산
        padding = dilation
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.pointwise(self.depthwise(x))))

class MicroASPP(nn.Module):
    """다양한 스케일의 특징을 포착하기 위한 초소형 ASPP 모듈"""
    def __init__(self, in_channels, out_channels):
        super(MicroASPP, self).__init__()
        inter_channels = in_channels // 4
        
        # 병렬 브랜치: 1x1 conv, dilated conv, global pooling
        self.branch1x1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False), nn.BatchNorm2d(inter_channels), nn.ReLU())
        self.branch_dil_2 = DepthwiseSeparableConv(in_channels, inter_channels, dilation=2)
        self.branch_dil_4 = DepthwiseSeparableConv(in_channels, inter_channels, dilation=4)
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, inter_channels, 1, bias=False), nn.BatchNorm2d(inter_channels), nn.ReLU())
        
        # 모든 브랜치의 출력을 합친 후 1x1 Conv로 채널 조정
        self.conv_cat = nn.Sequential(nn.Conv2d(inter_channels * 4, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())

    def forward(self, x):
        h, w = x.size()[2], x.size()[3]
        
        b1 = self.branch1x1(x)
        b2 = self.branch_dil_2(x)
        b3 = self.branch_dil_4(x)
        b_pool = F.interpolate(self.pool(x), size=(h, w), mode='bilinear', align_corners=True)
        
        out = torch.cat([b1, b2, b3, b_pool], dim=1)
        return self.conv_cat(out)

class submission_20224258(nn.Module):
    """ASPP를 탑재한 최종 초경량 모델 (< 2,000 params)"""
    def __init__(self, in_channels=3, num_classes=1):
        super(submission_20224258, self).__init__()
        self.num_classes = 1 if num_classes == 1 else num_classes

        # Encoder
        self.enc1 = DepthwiseSeparableConv(in_channels, 8, stride=2)
        self.enc2 = DepthwiseSeparableConv(8, 16, stride=2)
        
        # Bottleneck with Micro-ASPP
        self.aspp = MicroASPP(16, 16)
        
        # Decoder
        self.dec1 = DepthwiseSeparableConv(16 + 8, 8) # Skip-connection (enc1 + up_b)
        
        # Output Layer
        self.final_conv = nn.Conv2d(8, self.num_classes, kernel_size=1)

    def forward(self, x):
        H, W = x.size()[2:]

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        
        # Bottleneck
        b = self.aspp(e2)
        
        # Decoder
        d1 = F.interpolate(b, size=e1.size()[2:], mode='bilinear', align_corners=True)
        d1 = torch.cat([e1, d1], dim=1) # Skip Connection
        d1 = self.dec1(d1)
        
        # Final Upsampling and Output
        out = F.interpolate(d1, size=(H, W), mode='bilinear', align_corners=True)
        out = self.final_conv(out)
        
        return out