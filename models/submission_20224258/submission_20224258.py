import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution. 파라미터를 크게 줄입니다."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class FeatureFusionModule(nn.Module):
    """상세 경로와 문맥 경로의 특징을 융합하는 모듈입니다."""
    def __init__(self, in_channels, out_channels):
        super(FeatureFusionModule, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, detail_feature, context_feature):
        x = torch.cat([detail_feature, context_feature], dim=1)
        x = self.conv(x)
        return x

class submission_20224258(nn.Module):
    """
    < 20,000 파라미터를 목표로 한 Dual-Path 경량 네트워크
    """
    def __init__(self, in_channels=3, num_classes=1):
        super(submission_20224258, self).__init__()
        self.num_classes = 1 if num_classes == 1 else num_classes

        # 1. 상세 경로 (Detail Path) - 고해상도, 세부 특징 추출
        self.detail_path = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        # 2. 문맥 경로 (Context Path) - 저해상도, 전역 문맥 추출 (매우 경량)
        self.context_path = nn.Sequential(
            DepthwiseSeparableConv(in_channels, 8, stride=2),
            DepthwiseSeparableConv(8, 16, stride=2),
            DepthwiseSeparableConv(16, 32, stride=2)
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.context_arm = nn.Conv2d(32, 32, kernel_size=1, bias=False)
        
        # 3. 특징 융합 모듈 (Feature Fusion Module)
        self.fusion_module = FeatureFusionModule(16 + 32, 32)

        # 4. 최종 출력 레이어
        self.out_conv = nn.Conv2d(32, self.num_classes, kernel_size=1, bias=False)

    def forward(self, x):
        H, W = x.size()[2:]

        # 상세 경로 Forward
        detail_out = self.detail_path(x)

        # 문맥 경로 Forward
        context_out = self.context_path(x)
        context_gap = self.global_pool(context_out)
        context_arm_out = self.context_arm(context_gap)
        context_out = context_out * torch.sigmoid(context_arm_out)
        
        # 문맥 특징을 상세 특징의 크기로 업샘플링
        context_out = F.interpolate(context_out, size=detail_out.size()[2:], mode='bilinear', align_corners=True)

        # 특징 융합
        fused_out = self.fusion_module(detail_out, context_out)

        # 최종 예측
        out = F.interpolate(fused_out, size=(H, W), mode='bilinear', align_corners=True)
        out = self.out_conv(out)
        
        return out