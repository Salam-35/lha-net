"""
Enhanced Backbone with Large Kernel Depthwise Convolutions

This module provides an enhanced version of the LightweightBackbone that
incorporates large kernel depthwise separable convolutions inspired by 3D UX-Net.

Key enhancements:
1. Multi-scale large kernel convolutions (3, 11, 21) for large receptive fields
2. Depthwise separable design for efficiency
3. Adaptive kernel selection (optional)
4. Maintains compatibility with existing LHA-Net architecture
"""

import torch
import torch.nn as nn
from typing import List, Optional

# Import the large kernel modules
from .large_kernel_conv import (
    DepthwiseSeparableConv3D,
    LargeKernelDepthwiseConv3D,
    MultiScaleLargeKernelBlock,
    AdaptiveLargeKernelBlock,
    ChannelShuffle3D
)


class EnhancedBasicBlock3D(nn.Module):
    """
    Enhanced BasicBlock with optional large kernel depthwise convolutions.

    Can operate in two modes:
    1. Standard mode: regular 3x3 convolutions (original behavior)
    2. Large kernel mode: depthwise separable with large kernels
    """
    expansion = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            use_large_kernel: bool = True,
            kernel_size: int = 7
    ):
        super().__init__()

        self.use_large_kernel = use_large_kernel

        if use_large_kernel:
            # First conv: depthwise separable with potentially large kernel
            self.conv1 = DepthwiseSeparableConv3D(
                inplanes, planes,
                kernel_size=kernel_size,
                stride=stride
            )
            # Second conv: depthwise separable
            self.conv2 = DepthwiseSeparableConv3D(
                planes, planes,
                kernel_size=kernel_size,
                stride=1
            )
            self.bn1 = nn.Identity()  # Already in DepthwiseSeparableConv3D
            self.bn2 = nn.Identity()
        else:
            # Original standard convolutions
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm3d(planes)
            self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm3d(planes)

        self.relu = nn.GELU() if use_large_kernel else nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class EnhancedLightweightBackbone(nn.Module):
    """
    Enhanced LightweightBackbone with large kernel depthwise convolutions.

    This is a drop-in replacement for the original LightweightBackbone that:
    1. Uses multi-scale large kernel blocks at each stage
    2. Maintains same output feature dimensions
    3. Achieves larger receptive field with fewer parameters

    Architecture:
    - Stem: 3x3 conv (same as original)
    - Stages 1-4: Multi-scale large kernel blocks with kernels [3, 11, 21]
    """

    def __init__(
            self,
            in_channels: int = 1,
            base_channels: int = 32,
            channel_multipliers: List[float] = [1, 2, 4, 8, 16],
            use_adaptive_kernels: bool = False,
            kernel_sizes_per_stage: Optional[List[List[int]]] = None
    ):
        super().__init__()

        self.use_adaptive = use_adaptive_kernels

        # Default: progressively larger kernels at deeper stages
        if kernel_sizes_per_stage is None:
            kernel_sizes_per_stage = [
                [3, 7, 11],  # Stage 1: focus on local + medium
                [3, 7, 15],  # Stage 2: medium range
                [3, 11, 21],  # Stage 3: medium to large
                [3, 11, 21],  # Stage 4: large receptive field
            ]

        channels = [int(base_channels * mult) for mult in channel_multipliers]

        # Initial convolution - keep as standard 3x3
        self.initial_conv = nn.Sequential(
            nn.Conv3d(in_channels, channels[0], 3, padding=1, bias=False),
            nn.BatchNorm3d(channels[0]),
            nn.GELU()
        )

        # Encoder blocks with large kernel convolutions
        self.encoder_blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            stage_kernels = kernel_sizes_per_stage[i] if i < len(kernel_sizes_per_stage) else [3, 11, 21]

            if use_adaptive_kernels:
                block = nn.Sequential(
                    # Downsampling conv
                    nn.Conv3d(channels[i], channels[i + 1], 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm3d(channels[i + 1]),
                    nn.GELU(),
                    # Adaptive large kernel block
                    AdaptiveLargeKernelBlock(
                        channels=channels[i + 1],
                        kernel_sizes=[3, 7, 11, 15, 21]
                    )
                )
            else:
                block = nn.Sequential(
                    # Downsampling conv
                    nn.Conv3d(channels[i], channels[i + 1], 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm3d(channels[i + 1]),
                    nn.GELU(),
                    # Multi-scale large kernel block
                    MultiScaleLargeKernelBlock(
                        channels=channels[i + 1],
                        kernel_sizes=stage_kernels
                    )
                )

            self.encoder_blocks.append(block)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass returning multi-scale features.

        Returns same structure as original LightweightBackbone:
        [feat_1x, feat_2x, feat_4x, feat_8x, feat_16x]
        """
        features = []

        x = self.initial_conv(x)
        features.append(x)

        for block in self.encoder_blocks:
            x = block(x)
            features.append(x)

        return features


class HybridBackbone(nn.Module):
    """
    Hybrid backbone that combines standard and large kernel convolutions.

    Early stages use standard convolutions for local feature extraction,
    later stages use large kernel convolutions for global context.

    This provides a balance between:
    - Fine-grained local features (important for boundaries)
    - Large receptive field (important for organ context)
    """

    def __init__(
            self,
            in_channels: int = 1,
            base_channels: int = 32,
            channel_multipliers: List[float] = [1, 2, 4, 8, 16],
            large_kernel_stages: List[int] = [2, 3],  # Which stages use large kernels
            kernel_sizes: List[int] = [3, 11, 21]
    ):
        super().__init__()

        channels = [int(base_channels * mult) for mult in channel_multipliers]
        self.large_kernel_stages = large_kernel_stages

        # Initial conv
        self.initial_conv = nn.Sequential(
            nn.Conv3d(in_channels, channels[0], 3, padding=1, bias=False),
            nn.BatchNorm3d(channels[0]),
            nn.GELU()
        )

        # Encoder blocks
        self.encoder_blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            if i in large_kernel_stages:
                # Use large kernel block
                block = nn.Sequential(
                    nn.Conv3d(channels[i], channels[i + 1], 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm3d(channels[i + 1]),
                    nn.GELU(),
                    MultiScaleLargeKernelBlock(
                        channels=channels[i + 1],
                        kernel_sizes=kernel_sizes
                    )
                )
            else:
                # Use standard conv block
                block = nn.Sequential(
                    nn.Conv3d(channels[i], channels[i + 1], 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm3d(channels[i + 1]),
                    nn.GELU(),
                    nn.Conv3d(channels[i + 1], channels[i + 1], 3, padding=1, bias=False),
                    nn.BatchNorm3d(channels[i + 1]),
                    nn.GELU()
                )

            self.encoder_blocks.append(block)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []

        x = self.initial_conv(x)
        features.append(x)

        for block in self.encoder_blocks:
            x = block(x)
            features.append(x)

        return features


# Factory functions for easy instantiation
def create_enhanced_backbone(
        in_channels: int = 1,
        base_channels: int = 32,
        variant: str = "multi_scale"
) -> nn.Module:
    """
    Create enhanced backbone with specified variant.

    Args:
        in_channels: Input channels (1 for CT)
        base_channels: Base channel count
        variant: One of ["multi_scale", "adaptive", "hybrid"]

    Returns:
        Enhanced backbone module
    """
    if variant == "multi_scale":
        return EnhancedLightweightBackbone(
            in_channels=in_channels,
            base_channels=base_channels,
            use_adaptive_kernels=False
        )
    elif variant == "adaptive":
        return EnhancedLightweightBackbone(
            in_channels=in_channels,
            base_channels=base_channels,
            use_adaptive_kernels=True
        )
    elif variant == "hybrid":
        return HybridBackbone(
            in_channels=in_channels,
            base_channels=base_channels,
            large_kernel_stages=[2, 3]
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")


if __name__ == "__main__":
    print("Enhanced Backbone Module - Integration Guide")
    print("=" * 60)
    print("""
To integrate with your existing LHA-Net:

1. Copy these files to your src/models/ directory:
   - large_kernel_conv.py
   - enhanced_backbone.py

2. In lha_net.py, replace the backbone import:

   # Old:
   from .backbone import LightweightBackbone

   # New:
   from .enhanced_backbone import EnhancedLightweightBackbone

3. In your LHANet class, update backbone initialization:

   # Old:
   self.backbone = LightweightBackbone(
       in_channels=in_channels,
       base_channels=base_channels
   )

   # New (multi-scale large kernels):
   self.backbone = EnhancedLightweightBackbone(
       in_channels=in_channels,
       base_channels=base_channels,
       use_adaptive_kernels=False  # or True for adaptive version
   )

4. Update your config (lha_net_config.yaml):

   model:
     backbone_type: "enhanced"  # or "adaptive" or "hybrid"
     use_large_kernels: true
     kernel_sizes: [3, 11, 21]

The enhanced backbone produces the same output shape as the original,
so no changes are needed for the decoder or PMSA modules.
""")