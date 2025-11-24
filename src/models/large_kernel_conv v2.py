"""
Large Kernel Depthwise Convolution Module for LHA-Net

Inspired by 3D UX-Net and RepUX-Net, this module implements:
1. Depthwise separable convolutions with large kernels (up to 21x21x21)
2. Multi-scale parallel branches for capturing different receptive fields
3. Inverted bottleneck design for efficiency
4. Channel shuffle for information mixing in depthwise operations

Key insight: Large kernel depthwise conv achieves similar receptive field
to transformers but with much fewer parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class DepthwiseSeparableConv3D(nn.Module):
    """
    Depthwise separable 3D convolution.

    Splits convolution into:
    1. Depthwise conv: spatial filtering per channel
    2. Pointwise conv: channel mixing

    This reduces parameters from C_in * C_out * K^3 to C_in * K^3 + C_in * C_out
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        bias: bool = False
    ):
        super().__init__()

        if padding is None:
            padding = kernel_size // 2

        # Depthwise convolution - each channel processed independently
        self.depthwise = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,  # Key: groups = in_channels
            bias=bias
        )

        # Pointwise convolution - 1x1x1 conv for channel mixing
        self.pointwise = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias
        )

        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x


class LargeKernelDepthwiseConv3D(nn.Module):
    """
    Large kernel depthwise convolution with decomposition for efficiency.

    For very large kernels (>7), we use spatial decomposition:
    K×K×K → K×1×1 + 1×K×1 + 1×1×K

    This reduces computation from O(K^3) to O(3K) while maintaining
    effective receptive field.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 21,
        use_decomposition: bool = True
    ):
        super().__init__()

        self.channels = channels
        self.kernel_size = kernel_size
        self.use_decomposition = use_decomposition and kernel_size > 7

        padding = kernel_size // 2

        if self.use_decomposition:
            # Decomposed large kernel: three 1D convolutions
            self.conv_d = nn.Conv3d(
                channels, channels,
                kernel_size=(kernel_size, 1, 1),
                padding=(padding, 0, 0),
                groups=channels,
                bias=False
            )
            self.conv_h = nn.Conv3d(
                channels, channels,
                kernel_size=(1, kernel_size, 1),
                padding=(0, padding, 0),
                groups=channels,
                bias=False
            )
            self.conv_w = nn.Conv3d(
                channels, channels,
                kernel_size=(1, 1, kernel_size),
                padding=(0, 0, padding),
                groups=channels,
                bias=False
            )
        else:
            # Direct large kernel convolution
            self.conv = nn.Conv3d(
                channels, channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=channels,
                bias=False
            )

        self.bn = nn.BatchNorm3d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_decomposition:
            # Sequential 1D convolutions along each axis
            x = self.conv_d(x)
            x = self.conv_h(x)
            x = self.conv_w(x)
        else:
            x = self.conv(x)

        x = self.bn(x)
        return x


class MultiScaleLargeKernelBlock(nn.Module):
    """
    Multi-scale large kernel block with parallel branches.

    Uses multiple kernel sizes (e.g., 3, 11, 21) in parallel to capture
    features at different scales simultaneously, similar to SCANeXt and
    InceptionNeXt architectures.

    Architecture:
    Input → Split into groups → [3x3x3, 11x11x11, 21x21x21] → Concat → Mix
    """

    def __init__(
        self,
        channels: int,
        kernel_sizes: List[int] = [3, 11, 21],
        expansion_ratio: float = 4.0
    ):
        super().__init__()

        self.channels = channels
        self.kernel_sizes = kernel_sizes
        self.num_branches = len(kernel_sizes)

        # Split channels across branches
        self.branch_channels = channels // self.num_branches
        remainder = channels % self.num_branches

        # Create parallel branches with different kernel sizes
        self.branches = nn.ModuleList()
        for i, ks in enumerate(kernel_sizes):
            # Last branch gets remainder channels
            branch_ch = self.branch_channels + (remainder if i == self.num_branches - 1 else 0)

            branch = LargeKernelDepthwiseConv3D(
                channels=branch_ch,
                kernel_size=ks,
                use_decomposition=(ks > 7)
            )
            self.branches.append(branch)

        # Channel shuffle for information mixing between branches
        self.channel_shuffle = ChannelShuffle3D(groups=self.num_branches)

        # Inverted bottleneck: expand -> process -> compress
        hidden_channels = int(channels * expansion_ratio)
        self.inverted_bottleneck = nn.Sequential(
            nn.Conv3d(channels, hidden_channels, 1, bias=False),
            nn.BatchNorm3d(hidden_channels),
            nn.GELU(),
            nn.Conv3d(hidden_channels, channels, 1, bias=False),
            nn.BatchNorm3d(channels)
        )

        # Learnable scale for residual
        self.gamma = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        # Split input across branches
        splits = torch.split(x, [self.branch_channels] * (self.num_branches - 1) +
                            [self.channels - self.branch_channels * (self.num_branches - 1)], dim=1)

        # Process each branch with different kernel size
        branch_outputs = []
        for branch, split in zip(self.branches, splits):
            branch_outputs.append(branch(split))

        # Concatenate and shuffle for information mixing
        x = torch.cat(branch_outputs, dim=1)
        x = self.channel_shuffle(x)

        # Inverted bottleneck for channel interaction
        x = self.inverted_bottleneck(x)

        # Residual connection with learnable scale
        return identity + self.gamma * x


class ChannelShuffle3D(nn.Module):
    """
    Channel shuffle operation to enable information flow between groups
    in grouped/depthwise convolutions.
    """

    def __init__(self, groups: int):
        super().__init__()
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, d, h, w = x.shape
        channels_per_group = channels // self.groups

        # Reshape: (B, C, D, H, W) -> (B, G, C/G, D, H, W)
        x = x.view(batch, self.groups, channels_per_group, d, h, w)

        # Transpose: (B, G, C/G, D, H, W) -> (B, C/G, G, D, H, W)
        x = x.transpose(1, 2).contiguous()

        # Flatten: (B, C/G, G, D, H, W) -> (B, C, D, H, W)
        x = x.view(batch, channels, d, h, w)

        return x


class AdaptiveLargeKernelBlock(nn.Module):
    """
    Adaptive large kernel block that learns to select optimal kernel size.

    Instead of using fixed kernel sizes, this block learns attention weights
    for different kernel sizes and adaptively combines their outputs.

    This is particularly useful for multi-organ segmentation where different
    organs benefit from different receptive field sizes.
    """

    def __init__(
        self,
        channels: int,
        kernel_sizes: List[int] = [3, 7, 11, 15, 21],
        reduction_ratio: int = 4
    ):
        super().__init__()

        self.channels = channels
        self.kernel_sizes = kernel_sizes
        self.num_scales = len(kernel_sizes)

        # Create convolutions for each kernel size
        self.convs = nn.ModuleList([
            LargeKernelDepthwiseConv3D(
                channels=channels,
                kernel_size=ks,
                use_decomposition=(ks > 7)
            )
            for ks in kernel_sizes
        ])

        # Attention mechanism to select kernel sizes
        self.scale_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, self.num_scales),
            nn.Softmax(dim=1)
        )

        # Pointwise conv for channel mixing after scale selection
        self.mixing = nn.Sequential(
            nn.Conv3d(channels, channels, 1, bias=False),
            nn.BatchNorm3d(channels),
            nn.GELU()
        )

        # Learnable residual scale
        self.gamma = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        batch_size = x.shape[0]

        # Compute attention weights for each kernel size
        attention = self.scale_attention(x)  # (B, num_scales)

        # Apply each kernel size
        scale_outputs = []
        for conv in self.convs:
            scale_outputs.append(conv(x))

        # Stack outputs: (B, num_scales, C, D, H, W)
        stacked = torch.stack(scale_outputs, dim=1)

        # Apply attention weights
        # attention: (B, num_scales) -> (B, num_scales, 1, 1, 1, 1)
        attention = attention.view(batch_size, self.num_scales, 1, 1, 1, 1)

        # Weighted sum across scales
        x = (stacked * attention).sum(dim=1)

        # Channel mixing
        x = self.mixing(x)

        return identity + self.gamma * x


class LargeKernelEncoderBlock(nn.Module):
    """
    Encoder block with large kernel depthwise convolutions.

    Replaces standard 3x3 convolutions with large kernel depthwise separable
    convolutions while maintaining compatibility with existing architecture.

    Structure:
    1. Depthwise large kernel conv for spatial features
    2. Pointwise conv for channel mixing
    3. GELU activation
    4. Second depthwise large kernel conv
    5. Pointwise conv
    6. Residual connection
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 1,
        use_multi_scale: bool = True,
        kernel_sizes: List[int] = [3, 11, 21]
    ):
        super().__init__()

        self.use_multi_scale = use_multi_scale
        self.needs_projection = (in_channels != out_channels) or (stride != 1)

        if use_multi_scale and in_channels == out_channels and stride == 1:
            # Use multi-scale block for same-dimension processing
            self.main = MultiScaleLargeKernelBlock(
                channels=out_channels,
                kernel_sizes=kernel_sizes
            )
        else:
            # Use standard large kernel block for dimension changes
            self.main = nn.Sequential(
                # First conv: potentially change channels and stride
                DepthwiseSeparableConv3D(
                    in_channels, out_channels,
                    kernel_size=kernel_size,
                    stride=stride
                ),
                nn.GELU(),

                # Second conv: maintain dimensions
                DepthwiseSeparableConv3D(
                    out_channels, out_channels,
                    kernel_size=kernel_size,
                    stride=1
                )
            )

        # Projection for residual connection if dimensions change
        if self.needs_projection:
            self.projection = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.projection = nn.Identity()

        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.projection(x)
        out = self.main(x)

        if not self.use_multi_scale or self.needs_projection:
            out = out + identity
            out = self.activation(out)

        return out


class LargeKernelBackbone(nn.Module):
    """
    Complete backbone using large kernel depthwise convolutions.

    This is a drop-in replacement for the existing LightweightBackbone
    that uses large kernel convolutions for better receptive field.

    Architecture follows UX-Net style:
    - Initial stem with small kernel
    - 4 stages with increasing channels and decreasing resolution
    - Each stage uses multi-scale large kernel blocks
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        channel_multipliers: List[float] = [1, 2, 4, 8, 16],
        kernel_sizes_per_stage: List[List[int]] = None,
        use_adaptive_kernels: bool = False
    ):
        super().__init__()

        # Default kernel sizes: smaller at early stages, larger at deeper stages
        if kernel_sizes_per_stage is None:
            kernel_sizes_per_stage = [
                [3, 7, 11],      # Stage 1: small to medium
                [3, 7, 11],      # Stage 2: small to medium
                [3, 11, 15],     # Stage 3: medium
                [3, 11, 21],     # Stage 4: medium to large
            ]

        channels = [int(base_channels * mult) for mult in channel_multipliers]

        # Initial stem - standard convolution
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(channels[0]),
            nn.GELU()
        )

        # Encoder stages with large kernel blocks
        self.stages = nn.ModuleList()
        for i in range(len(channels) - 1):
            stage_kernels = kernel_sizes_per_stage[i] if i < len(kernel_sizes_per_stage) else [3, 11, 21]

            if use_adaptive_kernels:
                stage = nn.Sequential(
                    # Downsample
                    nn.Conv3d(channels[i], channels[i+1], 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm3d(channels[i+1]),
                    nn.GELU(),
                    # Adaptive large kernel block
                    AdaptiveLargeKernelBlock(
                        channels=channels[i+1],
                        kernel_sizes=[3, 7, 11, 15, 21]
                    )
                )
            else:
                stage = nn.Sequential(
                    # Downsample
                    nn.Conv3d(channels[i], channels[i+1], 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm3d(channels[i+1]),
                    nn.GELU(),
                    # Multi-scale large kernel block
                    MultiScaleLargeKernelBlock(
                        channels=channels[i+1],
                        kernel_sizes=stage_kernels
                    )
                )

            self.stages.append(stage)

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

        x = self.stem(x)
        features.append(x)

        for stage in self.stages:
            x = stage(x)
            features.append(x)

        return features


def create_large_kernel_backbone(
    in_channels: int = 1,
    base_channels: int = 32,
    use_adaptive: bool = False
) -> LargeKernelBackbone:
    """
    Factory function to create large kernel backbone.

    Args:
        in_channels: Number of input channels (1 for CT)
        base_channels: Base channel count
        use_adaptive: Whether to use adaptive kernel selection

    Returns:
        LargeKernelBackbone instance
    """
    return LargeKernelBackbone(
        in_channels=in_channels,
        base_channels=base_channels,
        use_adaptive_kernels=use_adaptive
    )


# Utility function to add large kernel capability to existing blocks
def convert_to_depthwise_separable(conv: nn.Conv3d) -> DepthwiseSeparableConv3D:
    """
    Convert a standard Conv3d to depthwise separable version.

    Useful for converting existing models without complete redesign.
    """
    return DepthwiseSeparableConv3D(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size[0] if isinstance(conv.kernel_size, tuple) else conv.kernel_size,
        stride=conv.stride[0] if isinstance(conv.stride, tuple) else conv.stride,
        padding=conv.padding[0] if isinstance(conv.padding, tuple) else conv.padding,
        bias=conv.bias is not None
    )


if __name__ == "__main__":
    # Test the modules
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Testing Large Kernel Convolution Modules")
    print("=" * 50)

    # Test input
    batch_size = 2
    channels = 64
    depth, height, width = 32, 64, 64
    x = torch.randn(batch_size, channels, depth, height, width).to(device)

    # Test 1: Depthwise Separable Conv
    print("\n1. Testing DepthwiseSeparableConv3D...")
    dw_conv = DepthwiseSeparableConv3D(channels, channels * 2, kernel_size=7).to(device)
    out = dw_conv(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")

    # Count parameters
    params = sum(p.numel() for p in dw_conv.parameters())
    standard_params = channels * (channels * 2) * 7 * 7 * 7  # Standard conv params
    print(f"   Parameters: {params:,} (vs {standard_params:,} for standard conv)")
    print(f"   Reduction: {standard_params / params:.1f}x")

    # Test 2: Large Kernel Depthwise Conv with decomposition
    print("\n2. Testing LargeKernelDepthwiseConv3D (21x21x21)...")
    lk_conv = LargeKernelDepthwiseConv3D(channels, kernel_size=21).to(device)
    out = lk_conv(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    params = sum(p.numel() for p in lk_conv.parameters())
    print(f"   Parameters: {params:,}")

    # Test 3: Multi-scale block
    print("\n3. Testing MultiScaleLargeKernelBlock...")
    ms_block = MultiScaleLargeKernelBlock(channels, kernel_sizes=[3, 11, 21]).to(device)
    out = ms_block(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    params = sum(p.numel() for p in ms_block.parameters())
    print(f"   Parameters: {params:,}")

    # Test 4: Adaptive kernel block
    print("\n4. Testing AdaptiveLargeKernelBlock...")
    adaptive_block = AdaptiveLargeKernelBlock(channels, kernel_sizes=[3, 7, 11, 15, 21]).to(device)
    out = adaptive_block(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    params = sum(p.numel() for p in adaptive_block.parameters())
    print(f"   Parameters: {params:,}")

    # Test 5: Full backbone
    print("\n5. Testing LargeKernelBackbone...")
    x_input = torch.randn(2, 1, 64, 128, 128).to(device)
    backbone = LargeKernelBackbone(in_channels=1, base_channels=32).to(device)
    features = backbone(x_input)
    print(f"   Input: {x_input.shape}")
    print("   Output features:")
    for i, feat in enumerate(features):
        print(f"      Stage {i}: {feat.shape}")

    total_params = sum(p.numel() for p in backbone.parameters())
    print(f"   Total backbone parameters: {total_params:,}")

    # Test 6: Adaptive backbone
    print("\n6. Testing LargeKernelBackbone with adaptive kernels...")
    adaptive_backbone = LargeKernelBackbone(
        in_channels=1,
        base_channels=32,
        use_adaptive_kernels=True
    ).to(device)
    features = adaptive_backbone(x_input)
    print(f"   Input: {x_input.shape}")
    print("   Output features:")
    for i, feat in enumerate(features):
        print(f"      Stage {i}: {feat.shape}")

    total_params = sum(p.numel() for p in adaptive_backbone.parameters())
    print(f"   Total backbone parameters: {total_params:,}")

    print("\n" + "=" * 50)
    print("All tests passed!")