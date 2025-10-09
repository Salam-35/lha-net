import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from einops import rearrange


class SpatialAttention3D(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class ChannelAttention3D(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out)


class ScaleSpecificAttention(nn.Module):
    def __init__(self, in_channels: int, scale_factor: float, organ_context: str = "small"):
        super().__init__()
        self.scale_factor = scale_factor
        self.organ_context = organ_context

        self.spatial_attention = SpatialAttention3D(in_channels)
        self.channel_attention = ChannelAttention3D(in_channels)

        self.context_conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True)
        )

        if organ_context == "small":
            self.receptive_field = 3
        elif organ_context == "medium":
            self.receptive_field = 5
        else:  # large
            self.receptive_field = 7

        self.organ_conv = nn.Conv3d(
            in_channels, in_channels,
            self.receptive_field,
            padding=self.receptive_field // 2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale_factor != 1.0:
            x_scaled = F.interpolate(x, scale_factor=self.scale_factor, mode='trilinear', align_corners=False)
        else:
            x_scaled = x

        spatial_att = self.spatial_attention(x_scaled)
        channel_att = self.channel_attention(x_scaled)

        x_att = x_scaled * spatial_att * channel_att

        x_context = self.context_conv(x_att)
        x_organ = self.organ_conv(x_context)

        if self.scale_factor != 1.0:
            x_organ = F.interpolate(x_organ, size=x.shape[2:], mode='trilinear', align_corners=False)

        return x_organ


class PMSAModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        scales: List[float] = [0.5, 0.75, 1.0, 1.25, 1.5],
        organ_contexts: List[str] = ["small", "small", "medium", "medium", "large"],
        reduction_ratio: int = 4
    ):
        super().__init__()

        assert len(scales) == len(organ_contexts), "scales and organ_contexts must have same length"

        self.scales = scales
        self.organ_contexts = organ_contexts
        self.num_scales = len(scales)

        self.scale_attentions = nn.ModuleList([
            ScaleSpecificAttention(in_channels, scale, context)
            for scale, context in zip(scales, organ_contexts)
        ])

        self.progressive_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(in_channels * (i + 1), in_channels, 1),
                nn.BatchNorm3d(in_channels),
                nn.ReLU(inplace=True)
            ) for i in range(1, self.num_scales)
        ])

        self.final_fusion = nn.Sequential(
            nn.Conv3d(in_channels * self.num_scales, in_channels, 1),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.gate_conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction_ratio, self.num_scales, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        batch_size = x.size(0)
        scale_features = []
        progressive_features = []

        # Progressive multi-scale fusion (CORE FEATURE - DO NOT REMOVE)
        # This progressively aggregates information from coarse to fine scales
        for i, (scale_att, scale, context) in enumerate(zip(self.scale_attentions, self.scales, self.organ_contexts)):
            scale_feat = scale_att(x)
            scale_features.append(scale_feat)

            if i == 0:
                progressive_feat = scale_feat
            else:
                # Concatenate ALL scale features seen so far
                # This is the key: progressive fusion aggregates cumulative multi-scale information
                concatenated = torch.cat(scale_features[:i+1], dim=1)
                progressive_feat = self.progressive_fusion[i-1](concatenated)

            progressive_features.append(progressive_feat)

        # Final fusion using all progressive features (which already aggregate multi-scale info)
        # The progressive features are the KEY: they contain cumulative multi-scale information
        all_progressive = torch.cat(progressive_features, dim=1)
        fused_features = self.final_fusion(all_progressive)

        gate_weights = self.gate_conv(fused_features)
        # gate_weights shape: [b, num_scales, h, w, d]
        # Reshape to [b, num_scales, 1, 1, 1] for broadcasting
        if len(gate_weights.shape) == 5:
            gate_weights = gate_weights.mean(dim=[2, 3, 4], keepdim=True)

        # Apply gating to progressive features (not raw scale features)
        # This allows the network to select which level of progressive aggregation is most useful
        weighted_features = []
        for i, feat in enumerate(progressive_features):
            weight = gate_weights[:, i:i+1]
            weighted_features.append(feat * weight)

        final_output = sum(weighted_features)

        return final_output, progressive_features


class HierarchicalPMSA(nn.Module):
    def __init__(
        self,
        channels_list: List[int],
        scales: List[float] = [0.5, 0.75, 1.0, 1.25, 1.5],
        organ_contexts: List[str] = ["small", "small", "medium", "medium", "large"]
    ):
        super().__init__()

        self.pmsa_modules = nn.ModuleDict()
        for i, channels in enumerate(channels_list):
            self.pmsa_modules[f'level_{i}'] = PMSAModule(
                in_channels=channels,
                scales=scales,
                organ_contexts=organ_contexts
            )

        self.cross_scale_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(channels_list[i] + channels_list[i+1], channels_list[i+1], 1),
                nn.BatchNorm3d(channels_list[i+1]),
                nn.ReLU(inplace=True)
            ) for i in range(len(channels_list) - 1)
        ])

    def forward(self, feature_maps: List[torch.Tensor]) -> List[torch.Tensor]:
        pmsa_outputs = []
        scale_features_all = []

        for i, feat_map in enumerate(feature_maps):
            pmsa_out, scale_feats = self.pmsa_modules[f'level_{i}'](feat_map)
            pmsa_outputs.append(pmsa_out)
            scale_features_all.append(scale_feats)

        hierarchical_outputs = []
        for i in range(len(pmsa_outputs)):
            if i == 0:
                hierarchical_outputs.append(pmsa_outputs[i])
            else:
                upsampled_prev = F.interpolate(
                    hierarchical_outputs[i-1],
                    size=pmsa_outputs[i].shape[2:],
                    mode='trilinear',
                    align_corners=False
                )
                fused = torch.cat([upsampled_prev, pmsa_outputs[i]], dim=1)
                fused_output = self.cross_scale_fusion[i-1](fused)
                hierarchical_outputs.append(fused_output)

        return hierarchical_outputs