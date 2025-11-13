"""
Adaptive Progressive Multi-Scale Attention (Adaptive PMSA)
Learnable scale selection for multi-organ segmentation

Key Innovation:
- Learns optimal scale allocation from data
- Eliminates manual hyperparameter tuning
- Adapts to different organ size distributions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
from einops import rearrange


class AdaptiveScaleSelector(nn.Module):
    """
    Learns which scales are most important for the task.

    Architecture:
    1. Maintain a bank of N candidate scales (e.g., 9 scales)
    2. Learn importance weights for each scale
    3. Select top-K scales dynamically during forward pass
    4. Use temperature-based softmax for differentiable selection
    """

    def __init__(
        self,
        scale_bank: List[float] = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5],
        num_active_scales: int = 5,
        temperature: float = 1.0,
        use_gumbel: bool = False
    ):
        super().__init__()

        self.scale_bank = scale_bank
        self.num_active_scales = num_active_scales
        self.temperature = temperature
        self.use_gumbel = use_gumbel

        # Learnable scale importance (raw logits)
        # Initialize with small random values so all scales have similar initial probability
        self.scale_logits = nn.Parameter(
            torch.randn(len(scale_bank)) * 0.1
        )

        # Optional: Add prior initialization to encourage certain scales
        # This helps convergence by starting near a good solution
        self._initialize_with_prior()

    def _initialize_with_prior(self):
        """
        Initialize scale importance with prior knowledge.
        Bias toward scales that worked in previous experiments.
        """
        with torch.no_grad():
            # Prior: Slightly favor scales [0.5, 0.75, 1.0, 1.5, 2.0]
            prior_scales = [0.5, 0.75, 1.0, 1.5, 2.0]
            for i, scale in enumerate(self.scale_bank):
                if scale in prior_scales:
                    self.scale_logits[i] += 0.5  # Small positive bias

    def get_scale_probabilities(self) -> torch.Tensor:
        """Compute probability distribution over scales"""
        return F.softmax(self.scale_logits / self.temperature, dim=0)

    def select_top_k_scales(self, k: Optional[int] = None) -> Tuple[torch.Tensor, List[float]]:
        """
        Select top-k most important scales.

        Returns:
            indices: Tensor of selected scale indices
            scales: List of selected scale values
        """
        if k is None:
            k = self.num_active_scales

        probs = self.get_scale_probabilities()

        # Select top-k indices
        top_k_values, top_k_indices = torch.topk(probs, k=k)

        # Sort to maintain scale order (small to large)
        top_k_indices, _ = torch.sort(top_k_indices)

        # Get actual scale values
        selected_scales = [self.scale_bank[i.item()] for i in top_k_indices]

        return top_k_indices, selected_scales

    def forward(self, training: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass returns scale selection info.

        During training: Use soft selection (weighted combination)
        During inference: Use hard selection (top-k)
        """
        probs = self.get_scale_probabilities()

        if training and self.use_gumbel:
            # Gumbel-Softmax for differentiable discrete selection
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(probs) + 1e-8) + 1e-8)
            logits_with_noise = (self.scale_logits + gumbel_noise) / self.temperature
            soft_selection = F.softmax(logits_with_noise, dim=0)
        else:
            soft_selection = probs

        # Get top-k for actual computation
        top_k_indices, selected_scales = self.select_top_k_scales()

        return {
            'scale_probabilities': probs,
            'soft_selection': soft_selection,
            'top_k_indices': top_k_indices,
            'selected_scales': selected_scales
        }


class AdaptivePMSAModule(nn.Module):
    """
    Adaptive Progressive Multi-Scale Attention Module.

    Key Changes from Original PMSA:
    1. Maintains a bank of N scales (instead of fixed 5)
    2. Learns which scales are most important
    3. Dynamically selects top-K scales during forward pass
    4. Progressive fusion only operates on selected scales

    Architecture preserves:
    - Scale-specific attention mechanisms
    - Progressive fusion strategy
    - Gating mechanism for final combination
    """

    def __init__(
        self,
        in_channels: int,
        scale_bank: List[float] = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5],
        num_active_scales: int = 5,
        reduction_ratio: int = 4,
        use_adaptive_selection: bool = True
    ):
        super().__init__()

        self.in_channels = in_channels
        self.scale_bank = scale_bank
        self.num_active_scales = num_active_scales
        self.use_adaptive_selection = use_adaptive_selection

        # Adaptive scale selector
        if use_adaptive_selection:
            self.scale_selector = AdaptiveScaleSelector(
                scale_bank=scale_bank,
                num_active_scales=num_active_scales
            )

        # Import from original pmsa_module
        from .pmsa_module import ScaleSpecificAttention

        # Create scale-specific attention for ALL scales in bank
        # We'll only use the selected ones during forward pass
        self.scale_attentions = nn.ModuleDict({
            f'scale_{i}': ScaleSpecificAttention(
                in_channels=in_channels,
                scale_factor=scale,
                organ_context=self._assign_organ_context(scale)
            )
            for i, scale in enumerate(scale_bank)
        })

        # Progressive fusion modules (for top-K scales)
        self.progressive_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(in_channels * (i + 1), in_channels, 1),
                nn.BatchNorm3d(in_channels),
                nn.ReLU(inplace=True)
            ) for i in range(1, num_active_scales)
        ])

        # Final fusion
        self.final_fusion = nn.Sequential(
            nn.Conv3d(in_channels * num_active_scales, in_channels, 1),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True)
        )

        # Gating network
        self.gate_conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction_ratio, num_active_scales, 1),
            nn.Sigmoid()
        )

    def _assign_organ_context(self, scale: float) -> str:
        """
        Assign organ context based on scale value.

        Heuristic:
        - scale < 0.8: small organs
        - 0.8 <= scale < 1.3: medium organs
        - scale >= 1.3: large organs
        """
        if scale < 0.8:
            return "small"
        elif scale < 1.3:
            return "medium"
        else:
            return "large"

    def forward(
        self,
        x: torch.Tensor,
        return_scale_info: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with adaptive scale selection.

        Args:
            x: Input feature map [B, C, D, H, W]
            return_scale_info: Whether to return scale selection information

        Returns:
            output: Enhanced feature map [B, C, D, H, W]
            info: Dictionary with scale selection info and intermediate features
        """
        # Get scale selection
        if self.use_adaptive_selection:
            selection_info = self.scale_selector(training=self.training)
            selected_indices = selection_info['top_k_indices']
            selected_scales = selection_info['selected_scales']
        else:
            # Fallback to first K scales if not using adaptive
            selected_indices = torch.arange(self.num_active_scales, device=x.device)
            selected_scales = self.scale_bank[:self.num_active_scales]

        # Process selected scales
        scale_features = []
        progressive_features = []

        for i, scale_idx in enumerate(selected_indices):
            # Get scale-specific attention module
            scale_module = self.scale_attentions[f'scale_{scale_idx.item()}']

            # Process at this scale
            scale_feat = scale_module(x)
            scale_features.append(scale_feat)

            # Progressive fusion
            if i == 0:
                progressive_feat = scale_feat
            else:
                # Concatenate all scale features up to this point
                concatenated = torch.cat(scale_features[:i+1], dim=1)
                progressive_feat = self.progressive_fusion[i-1](concatenated)

            progressive_features.append(progressive_feat)

        # Final fusion of all progressive features
        all_progressive = torch.cat(progressive_features, dim=1)
        fused_features = self.final_fusion(all_progressive)

        # Gating mechanism
        gate_weights = self.gate_conv(fused_features)

        # Average over spatial dimensions for gating
        if len(gate_weights.shape) == 5:
            gate_weights = gate_weights.mean(dim=[2, 3, 4], keepdim=True)

        # Apply gating to progressive features
        weighted_features = []
        for i, feat in enumerate(progressive_features):
            weight = gate_weights[:, i:i+1]
            weighted_features.append(feat * weight)

        final_output = sum(weighted_features)

        # Prepare return info
        info = {
            'scale_features': scale_features,
            'progressive_features': progressive_features,
            'gate_weights': gate_weights,
            'selected_scales': selected_scales,
        }

        if self.use_adaptive_selection and return_scale_info:
            info.update({
                'scale_probabilities': selection_info['scale_probabilities'],
                'scale_indices': selected_indices
            })

        return final_output, info


class AdaptiveHierarchicalPMSA(nn.Module):
    """
    Hierarchical PMSA with adaptive scale selection at each level.

    This replaces the original HierarchicalPMSA but maintains
    the same interface for compatibility with LHA-Net.
    """

    def __init__(
        self,
        channels_list: List[int],
        scale_bank: List[float] = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5],
        num_active_scales: int = 5,
        use_adaptive_selection: bool = True,
        share_scale_selection: bool = False
    ):
        """
        Args:
            channels_list: Number of channels at each encoder level
            scale_bank: Available scales to choose from
            num_active_scales: Number of scales to use
            use_adaptive_selection: Whether to use learnable scale selection
            share_scale_selection: Whether all levels share the same scale selection
        """
        super().__init__()

        self.channels_list = channels_list
        self.num_levels = len(channels_list)
        self.share_scale_selection = share_scale_selection

        # Create PMSA modules for each level
        self.pmsa_modules = nn.ModuleDict()

        if share_scale_selection:
            # All levels share the same scale selector
            shared_selector = AdaptiveScaleSelector(
                scale_bank=scale_bank,
                num_active_scales=num_active_scales
            )

        for i, channels in enumerate(channels_list):
            pmsa = AdaptivePMSAModule(
                in_channels=channels,
                scale_bank=scale_bank,
                num_active_scales=num_active_scales,
                use_adaptive_selection=use_adaptive_selection
            )

            # Share scale selector across levels if requested
            if share_scale_selection and use_adaptive_selection:
                pmsa.scale_selector = shared_selector

            self.pmsa_modules[f'level_{i}'] = pmsa

        # Cross-scale fusion (same as original)
        self.cross_scale_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(channels_list[i] + channels_list[i+1], channels_list[i+1], 1),
                nn.BatchNorm3d(channels_list[i+1]),
                nn.ReLU(inplace=True)
            ) for i in range(len(channels_list) - 1)
        ])

    def forward(
        self,
        feature_maps: List[torch.Tensor],
        return_scale_info: bool = False
    ) -> Tuple[List[torch.Tensor], Dict]:
        """
        Forward pass with hierarchical adaptive PMSA.

        Args:
            feature_maps: List of feature maps from encoder [level_0, level_1, ...]
            return_scale_info: Whether to return scale selection info

        Returns:
            hierarchical_outputs: Enhanced feature maps at each level
            scale_info: Dictionary with scale selection info per level
        """
        pmsa_outputs = []
        scale_info_per_level = {}

        # Apply PMSA at each level
        for i, feat_map in enumerate(feature_maps):
            pmsa_out, info = self.pmsa_modules[f'level_{i}'](
                feat_map,
                return_scale_info=return_scale_info
            )
            pmsa_outputs.append(pmsa_out)

            if return_scale_info:
                scale_info_per_level[f'level_{i}'] = info

        # Hierarchical fusion (same as original)
        hierarchical_outputs = []
        for i in range(len(pmsa_outputs)):
            if i == 0:
                hierarchical_outputs.append(pmsa_outputs[i])
            else:
                # Upsample previous level and fuse with current
                upsampled_prev = F.interpolate(
                    hierarchical_outputs[i-1],
                    size=pmsa_outputs[i].shape[2:],
                    mode='trilinear',
                    align_corners=False
                )
                fused = torch.cat([upsampled_prev, pmsa_outputs[i]], dim=1)
                fused_output = self.cross_scale_fusion[i-1](fused)
                hierarchical_outputs.append(fused_output)

        return hierarchical_outputs, scale_info_per_level

    def get_scale_statistics(self) -> Dict[str, torch.Tensor]:
        """
        Get statistics about learned scale importance across all levels.

        Returns:
            Dictionary with scale probabilities per level
        """
        stats = {}

        for i in range(self.num_levels):
            pmsa = self.pmsa_modules[f'level_{i}']
            if hasattr(pmsa, 'scale_selector'):
                probs = pmsa.scale_selector.get_scale_probabilities()
                stats[f'level_{i}_scale_probs'] = probs

        return stats
