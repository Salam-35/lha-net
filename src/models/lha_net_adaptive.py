"""
LHA-Net with Adaptive PMSA

Modified version of LHA-Net that uses learnable scale selection
instead of hardcoded scales.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union

from .adaptive_pmsa import AdaptiveHierarchicalPMSA
from .decoder import OrganSizeAwareDecoder
from .backbone import LightweightBackbone, resnet18_3d, resnet34_3d


class LHANetAdaptive(nn.Module):
    """
    LHA-Net with Adaptive PMSA.

    Key Difference from Original:
    - Uses AdaptiveHierarchicalPMSA instead of HierarchicalPMSA
    - Learns optimal scales during training
    - No need for manual scale configuration
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 16,
        backbone_type: str = "resnet18",
        use_lightweight: bool = True,
        scale_bank: List[float] = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5],
        num_active_scales: int = 5,
        base_channels: int = 32,
        use_deep_supervision: bool = True,
        memory_efficient: bool = True,
        use_adaptive_pmsa: bool = True,
        share_scale_selection: bool = False
    ):
        super().__init__()

        self.num_classes = num_classes
        self.use_deep_supervision = use_deep_supervision
        self.memory_efficient = memory_efficient
        self.use_adaptive_pmsa = use_adaptive_pmsa

        # Backbone (same as original)
        if use_lightweight:
            self.backbone = LightweightBackbone(
                in_channels=in_channels,
                base_channels=base_channels,
                channel_multipliers=[1, 2, 4, 8, 16]
            )
            backbone_channels = [int(base_channels * mult) for mult in [1, 2, 4, 8, 16]]
        else:
            if backbone_type == "resnet18":
                self.backbone = resnet18_3d(in_channels=in_channels, base_channels=base_channels)
            elif backbone_type == "resnet34":
                self.backbone = resnet34_3d(in_channels=in_channels, base_channels=base_channels)
            else:
                raise ValueError(f"Unsupported backbone type: {backbone_type}")

            backbone_channels = [base_channels, base_channels, base_channels*2, base_channels*4, base_channels*8]

        # Adaptive Hierarchical PMSA
        self.hierarchical_pmsa = AdaptiveHierarchicalPMSA(
            channels_list=backbone_channels[1:],  # Skip initial conv features
            scale_bank=scale_bank,
            num_active_scales=num_active_scales,
            use_adaptive_selection=use_adaptive_pmsa,
            share_scale_selection=share_scale_selection
        )

        # Decoder (same as original)
        decoder_channels = [ch // 2 for ch in backbone_channels[1:]]

        self.decoder = OrganSizeAwareDecoder(
            feature_channels=backbone_channels[1:],
            decoder_channels=decoder_channels,
            num_classes=num_classes
        )

        # Deep supervision heads (same as original)
        if use_deep_supervision:
            self.deep_supervision_heads = nn.ModuleList([
                nn.Conv3d(ch, num_classes, 1) for ch in backbone_channels[1:]
            ])

        self.final_conv = nn.Conv3d(decoder_channels[-1], num_classes, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        return_scale_info: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input image [B, C, D, H, W]
            return_features: Whether to return intermediate features
            return_scale_info: Whether to return scale selection info

        Returns:
            If return_features=False: segmentation prediction
            If return_features=True: dict with all intermediate outputs
        """
        original_size = x.shape[2:]

        # Backbone
        backbone_features = self.backbone(x)
        encoder_features = backbone_features[1:]

        # Adaptive Hierarchical PMSA
        pmsa_features, scale_info = self.hierarchical_pmsa(
            encoder_features,
            return_scale_info=return_scale_info
        )

        # Decoder
        decoder_output = self.decoder(pmsa_features, target_size=original_size)
        final_prediction = decoder_output['final_output']

        # Deep supervision (training only)
        if self.training and self.use_deep_supervision:
            deep_outputs = []
            for i, (head, feat) in enumerate(zip(self.deep_supervision_heads, pmsa_features)):
                deep_out = head(feat)
                if deep_out.shape[2:] != original_size:
                    deep_out = F.interpolate(
                        deep_out,
                        size=original_size,
                        mode='trilinear',
                        align_corners=False
                    )
                deep_outputs.append(deep_out)

            if return_features or return_scale_info:
                output_dict = {
                    'final_prediction': final_prediction,
                    'deep_supervision_outputs': deep_outputs,
                    'size_predictions': decoder_output['size_predictions'],
                    'routing_weights': decoder_output['routing_weights'],
                    'pmsa_features': pmsa_features,
                    'backbone_features': backbone_features
                }

                if return_scale_info:
                    output_dict['scale_info'] = scale_info

                return output_dict
            else:
                return {
                    'final_prediction': final_prediction,
                    'deep_supervision_outputs': deep_outputs
                }

        # Inference mode
        if return_features or return_scale_info:
            output_dict = {
                'final_prediction': final_prediction,
                'size_predictions': decoder_output['size_predictions'],
                'routing_weights': decoder_output['routing_weights'],
                'pmsa_features': pmsa_features,
                'backbone_features': backbone_features
            }

            if return_scale_info:
                output_dict['scale_info'] = scale_info

            return output_dict

        return final_prediction

    def get_scale_statistics(self) -> Dict[str, torch.Tensor]:
        """Get learned scale importance at each level"""
        return self.hierarchical_pmsa.get_scale_statistics()


def create_lha_net_adaptive(
    config_type: str = "lightweight",
    num_classes: int = 16,
    **kwargs
) -> LHANetAdaptive:
    """Factory function to create LHA-Net with Adaptive PMSA"""

    # Extract adaptive_pmsa flag if provided in kwargs
    use_adaptive_pmsa = kwargs.pop('use_adaptive_pmsa', True)

    if config_type == "lightweight":
        return LHANetAdaptive(
            num_classes=num_classes,
            backbone_type="resnet18",
            use_lightweight=True,
            base_channels=32,
            memory_efficient=True,
            use_adaptive_pmsa=use_adaptive_pmsa,
            **kwargs
        )
    elif config_type == "standard":
        return LHANetAdaptive(
            num_classes=num_classes,
            backbone_type="resnet18",
            use_lightweight=False,
            base_channels=64,
            memory_efficient=True,
            use_adaptive_pmsa=use_adaptive_pmsa,
            **kwargs
        )
    elif config_type == "high_capacity":
        return LHANetAdaptive(
            num_classes=num_classes,
            backbone_type="resnet34",
            use_lightweight=False,
            base_channels=64,
            memory_efficient=False,
            use_adaptive_pmsa=use_adaptive_pmsa,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown config type: {config_type}")
