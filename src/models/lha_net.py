import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union

from .pmsa_module import HierarchicalPMSA
from .decoder import OrganSizeAwareDecoder, MultiLevelDecoder
from .backbone import ResNet3DBackbone, LightweightBackbone, resnet18_3d, resnet34_3d


class LHANet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 14,
        backbone_type: str = "resnet18",
        use_lightweight: bool = True,
        pmsa_scales: List[float] = [0.5, 0.75, 1.0, 1.25, 1.5],
        organ_contexts: List[str] = ["small", "small", "medium", "medium", "large"],
        base_channels: int = 32,
        use_deep_supervision: bool = True,
        memory_efficient: bool = True
    ):
        super().__init__()

        self.num_classes = num_classes
        self.use_deep_supervision = use_deep_supervision
        self.memory_efficient = memory_efficient

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

        self.hierarchical_pmsa = HierarchicalPMSA(
            channels_list=backbone_channels[1:],  # Skip initial conv features
            scales=pmsa_scales,
            organ_contexts=organ_contexts
        )

        decoder_channels = [ch // 2 for ch in backbone_channels[1:]]

        self.decoder = OrganSizeAwareDecoder(
            feature_channels=backbone_channels[1:],
            decoder_channels=decoder_channels,
            num_classes=num_classes
        )

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
        return_features: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:

        original_size = x.shape[2:]

        backbone_features = self.backbone(x)

        encoder_features = backbone_features[1:]

        pmsa_features = self.hierarchical_pmsa(encoder_features)

        decoder_output = self.decoder(pmsa_features, target_size=original_size)

        final_prediction = decoder_output['final_output']

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

            if return_features:
                return {
                    'final_prediction': final_prediction,
                    'deep_supervision_outputs': deep_outputs,
                    'size_predictions': decoder_output['size_predictions'],
                    'routing_weights': decoder_output['routing_weights'],
                    'pmsa_features': pmsa_features,
                    'backbone_features': backbone_features
                }
            else:
                return {
                    'final_prediction': final_prediction,
                    'deep_supervision_outputs': deep_outputs
                }

        if return_features:
            return {
                'final_prediction': final_prediction,
                'size_predictions': decoder_output['size_predictions'],
                'routing_weights': decoder_output['routing_weights'],
                'pmsa_features': pmsa_features,
                'backbone_features': backbone_features
            }

        return final_prediction

    def get_model_size(self) -> Dict[str, int]:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        pmsa_params = sum(p.numel() for p in self.hierarchical_pmsa.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'backbone_parameters': backbone_params,
            'pmsa_parameters': pmsa_params,
            'decoder_parameters': decoder_params
        }

    def get_memory_usage(self, input_size: Tuple[int, int, int, int, int]) -> Dict[str, float]:
        device = next(self.parameters()).device
        dummy_input = torch.randn(input_size, device=device)

        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            _ = self(dummy_input)

        memory_stats = {
            'peak_memory_mb': torch.cuda.max_memory_allocated(device) / 1024 / 1024,
            'current_memory_mb': torch.cuda.memory_allocated(device) / 1024 / 1024
        }

        return memory_stats


class LHANetWithAuxiliaryLoss(LHANet):
    def __init__(
        self,
        auxiliary_weight: float = 0.4,
        size_prediction_weight: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.auxiliary_weight = auxiliary_weight
        self.size_prediction_weight = size_prediction_weight

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:

        results = super().forward(x, return_features=True)

        if self.training:
            return {
                'final_prediction': results['final_prediction'],
                'deep_supervision_outputs': results.get('deep_supervision_outputs', []),
                'size_predictions': results['size_predictions'],
                'auxiliary_weight': self.auxiliary_weight,
                'size_prediction_weight': self.size_prediction_weight
            }

        if return_features:
            return results

        return results['final_prediction']


def create_lha_net(
    config_type: str = "lightweight",
    num_classes: int = 14,
    **kwargs
) -> LHANet:

    if config_type == "lightweight":
        return LHANet(
            num_classes=num_classes,
            backbone_type="resnet18",
            use_lightweight=True,
            base_channels=32,
            memory_efficient=True,
            **kwargs
        )
    elif config_type == "standard":
        return LHANet(
            num_classes=num_classes,
            backbone_type="resnet18",
            use_lightweight=False,
            base_channels=64,
            memory_efficient=True,
            **kwargs
        )
    elif config_type == "high_capacity":
        return LHANet(
            num_classes=num_classes,
            backbone_type="resnet34",
            use_lightweight=False,
            base_channels=64,
            memory_efficient=False,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown config type: {config_type}")


def create_lha_net_with_auxiliary(
    config_type: str = "lightweight",
    num_classes: int = 14,
    auxiliary_weight: float = 0.4,
    **kwargs
) -> LHANetWithAuxiliaryLoss:

    base_config = {
        "lightweight": {
            "backbone_type": "resnet18",
            "use_lightweight": True,
            "base_channels": 32,
            "memory_efficient": True
        },
        "standard": {
            "backbone_type": "resnet18",
            "use_lightweight": False,
            "base_channels": 64,
            "memory_efficient": True
        },
        "high_capacity": {
            "backbone_type": "resnet34",
            "use_lightweight": False,
            "base_channels": 64,
            "memory_efficient": False
        }
    }

    if config_type not in base_config:
        raise ValueError(f"Unknown config type: {config_type}")

    config = base_config[config_type]
    config.update(kwargs)

    return LHANetWithAuxiliaryLoss(
        num_classes=num_classes,
        auxiliary_weight=auxiliary_weight,
        **config
    )