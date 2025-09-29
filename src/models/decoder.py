import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from enum import Enum


class OrganSize(Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class OrganSizeClassifier(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 3):
        super().__init__()

        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(in_channels // 4, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_pool = self.global_pool(x).flatten(1)
        return self.classifier(x_pool)


class SizeSpecificDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        organ_size: OrganSize,
        num_classes: int
    ):
        super().__init__()
        self.organ_size = organ_size

        if organ_size == OrganSize.SMALL:
            self.decoder = self._build_small_organ_decoder(in_channels, out_channels, num_classes)
        elif organ_size == OrganSize.MEDIUM:
            self.decoder = self._build_medium_organ_decoder(in_channels, out_channels, num_classes)
        else:  # LARGE
            self.decoder = self._build_large_organ_decoder(in_channels, out_channels, num_classes)

    def _build_small_organ_decoder(self, in_channels: int, out_channels: int, num_classes: int):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_channels, out_channels // 2, 3, padding=1),
            nn.BatchNorm3d(out_channels // 2),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_channels // 2, num_classes, 1)
        )

    def _build_medium_organ_decoder(self, in_channels: int, out_channels: int, num_classes: int):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_channels, out_channels // 2, 3, padding=1),
            nn.BatchNorm3d(out_channels // 2),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_channels // 2, num_classes, 1)
        )

    def _build_large_organ_decoder(self, in_channels: int, out_channels: int, num_classes: int):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels // 2, 3, padding=1),
            nn.BatchNorm3d(out_channels // 2),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_channels // 2, num_classes, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class OrganSizeAwareDecoder(nn.Module):
    def __init__(
        self,
        feature_channels: List[int],
        decoder_channels: List[int],
        num_classes: int,
        organ_size_mapping: Optional[Dict[int, OrganSize]] = None
    ):
        super().__init__()

        self.num_levels = len(feature_channels)
        self.num_classes = num_classes

        if organ_size_mapping is None:
            self.organ_size_mapping = self._get_default_organ_mapping()
        else:
            self.organ_size_mapping = organ_size_mapping

        self.size_classifiers = nn.ModuleList([
            OrganSizeClassifier(channels) for channels in feature_channels
        ])

        self.size_specific_decoders = nn.ModuleDict()
        for level, (in_ch, out_ch) in enumerate(zip(feature_channels, decoder_channels)):
            for size in OrganSize:
                key = f"level_{level}_{size.value}"
                self.size_specific_decoders[key] = SizeSpecificDecoder(
                    in_ch, out_ch, size, num_classes
                )

        self.feature_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(feature_channels[i] + feature_channels[i-1], feature_channels[i], 1),
                nn.BatchNorm3d(feature_channels[i]),
                nn.ReLU(inplace=True)
            ) for i in range(1, len(feature_channels))
        ])

        self.final_classifier = nn.Conv3d(decoder_channels[-1], num_classes, 1)

        self.routing_gate = nn.Sequential(
            nn.Conv3d(feature_channels[-1], feature_channels[-1] // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(feature_channels[-1] // 4, 3, 1),
            nn.Softmax(dim=1)
        )

    def _get_default_organ_mapping(self) -> Dict[int, OrganSize]:
        return {
            1: OrganSize.LARGE,   # liver
            2: OrganSize.LARGE,   # right kidney
            3: OrganSize.LARGE,   # spleen
            4: OrganSize.LARGE,   # pancreas
            5: OrganSize.LARGE,   # aorta
            6: OrganSize.LARGE,   # IVC
            7: OrganSize.MEDIUM,  # right adrenal gland
            8: OrganSize.MEDIUM,  # left adrenal gland
            9: OrganSize.SMALL,   # gallbladder
            10: OrganSize.LARGE,  # esophagus
            11: OrganSize.LARGE,  # stomach
            12: OrganSize.SMALL,  # duodenum
            13: OrganSize.LARGE,  # left kidney
        }

    def forward(
        self,
        features: List[torch.Tensor],
        target_size: Optional[Tuple[int, int, int]] = None
    ) -> Dict[str, torch.Tensor]:

        batch_size = features[0].size(0)

        if target_size is None:
            target_size = features[0].shape[2:]

        size_predictions = []
        for i, feat in enumerate(features):
            size_pred = self.size_classifiers[i](feat)
            size_predictions.append(size_pred)

        fused_features = [features[0]]
        for i in range(1, len(features)):
            upsampled_prev = F.interpolate(
                fused_features[i-1],
                size=features[i].shape[2:],
                mode='trilinear',
                align_corners=False
            )
            concatenated = torch.cat([features[i], upsampled_prev], dim=1)
            fused = self.feature_fusion[i-1](concatenated)
            fused_features.append(fused)

        routing_weights = self.routing_gate(fused_features[-1])

        level_outputs = []
        for level, feat in enumerate(fused_features):
            size_outputs = []
            for size in OrganSize:
                key = f"level_{level}_{size.value}"
                decoder_output = self.size_specific_decoders[key](feat)

                if decoder_output.shape[2:] != target_size:
                    decoder_output = F.interpolate(
                        decoder_output,
                        size=target_size,
                        mode='trilinear',
                        align_corners=False
                    )
                size_outputs.append(decoder_output)

            size_weights = routing_weights if level == len(fused_features) - 1 else \
                          F.interpolate(routing_weights, size=feat.shape[2:], mode='trilinear', align_corners=False)

            weighted_output = (
                size_outputs[0] * size_weights[:, 0:1] +
                size_outputs[1] * size_weights[:, 1:2] +
                size_outputs[2] * size_weights[:, 2:3]
            )
            level_outputs.append(weighted_output)

        final_output = self.final_classifier(fused_features[-1])
        if final_output.shape[2:] != target_size:
            final_output = F.interpolate(
                final_output,
                size=target_size,
                mode='trilinear',
                align_corners=False
            )

        return {
            'final_output': final_output,
            'level_outputs': level_outputs,
            'size_predictions': size_predictions,
            'routing_weights': routing_weights,
            'fused_features': fused_features
        }


class MultiLevelDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels: List[int],
        decoder_channels: List[int],
        num_classes: int,
        use_deep_supervision: bool = True
    ):
        super().__init__()

        self.use_deep_supervision = use_deep_supervision
        self.num_classes = num_classes

        self.decoders = nn.ModuleList()
        for i in range(len(encoder_channels)):
            self.decoders.append(
                OrganSizeAwareDecoder(
                    feature_channels=[encoder_channels[j] for j in range(i+1)],
                    decoder_channels=[decoder_channels[j] for j in range(i+1)],
                    num_classes=num_classes
                )
            )

        if use_deep_supervision:
            self.deep_supervision_heads = nn.ModuleList([
                nn.Conv3d(decoder_channels[i], num_classes, 1)
                for i in range(len(decoder_channels))
            ])

    def forward(self, encoder_features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:

        outputs = []
        deep_outputs = []

        for i, decoder in enumerate(self.decoders):
            decoder_input = encoder_features[:i+1]
            decoder_out = decoder(decoder_input)
            outputs.append(decoder_out)

            if self.use_deep_supervision and i < len(self.deep_supervision_heads):
                deep_out = self.deep_supervision_heads[i](decoder_out['fused_features'][-1])
                deep_outputs.append(deep_out)

        return {
            'outputs': outputs,
            'deep_supervision_outputs': deep_outputs if self.use_deep_supervision else None,
            'final_prediction': outputs[-1]['final_output'] if outputs else None
        }