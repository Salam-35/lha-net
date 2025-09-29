import torch
import torch.nn as nn
from typing import List, Tuple, Optional


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
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


class Bottleneck3D(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet3DBackbone(nn.Module):
    def __init__(
        self,
        block: nn.Module,
        layers: List[int],
        in_channels: int = 1,
        base_channels: int = 64,
        memory_efficient: bool = True
    ):
        super().__init__()

        self.inplanes = base_channels
        self.memory_efficient = memory_efficient

        self.conv1 = nn.Conv3d(
            in_channels, base_channels,
            kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm3d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, base_channels, layers[0])
        self.layer2 = self._make_layer(block, base_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, base_channels * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, base_channels * 8, layers[3], stride=2)

        self._initialize_weights()

    def _make_layer(
        self,
        block: nn.Module,
        planes: int,
        blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features.append(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        features.append(x)

        x = self.layer2(x)
        features.append(x)

        x = self.layer3(x)
        features.append(x)

        x = self.layer4(x)
        features.append(x)

        return features


def resnet18_3d(in_channels: int = 1, **kwargs) -> ResNet3DBackbone:
    return ResNet3DBackbone(BasicBlock3D, [2, 2, 2, 2], in_channels=in_channels, **kwargs)


def resnet34_3d(in_channels: int = 1, **kwargs) -> ResNet3DBackbone:
    return ResNet3DBackbone(BasicBlock3D, [3, 4, 6, 3], in_channels=in_channels, **kwargs)


def resnet50_3d(in_channels: int = 1, **kwargs) -> ResNet3DBackbone:
    return ResNet3DBackbone(Bottleneck3D, [3, 4, 6, 3], in_channels=in_channels, **kwargs)


class LightweightBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        channel_multipliers: List[float] = [1, 2, 4, 8, 16]
    ):
        super().__init__()

        channels = [int(base_channels * mult) for mult in channel_multipliers]

        self.initial_conv = nn.Sequential(
            nn.Conv3d(in_channels, channels[0], 3, padding=1),
            nn.BatchNorm3d(channels[0]),
            nn.ReLU(inplace=True)
        )

        self.encoder_blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            block = nn.Sequential(
                nn.Conv3d(channels[i], channels[i+1], 3, stride=2, padding=1),
                nn.BatchNorm3d(channels[i+1]),
                nn.ReLU(inplace=True),
                nn.Conv3d(channels[i+1], channels[i+1], 3, padding=1),
                nn.BatchNorm3d(channels[i+1]),
                nn.ReLU(inplace=True)
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