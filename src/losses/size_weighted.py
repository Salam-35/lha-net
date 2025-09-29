import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np


class SizeWeightedLoss(nn.Module):
    def __init__(
        self,
        organ_size_mapping: Optional[Dict[int, str]] = None,
        size_weights: Optional[Dict[str, float]] = None,
        volume_adaptive: bool = True,
        spatial_weighting: bool = True,
        boundary_emphasis: float = 2.0
    ):
        super().__init__()

        self.volume_adaptive = volume_adaptive
        self.spatial_weighting = spatial_weighting
        self.boundary_emphasis = boundary_emphasis

        # Default organ size mapping for AMOS22
        if organ_size_mapping is None:
            self.organ_size_mapping = {
                0: 'background',
                1: 'large',     # liver
                2: 'large',     # right kidney
                3: 'large',     # spleen
                4: 'large',     # pancreas
                5: 'large',     # aorta
                6: 'large',     # IVC
                7: 'medium',    # right adrenal gland
                8: 'medium',    # left adrenal gland
                9: 'small',     # gallbladder
                10: 'large',    # esophagus
                11: 'large',    # stomach
                12: 'small',    # duodenum
                13: 'large',    # left kidney
            }
        else:
            self.organ_size_mapping = organ_size_mapping

        # Default size-based weights
        if size_weights is None:
            self.size_weights = {
                'background': 0.1,
                'small': 5.0,
                'medium': 3.0,
                'large': 1.0
            }
        else:
            self.size_weights = size_weights

    def _compute_volume_weights(self, targets: torch.Tensor) -> torch.Tensor:
        """Compute adaptive weights based on actual organ volumes in the batch"""
        batch_size, height, width, depth = targets.shape
        device = targets.device

        # Compute volume for each class in each sample
        volume_weights = torch.ones_like(targets, dtype=torch.float, device=device)

        for b in range(batch_size):
            target_sample = targets[b]
            unique_classes, counts = torch.unique(target_sample, return_counts=True)

            total_voxels = height * width * depth

            for class_id, count in zip(unique_classes, counts):
                if class_id.item() in self.organ_size_mapping:
                    # Volume ratio
                    volume_ratio = count.float() / total_voxels

                    # Inverse volume weighting with size category modulation
                    size_category = self.organ_size_mapping[class_id.item()]
                    base_weight = self.size_weights.get(size_category, 1.0)

                    # Adaptive weight: smaller volumes get higher weights
                    if volume_ratio > 0:
                        adaptive_weight = base_weight / torch.sqrt(volume_ratio + 1e-8)
                    else:
                        adaptive_weight = base_weight * 10.0

                    # Apply weight to all voxels of this class
                    class_mask = (target_sample == class_id)
                    volume_weights[b][class_mask] = adaptive_weight

        return volume_weights

    def _compute_boundary_weights(self, targets: torch.Tensor) -> torch.Tensor:
        """Compute higher weights for boundary voxels"""
        # Apply 3D morphological operations to find boundaries
        kernel_size = 3
        padding = kernel_size // 2

        # Create kernel for erosion
        kernel = torch.ones(1, 1, kernel_size, kernel_size, kernel_size, device=targets.device)

        boundary_weights = torch.ones_like(targets, dtype=torch.float)

        for class_id in torch.unique(targets):
            if class_id == 0:  # Skip background
                continue

            # Create binary mask for current class
            class_mask = (targets == class_id).float().unsqueeze(1)

            # Erode the mask
            eroded = F.conv3d(class_mask, kernel, padding=padding)
            eroded = (eroded == kernel.sum()).float()

            # Boundary is original mask minus eroded mask
            boundary = class_mask.squeeze(1) - eroded.squeeze(1)

            # Apply boundary emphasis
            size_category = self.organ_size_mapping.get(class_id.item(), 'large')
            size_multiplier = self.size_weights.get(size_category, 1.0)

            boundary_weight = 1.0 + self.boundary_emphasis * size_multiplier
            boundary_weights[boundary > 0] = boundary_weight

        return boundary_weights

    def _compute_spatial_weights(self, targets: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
        """Compute spatial weights based on prediction uncertainty and organ context"""
        # Compute prediction entropy as uncertainty measure
        probs = F.softmax(predictions, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

        # Normalize entropy to [0, 1]
        max_entropy = torch.log(torch.tensor(predictions.size(1), dtype=torch.float))
        normalized_entropy = entropy / max_entropy

        # Higher weights for uncertain regions
        uncertainty_weights = 1.0 + normalized_entropy

        # Combine with organ size weights
        size_weights = torch.ones_like(targets, dtype=torch.float)
        for class_id in torch.unique(targets):
            class_mask = (targets == class_id)
            size_category = self.organ_size_mapping.get(class_id.item(), 'large')
            size_weight = self.size_weights.get(size_category, 1.0)
            size_weights[class_mask] = size_weight

        return uncertainty_weights * size_weights

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: (N, C, H, W, D) - model predictions (logits)
            targets: (N, H, W, D) - ground truth labels

        Returns:
            Dictionary with loss components
        """
        device = predictions.device
        batch_size = targets.size(0)

        # Initialize weight tensor
        total_weights = torch.ones_like(targets, dtype=torch.float, device=device)

        # 1. Volume-adaptive weights
        if self.volume_adaptive:
            volume_weights = self._compute_volume_weights(targets)
            total_weights *= volume_weights

        # 2. Spatial uncertainty weights
        if self.spatial_weighting:
            spatial_weights = self._compute_spatial_weights(targets, predictions)
            total_weights *= spatial_weights

        # 3. Boundary emphasis weights
        boundary_weights = self._compute_boundary_weights(targets)
        total_weights *= boundary_weights

        # Flatten for loss computation
        predictions_flat = predictions.permute(0, 2, 3, 4, 1).contiguous().view(-1, predictions.size(1))
        targets_flat = targets.view(-1)
        weights_flat = total_weights.view(-1)

        # Compute weighted cross-entropy loss
        ce_loss = F.cross_entropy(predictions_flat, targets_flat.long(), reduction='none')
        weighted_ce_loss = (ce_loss * weights_flat).mean()

        # Compute individual loss components for monitoring
        volume_weights_only = self._compute_volume_weights(targets) if self.volume_adaptive else torch.ones_like(targets)
        spatial_weights_only = self._compute_spatial_weights(targets, predictions) if self.spatial_weighting else torch.ones_like(targets)

        volume_loss = (ce_loss * volume_weights_only.view(-1)).mean()
        spatial_loss = (ce_loss * spatial_weights_only.view(-1)).mean()
        boundary_loss = (ce_loss * boundary_weights.view(-1)).mean()

        return {
            'total_loss': weighted_ce_loss,
            'volume_weighted_loss': volume_loss,
            'spatial_weighted_loss': spatial_loss,
            'boundary_weighted_loss': boundary_loss,
            'base_ce_loss': ce_loss.mean()
        }


class MultiScaleSizeWeightedLoss(nn.Module):
    """Multi-scale size-weighted loss for hierarchical supervision"""

    def __init__(
        self,
        scale_weights: List[float] = [1.0, 0.8, 0.6, 0.4],
        **size_weighted_kwargs
    ):
        super().__init__()

        self.scale_weights = scale_weights
        self.size_weighted_loss = SizeWeightedLoss(**size_weighted_kwargs)

    def forward(
        self,
        multi_scale_predictions: List[torch.Tensor],
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            multi_scale_predictions: List of predictions at different scales
            targets: Ground truth labels

        Returns:
            Dictionary with multi-scale loss components
        """
        total_loss = 0.0
        scale_losses = []

        original_size = targets.shape[2:]

        for i, (predictions, weight) in enumerate(zip(multi_scale_predictions, self.scale_weights)):
            # Resize predictions to match target size if needed
            if predictions.shape[2:] != original_size:
                predictions = F.interpolate(
                    predictions,
                    size=original_size,
                    mode='trilinear',
                    align_corners=False
                )

            # Compute size-weighted loss for this scale
            scale_loss_dict = self.size_weighted_loss(predictions, targets)
            scale_loss = scale_loss_dict['total_loss']

            # Weight the loss
            weighted_scale_loss = weight * scale_loss
            total_loss += weighted_scale_loss

            scale_losses.append({
                'scale': i,
                'weight': weight,
                'loss': scale_loss,
                'weighted_loss': weighted_scale_loss
            })

        return {
            'total_loss': total_loss,
            'scale_losses': scale_losses,
            'num_scales': len(multi_scale_predictions)
        }