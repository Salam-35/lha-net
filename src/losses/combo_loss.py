import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union

from .adaptive_focal import AdaptiveFocalLoss, DiceAdaptiveFocalLoss
from .size_weighted import SizeWeightedLoss, MultiScaleSizeWeightedLoss


class ComboLoss(nn.Module):
    """Combined loss function integrating multiple loss components for LHA-Net"""

    def __init__(
        self,
        focal_weight: float = 0.5,
        dice_weight: float = 0.3,
        size_weight: float = 0.2,
        boundary_weight: float = 0.1,
        use_adaptive_focal: bool = True,
        use_size_weighting: bool = True,
        **kwargs
    ):
        super().__init__()

        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.size_weight = size_weight
        self.boundary_weight = boundary_weight

        # Initialize loss components
        if use_adaptive_focal:
            self.focal_loss = DiceAdaptiveFocalLoss(
                focal_weight=0.7,
                dice_weight=0.3,
                **kwargs.get('focal_kwargs', {})
            )
        else:
            self.focal_loss = nn.CrossEntropyLoss(reduction='mean')

        if use_size_weighting:
            self.size_weighted_loss = SizeWeightedLoss(
                **kwargs.get('size_kwargs', {})
            )
        else:
            self.size_weighted_loss = None

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: Model predictions (logits)
            targets: Ground truth labels

        Returns:
            Dictionary containing loss components and total loss
        """
        losses = {}

        # 1. Focal loss (includes Dice if using DiceAdaptiveFocalLoss)
        if isinstance(self.focal_loss, DiceAdaptiveFocalLoss):
            focal_result = self.focal_loss(predictions, targets)
            losses['focal_loss'] = focal_result['focal_loss']
            losses['dice_loss'] = focal_result['dice_loss']
            focal_contribution = self.focal_weight * focal_result['total_loss']
        else:
            focal_loss_val = self.focal_loss(predictions, targets)
            losses['focal_loss'] = focal_loss_val
            focal_contribution = self.focal_weight * focal_loss_val

        # 2. Size-weighted loss
        if self.size_weighted_loss is not None:
            size_result = self.size_weighted_loss(predictions, targets)
            losses['size_weighted_loss'] = size_result['total_loss']
            losses['volume_loss'] = size_result['volume_weighted_loss']
            losses['boundary_loss'] = size_result['boundary_weighted_loss']
            size_contribution = self.size_weight * size_result['total_loss']
        else:
            size_contribution = 0.0

        # Combine losses
        total_loss = focal_contribution + size_contribution

        losses['total_loss'] = total_loss
        losses['focal_contribution'] = focal_contribution
        losses['size_contribution'] = size_contribution

        return losses


class LHANetLoss(nn.Module):
    """Complete loss function for LHA-Net with deep supervision and auxiliary losses"""

    def __init__(
        self,
        primary_loss_weight: float = 1.0,
        deep_supervision_weight: float = 0.4,
        size_prediction_weight: float = 0.1,
        routing_weight: float = 0.05,
        deep_supervision_weights: Optional[List[float]] = None,
        **combo_loss_kwargs
    ):
        super().__init__()

        self.primary_loss_weight = primary_loss_weight
        self.deep_supervision_weight = deep_supervision_weight
        self.size_prediction_weight = size_prediction_weight
        self.routing_weight = routing_weight

        # Deep supervision weights (from deep to shallow)
        if deep_supervision_weights is None:
            self.deep_supervision_weights = [1.0, 0.8, 0.6, 0.4]
        else:
            self.deep_supervision_weights = deep_supervision_weights

        # Primary loss function
        self.combo_loss = ComboLoss(**combo_loss_kwargs)

        # Size prediction loss (classification)
        self.size_prediction_loss = nn.CrossEntropyLoss()

        # Routing consistency loss
        self.routing_loss = nn.MSELoss()

    def _compute_deep_supervision_loss(
        self,
        deep_outputs: List[torch.Tensor],
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute deep supervision losses"""
        deep_losses = []
        total_deep_loss = 0.0

        for i, (output, weight) in enumerate(zip(deep_outputs, self.deep_supervision_weights)):
            # Resize output to match target size if needed
            if output.shape[2:] != targets.shape[2:]:
                output = torch.nn.functional.interpolate(
                    output,
                    size=targets.shape[2:],
                    mode='trilinear',
                    align_corners=False
                )

            # Compute loss for this deep supervision level
            deep_loss_dict = self.combo_loss(output, targets)
            weighted_loss = weight * deep_loss_dict['total_loss']

            deep_losses.append({
                'level': i,
                'loss': deep_loss_dict['total_loss'],
                'weighted_loss': weighted_loss
            })

            total_deep_loss += weighted_loss

        return {
            'total_deep_loss': total_deep_loss,
            'deep_losses': deep_losses
        }

    def _compute_size_prediction_loss(
        self,
        size_predictions: List[torch.Tensor],
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute auxiliary size prediction loss"""
        # Create size targets based on organ presence in the sample
        batch_size = targets.size(0)
        device = targets.device

        # Define organ size mapping
        organ_size_map = {
            0: 0,  # background -> background
            1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 10: 2, 11: 2, 13: 2,  # large -> 2
            7: 1, 8: 1,  # medium -> 1
            9: 0, 12: 0  # small -> 0 (but we'll use label 0 for small, 1 for medium, 2 for large)
        }

        # Correct mapping: 0=small, 1=medium, 2=large
        size_targets = torch.zeros(batch_size, device=device, dtype=torch.long)

        for b in range(batch_size):
            unique_classes = torch.unique(targets[b])
            has_small = any(cls.item() in [9, 12] for cls in unique_classes if cls > 0)
            has_medium = any(cls.item() in [7, 8] for cls in unique_classes if cls > 0)
            has_large = any(cls.item() in [1, 2, 3, 4, 5, 6, 10, 11, 13] for cls in unique_classes if cls > 0)

            if has_small:
                size_targets[b] = 0  # small
            elif has_medium:
                size_targets[b] = 1  # medium
            elif has_large:
                size_targets[b] = 2  # large

        # Compute loss for each level's size predictions
        total_size_loss = 0.0
        for size_pred in size_predictions:
            size_loss = self.size_prediction_loss(size_pred, size_targets)
            total_size_loss += size_loss

        return total_size_loss / len(size_predictions)

    def _compute_routing_consistency_loss(
        self,
        routing_weights: torch.Tensor,
        size_predictions: List[torch.Tensor]
    ) -> torch.Tensor:
        """Compute routing consistency loss"""
        # Average size predictions across levels
        avg_size_pred = torch.stack(size_predictions).mean(dim=0)
        size_probs = torch.softmax(avg_size_pred, dim=1)

        # Expected routing weights should align with size predictions
        expected_routing = size_probs

        # Compute MSE between actual and expected routing weights
        routing_consistency_loss = self.routing_loss(
            routing_weights.mean(dim=[2, 3, 4]),  # Average over spatial dimensions
            expected_routing
        )

        return routing_consistency_loss

    def forward(
        self,
        model_output: Union[torch.Tensor, Dict[str, torch.Tensor]],
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            model_output: Either final prediction tensor or dict with multiple outputs
            targets: Ground truth labels

        Returns:
            Dictionary with all loss components
        """
        if isinstance(model_output, torch.Tensor):
            # Simple case: only final prediction
            primary_loss_dict = self.combo_loss(model_output, targets)
            return {
                'total_loss': primary_loss_dict['total_loss'],
                **primary_loss_dict
            }

        # Complex case: multiple outputs from LHA-Net
        final_prediction = model_output['final_prediction']
        losses = {}

        # 1. Primary segmentation loss
        primary_loss_dict = self.combo_loss(final_prediction, targets)
        primary_loss = self.primary_loss_weight * primary_loss_dict['total_loss']
        losses.update({f'primary_{k}': v for k, v in primary_loss_dict.items()})

        total_loss = primary_loss

        # 2. Deep supervision loss
        if 'deep_supervision_outputs' in model_output and model_output['deep_supervision_outputs']:
            deep_loss_dict = self._compute_deep_supervision_loss(
                model_output['deep_supervision_outputs'],
                targets
            )
            deep_loss = self.deep_supervision_weight * deep_loss_dict['total_deep_loss']
            losses['deep_supervision_loss'] = deep_loss_dict['total_deep_loss']
            total_loss += deep_loss

        # 3. Size prediction auxiliary loss
        if 'size_predictions' in model_output:
            size_loss = self._compute_size_prediction_loss(
                model_output['size_predictions'],
                targets
            )
            weighted_size_loss = self.size_prediction_weight * size_loss
            losses['size_prediction_loss'] = size_loss
            total_loss += weighted_size_loss

        # 4. Routing consistency loss
        if 'routing_weights' in model_output and 'size_predictions' in model_output:
            routing_loss = self._compute_routing_consistency_loss(
                model_output['routing_weights'],
                model_output['size_predictions']
            )
            weighted_routing_loss = self.routing_weight * routing_loss
            losses['routing_consistency_loss'] = routing_loss
            total_loss += weighted_routing_loss

        losses['total_loss'] = total_loss
        losses['primary_contribution'] = primary_loss

        return losses