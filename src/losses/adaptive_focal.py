import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
import numpy as np


class AdaptiveFocalLoss(nn.Module):
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
        ignore_index: int = -100,
        organ_size_weights: Optional[Dict[str, float]] = None,
        adaptive_gamma: bool = True,
        size_aware_alpha: bool = True
    ):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.adaptive_gamma = adaptive_gamma
        self.size_aware_alpha = size_aware_alpha

        if organ_size_weights is None:
            self.organ_size_weights = {
                'small': 3.0,    # Higher weight for small organs
                'medium': 2.0,   # Medium weight for medium organs
                'large': 1.0     # Standard weight for large organs
            }
        else:
            self.organ_size_weights = organ_size_weights

        # AMOS22 organ size mapping
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

    def _compute_class_frequencies(self, targets: torch.Tensor) -> torch.Tensor:
        """Compute class frequencies for adaptive weighting"""
        unique, counts = torch.unique(targets, return_counts=True)
        total_pixels = targets.numel()

        # Initialize frequency tensor
        num_classes = len(self.organ_size_mapping)
        frequencies = torch.zeros(num_classes, device=targets.device)

        for class_id, count in zip(unique, counts):
            if 0 <= class_id < num_classes:
                frequencies[class_id] = count.float() / total_pixels

        return frequencies

    def _get_adaptive_alpha(self, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
        """Compute adaptive alpha based on organ sizes and class frequencies"""
        if not self.size_aware_alpha:
            if self.alpha is not None:
                return self.alpha
            else:
                return torch.ones(num_classes, device=targets.device)

        frequencies = self._compute_class_frequencies(targets)

        # Initialize alpha with organ size weights
        alpha = torch.ones(num_classes, device=targets.device)

        for class_id, size_category in self.organ_size_mapping.items():
            if class_id < num_classes:
                base_weight = self.organ_size_weights.get(size_category, 1.0)

                # Adjust based on frequency (inverse frequency weighting)
                freq = frequencies[class_id]
                if freq > 0:
                    # Use inverse square root for smoother weighting
                    freq_weight = 1.0 / torch.sqrt(freq + 1e-8)
                    alpha[class_id] = base_weight * freq_weight
                else:
                    alpha[class_id] = base_weight * 10.0  # High weight for missing classes

        # Normalize alpha values to prevent extreme imbalance
        alpha = alpha / alpha.mean()

        return alpha

    def _get_adaptive_gamma(self, targets: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
        """Compute adaptive gamma based on prediction confidence and organ size"""
        if not self.adaptive_gamma:
            return torch.full_like(targets, self.gamma, dtype=torch.float)

        # Get prediction probabilities
        probs = F.softmax(predictions, dim=1)
        predicted_probs = torch.gather(probs, 1, targets.unsqueeze(1).long()).squeeze(1)

        # Base gamma adjustment based on prediction confidence
        confidence_gamma = self.gamma + (1 - predicted_probs) * 2.0

        # Organ-size specific gamma adjustment
        size_gamma = torch.ones_like(targets, dtype=torch.float)
        for class_id, size_category in self.organ_size_mapping.items():
            mask = (targets == class_id)
            if mask.any():
                if size_category == 'small':
                    size_gamma[mask] = 3.0  # Higher focusing for small organs
                elif size_category == 'medium':
                    size_gamma[mask] = 2.5
                else:  # large
                    size_gamma[mask] = 2.0

        # Combine confidence and size-based gamma
        adaptive_gamma = confidence_gamma * size_gamma

        return adaptive_gamma

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            predictions: (N, C, H, W, D) - logits
            targets: (N, H, W, D) - ground truth labels
            class_weights: Optional additional class weights
        """
        # Flatten spatial dimensions
        predictions = predictions.contiguous().view(predictions.size(0), predictions.size(1), -1)
        targets = targets.contiguous().view(targets.size(0), -1)

        # Move to same device
        predictions = predictions.transpose(1, 2).contiguous().view(-1, predictions.size(1))
        targets = targets.view(-1)

        # Create valid mask (ignore index)
        valid_mask = (targets != self.ignore_index)
        if not valid_mask.any():
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        # Apply valid mask
        predictions = predictions[valid_mask]
        targets = targets[valid_mask]

        num_classes = predictions.size(1)

        # Compute adaptive alpha
        alpha = self._get_adaptive_alpha(targets, num_classes)

        # Combine with additional class weights if provided
        if class_weights is not None:
            alpha = alpha * class_weights

        # Compute cross entropy
        ce_loss = F.cross_entropy(predictions, targets.long(), reduction='none')

        # Compute probabilities and pt
        pt = torch.exp(-ce_loss)

        # Get adaptive gamma
        adaptive_gamma = self._get_adaptive_gamma(targets, predictions)

        # Gather alpha for each target
        alpha_t = alpha.gather(0, targets.long())

        # Compute focal loss
        focal_loss = alpha_t * (1 - pt) ** adaptive_gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceAdaptiveFocalLoss(nn.Module):
    """Combined Dice and Adaptive Focal Loss for better small organ segmentation"""

    def __init__(
        self,
        focal_weight: float = 0.7,
        dice_weight: float = 0.3,
        smooth: float = 1e-5,
        **focal_kwargs
    ):
        super().__init__()

        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

        self.focal_loss = AdaptiveFocalLoss(**focal_kwargs)

    def _dice_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute soft Dice loss"""
        # Convert to probabilities
        probs = F.softmax(predictions, dim=1)

        # Convert targets to one-hot
        num_classes = predictions.size(1)
        targets_one_hot = F.one_hot(targets.long(), num_classes).permute(0, 4, 1, 2, 3).float()

        # Compute Dice for each class
        dice_scores = []
        for class_idx in range(num_classes):
            if class_idx == 0:  # Skip background for Dice calculation
                continue

            pred_class = probs[:, class_idx]
            target_class = targets_one_hot[:, class_idx]

            intersection = (pred_class * target_class).sum()
            union = pred_class.sum() + target_class.sum()

            dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice_score)

        # Average Dice across classes (excluding background)
        if dice_scores:
            mean_dice = torch.stack(dice_scores).mean()
            return 1.0 - mean_dice
        else:
            return torch.tensor(0.0, device=predictions.device)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dict containing individual and combined loss values
        """
        focal_loss_val = self.focal_loss(predictions, targets)
        dice_loss_val = self._dice_loss(predictions, targets)

        combined_loss = (
            self.focal_weight * focal_loss_val +
            self.dice_weight * dice_loss_val
        )

        return {
            'total_loss': combined_loss,
            'focal_loss': focal_loss_val,
            'dice_loss': dice_loss_val
        }