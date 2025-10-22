import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from scipy import ndimage
import warnings


def compute_dice_score(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    smooth: float = 1e-8,
    ignore_background: bool = True
) -> Union[float, np.ndarray]:
    """
    Compute Dice coefficient

    Args:
        pred: Predicted segmentation (H, W, D) or (C, H, W, D) - can be class indices or one-hot
        target: Ground truth segmentation (H, W, D) or (C, H, W, D) - can be class indices or one-hot
        smooth: Smoothing factor to avoid division by zero
        ignore_background: Whether to ignore background class (class 0)

    Returns:
        Dice score(s) - always in range [0, 1]
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    # Ensure same shape
    assert pred.shape == target.shape, f"Shape mismatch: {pred.shape} vs {target.shape}"

    # If multi-class, compute per-class Dice and return per-class scores
    if len(pred.shape) == 4:  # (C, H, W, D) - one-hot encoded
        dice_scores = []
        start_idx = 1 if ignore_background else 0

        for c in range(start_idx, pred.shape[0]):
            pred_c = pred[c]
            target_c = target[c]

            intersection = np.sum(pred_c * target_c)
            union = np.sum(pred_c) + np.sum(target_c)

            if union == 0:
                # Both pred and target are empty for this class
                dice = 1.0 if intersection == 0 else 0.0
            else:
                dice = (2.0 * intersection + smooth) / (union + smooth)

            # Clamp to [0, 1] range
            dice = np.clip(dice, 0.0, 1.0)
            dice_scores.append(dice)

        return np.array(dice_scores, dtype=np.float32)

    else:  # Single class (H, W, D) - binary mask or class indices
        # If it looks like class indices (integer values), binarize it
        if pred.dtype in [np.int32, np.int64, np.uint8] and len(np.unique(pred)) > 2:
            # This is class indices, not binary mask - shouldn't use this function directly
            warnings.warn("compute_dice_score received class indices instead of binary mask. Converting to binary.")
            pred = (pred > 0).astype(np.float32)
            target = (target > 0).astype(np.float32)

        intersection = np.sum(pred * target)
        union = np.sum(pred) + np.sum(target)

        if union == 0:
            dice = 1.0 if intersection == 0 else 0.0
        else:
            dice = (2.0 * intersection + smooth) / (union + smooth)

        # Clamp to [0, 1] range
        return np.clip(dice, 0.0, 1.0)


def _binary_surface(mask: np.ndarray) -> np.ndarray:
    """Return binary surface mask (1 where on surface)."""
    structure = ndimage.generate_binary_structure(3, 2)
    eroded = ndimage.binary_erosion(mask, structure, iterations=1, border_value=0)
    surface = mask ^ eroded
    return surface.astype(np.uint8)


def _surface_distances_mm(a: np.ndarray, b: np.ndarray, spacing: Tuple[float, float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute distances from surface(a) to surface(b) and vice versa using EDT (in mm)."""
    a = (a > 0).astype(np.uint8)
    b = (b > 0).astype(np.uint8)

    surf_a = _binary_surface(a)
    surf_b = _binary_surface(b)

    # If surfaces are empty, return empty arrays
    if surf_a.sum() == 0 and surf_b.sum() == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    # Distance transform to the nearest surface voxel
    # EDT computes distance to the nearest zero; use inverse surface mask so that surface voxels are zeros
    dt_b = ndimage.distance_transform_edt(1 - surf_b, sampling=spacing)
    dt_a = ndimage.distance_transform_edt(1 - surf_a, sampling=spacing)

    dists_a_to_b = dt_b[surf_a.astype(bool)] if surf_a.sum() > 0 else np.array([], dtype=np.float32)
    dists_b_to_a = dt_a[surf_b.astype(bool)] if surf_b.sum() > 0 else np.array([], dtype=np.float32)

    return dists_a_to_b.astype(np.float32), dists_b_to_a.astype(np.float32)


def compute_hausdorff_distance(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    percentile: float = 95.0
) -> float:
    """
    Robust Hausdorff distance (percentile) using distance transforms (fast, in mm).
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    pred = (pred > 0.5).astype(np.uint8)
    target = (target > 0.5).astype(np.uint8)

    vol_diag_mm = float(np.linalg.norm(np.array(pred.shape) * np.array(spacing)))

    if pred.sum() == 0 and target.sum() == 0:
        return 0.0
    if pred.sum() == 0 or target.sum() == 0:
        return vol_diag_mm

    try:
        d1, d2 = _surface_distances_mm(pred, target, spacing)
        if d1.size == 0 and d2.size == 0:
            return vol_diag_mm
        all_d = np.concatenate([d1, d2]) if d1.size and d2.size else (d1 if d1.size else d2)
        if all_d.size == 0:
            return vol_diag_mm
        if percentile >= 100.0:
            return float(np.max(all_d))
        return float(np.percentile(all_d, percentile))
    except Exception as e:
        warnings.warn(f"HD95 computation failed: {e}")
        return vol_diag_mm


def _extract_surface_points(mask: np.ndarray, spacing: Tuple[float, float, float]) -> List[np.ndarray]:
    """Extract surface points from binary mask"""
    # Find edges using morphological operations
    structure = ndimage.generate_binary_structure(3, 2)
    eroded = ndimage.binary_erosion(mask, structure)
    surface = mask ^ eroded

    # Get coordinates of surface points
    coords = np.where(surface)
    if len(coords[0]) == 0:
        return []

    # Apply spacing
    surface_points = []
    for i in range(len(coords[0])):
        point = np.array([
            coords[0][i] * spacing[0],
            coords[1][i] * spacing[1],
            coords[2][i] * spacing[2]
        ])
        surface_points.append(point)

    return surface_points


def compute_normalized_surface_distance(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    tolerance: float = 1.0
) -> float:
    """Normalized Surface Dice using distance transforms (fast)."""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    pred = (pred > 0.5).astype(np.uint8)
    target = (target > 0.5).astype(np.uint8)

    if pred.sum() == 0 and target.sum() == 0:
        return 1.0
    if pred.sum() == 0 or target.sum() == 0:
        return 0.0

    try:
        d1, d2 = _surface_distances_mm(pred, target, spacing)
        n1 = (d1 <= tolerance).sum() if d1.size else 0
        n2 = (d2 <= tolerance).sum() if d2.size else 0
        total = (d1.size if d1 is not None else 0) + (d2.size if d2 is not None else 0)
        if total == 0:
            return 0.0
        return float((n1 + n2) / total)
    except Exception as e:
        warnings.warn(f"NSD computation failed: {e}")
        return 0.0


class SegmentationMetrics:
    """Comprehensive segmentation metrics calculator"""

    def __init__(
        self,
        num_classes: int = 14,
        spacing: Tuple[float, float, float] = (1.5, 1.5, 1.5),
        organ_names: Optional[List[str]] = None
    ):
        self.num_classes = num_classes
        self.spacing = spacing

        if organ_names is None:
            self.organ_names = [
                'background', 'liver', 'right_kidney', 'spleen', 'pancreas',
                'aorta', 'ivc', 'right_adrenal', 'left_adrenal', 'gallbladder',
                'esophagus', 'stomach', 'duodenum', 'left_kidney'
            ]
        else:
            self.organ_names = organ_names

        # Organ size categories for AMOS22
        self.small_organs = [9, 12]  # gallbladder, duodenum
        self.medium_organs = [7, 8]  # adrenal glands
        self.large_organs = [1, 2, 3, 4, 5, 6, 10, 11, 13]  # rest

    def compute_all_metrics(
        self,
        predictions: Union[torch.Tensor, np.ndarray],
        targets: Union[torch.Tensor, np.ndarray],
        case_id: Optional[str] = None
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Compute all segmentation metrics

        Args:
            predictions: Model predictions (C, H, W, D) or (H, W, D)
            targets: Ground truth labels (C, H, W, D) or (H, W, D)
            case_id: Optional case identifier

        Returns:
            Dictionary with all computed metrics
        """
        # Convert to numpy if needed
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        # Handle different input formats
        if len(predictions.shape) == 3:  # (H, W, D) - class indices
            pred_one_hot = self._to_one_hot(predictions, self.num_classes)
            target_one_hot = self._to_one_hot(targets, self.num_classes)
        elif len(predictions.shape) == 4:  # (C, H, W, D) - one-hot or probabilities
            if predictions.max() <= 1.0:  # Probabilities
                pred_indices = np.argmax(predictions, axis=0)
                pred_one_hot = self._to_one_hot(pred_indices, self.num_classes)
            else:  # Already one-hot
                pred_one_hot = predictions

            if targets.max() <= 1.0 and len(np.unique(targets)) > 2:  # Probabilities
                target_indices = np.argmax(targets, axis=0)
                target_one_hot = self._to_one_hot(target_indices, self.num_classes)
            elif len(targets.shape) == 3:  # Class indices
                target_one_hot = self._to_one_hot(targets, self.num_classes)
            else:  # Already one-hot
                target_one_hot = targets
        else:
            raise ValueError(f"Unsupported prediction shape: {predictions.shape}")

        metrics = {}

        # 1. Dice scores
        dice_scores = self._compute_dice_per_class(pred_one_hot, target_one_hot)
        metrics['dice_scores'] = {
            self.organ_names[i]: dice_scores[i] for i in range(len(dice_scores))
        }
        metrics['mean_dice'] = np.mean(dice_scores[1:])  # Exclude background

        # 2. Dice by organ size categories
        small_dice = [dice_scores[i] for i in self.small_organs if i < len(dice_scores)]
        medium_dice = [dice_scores[i] for i in self.medium_organs if i < len(dice_scores)]
        large_dice = [dice_scores[i] for i in self.large_organs if i < len(dice_scores)]

        metrics['small_organ_dice'] = np.mean(small_dice) if small_dice else 0.0
        metrics['medium_organ_dice'] = np.mean(medium_dice) if medium_dice else 0.0
        metrics['large_organ_dice'] = np.mean(large_dice) if large_dice else 0.0

        # 3. Hausdorff distances
        hd95_scores = {}
        for class_idx in range(1, self.num_classes):  # Skip background
            if class_idx < len(self.organ_names):
                try:
                    hd95 = compute_hausdorff_distance(
                        pred_one_hot[class_idx],
                        target_one_hot[class_idx],
                        self.spacing,
                        percentile=95.0
                    )
                    hd95_scores[self.organ_names[class_idx]] = hd95
                except Exception as e:
                    hd95_scores[self.organ_names[class_idx]] = np.nan

        metrics['hd95_scores'] = hd95_scores
        valid_hd95 = [v for v in hd95_scores.values() if not np.isnan(v) and not np.isinf(v)]
        metrics['mean_hd95'] = np.mean(valid_hd95) if valid_hd95 else np.nan

        # 4. Normalized Surface Distance
        nsd_scores = {}
        for class_idx in range(1, self.num_classes):
            if class_idx < len(self.organ_names):
                try:
                    nsd = compute_normalized_surface_distance(
                        pred_one_hot[class_idx],
                        target_one_hot[class_idx],
                        self.spacing,
                        tolerance=1.0
                    )
                    nsd_scores[self.organ_names[class_idx]] = nsd
                except Exception:
                    nsd_scores[self.organ_names[class_idx]] = np.nan

        metrics['nsd_scores'] = nsd_scores
        valid_nsd = [v for v in nsd_scores.values() if not np.isnan(v)]
        metrics['mean_nsd'] = np.mean(valid_nsd) if valid_nsd else np.nan

        # 5. Volume metrics
        volume_metrics = self._compute_volume_metrics(pred_one_hot, target_one_hot)
        metrics['volume_metrics'] = volume_metrics

        # 6. Volume similarity scores
        volume_similarity_scores = {}
        for class_idx in range(1, self.num_classes):
            if class_idx < len(self.organ_names):
                organ_name = self.organ_names[class_idx]
                if organ_name in volume_metrics:
                    rve = volume_metrics[organ_name]['relative_volume_error']
                    # Volume similarity: 1 - relative_volume_error (clamped to [0,1])
                    vs = max(0.0, 1.0 - rve)
                    volume_similarity_scores[organ_name] = vs
                else:
                    volume_similarity_scores[organ_name] = np.nan

        metrics['volume_similarity_scores'] = volume_similarity_scores

        # 7. Summary metrics
        metrics['case_id'] = case_id
        metrics['median_dice'] = np.median([dice_scores[i] for i in range(1, len(dice_scores))])

        return metrics

    def _to_one_hot(self, labels: np.ndarray, num_classes: int) -> np.ndarray:
        """Convert class indices to one-hot encoding"""
        one_hot = np.zeros((num_classes,) + labels.shape, dtype=np.float32)
        for c in range(num_classes):
            one_hot[c] = (labels == c).astype(np.float32)
        return one_hot

    def _compute_dice_per_class(self, pred: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Compute Dice score for each class"""
        dice_scores = []
        for c in range(pred.shape[0]):
            dice = compute_dice_score(pred[c], target[c], ignore_background=False)
            dice_scores.append(dice)
        return np.array(dice_scores)

    def _compute_volume_metrics(self, pred: np.ndarray, target: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Compute volume-related metrics"""
        voxel_volume = np.prod(self.spacing)  # mm³
        volume_metrics = {}

        for class_idx in range(self.num_classes):
            pred_volume = np.sum(pred[class_idx]) * voxel_volume
            target_volume = np.sum(target[class_idx]) * voxel_volume

            if target_volume > 0:
                relative_volume_error = abs(pred_volume - target_volume) / target_volume
            else:
                relative_volume_error = 1.0 if pred_volume > 0 else 0.0

            volume_metrics[self.organ_names[class_idx]] = {
                'predicted_volume_mm3': pred_volume,
                'target_volume_mm3': target_volume,
                'relative_volume_error': relative_volume_error
            }

        return volume_metrics

    def aggregate_metrics(self, metric_list: List[Dict]) -> Dict[str, Union[float, Dict]]:
        """Aggregate metrics across multiple cases"""
        if not metric_list:
            return {}

        # Initialize aggregated metrics
        aggregated = {
            'num_cases': len(metric_list),
            'dice_scores': {},
            'hd95_scores': {},
            'nsd_scores': {},
            'volume_metrics': {}
        }

        # Aggregate Dice scores
        for organ in self.organ_names:
            dice_values = [m['dice_scores'][organ] for m in metric_list if organ in m['dice_scores']]
            if dice_values:
                aggregated['dice_scores'][organ] = {
                    'mean': np.mean(dice_values),
                    'std': np.std(dice_values),
                    'median': np.median(dice_values),
                    'min': np.min(dice_values),
                    'max': np.max(dice_values)
                }

        # Aggregate by organ size
        small_dice_all = [m['small_organ_dice'] for m in metric_list]
        medium_dice_all = [m['medium_organ_dice'] for m in metric_list]
        large_dice_all = [m['large_organ_dice'] for m in metric_list]

        aggregated['small_organ_dice'] = {
            'mean': np.mean(small_dice_all),
            'std': np.std(small_dice_all)
        }
        aggregated['medium_organ_dice'] = {
            'mean': np.mean(medium_dice_all),
            'std': np.std(medium_dice_all)
        }
        aggregated['large_organ_dice'] = {
            'mean': np.mean(large_dice_all),
            'std': np.std(large_dice_all)
        }

        # Overall metrics
        overall_dice = [m['mean_dice'] for m in metric_list]
        aggregated['overall_dice'] = {
            'mean': np.mean(overall_dice),
            'std': np.std(overall_dice)
        }

        # Aggregate Hausdorff distances
        for organ in self.organ_names[1:]:  # Skip background
            hd95_values = [m['hd95_scores'][organ] for m in metric_list
                          if organ in m['hd95_scores'] and not np.isinf(m['hd95_scores'][organ])]
            if hd95_values:
                aggregated['hd95_scores'][organ] = {
                    'mean': np.mean(hd95_values),
                    'std': np.std(hd95_values),
                    'median': np.median(hd95_values)
                }

        return aggregated
