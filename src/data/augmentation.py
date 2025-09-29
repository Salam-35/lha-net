import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Union
import random


class AMOS22Augmentation:
    """Data augmentation pipeline for AMOS22 medical volumes"""

    def __init__(
        self,
        enabled: bool = True,
        rotation_range: Tuple[float, float] = (-15, 15),
        scaling_range: Tuple[float, float] = (0.9, 1.1),
        intensity_shift_range: Tuple[float, float] = (-0.1, 0.1),
        intensity_scale_range: Tuple[float, float] = (0.9, 1.1),
        elastic_deformation: bool = False,
        random_flip: bool = True,
        gaussian_noise_std: float = 0.01,
        probability: float = 0.8
    ):
        self.enabled = enabled
        self.rotation_range = rotation_range
        self.scaling_range = scaling_range
        self.intensity_shift_range = intensity_shift_range
        self.intensity_scale_range = intensity_scale_range
        self.elastic_deformation = elastic_deformation
        self.random_flip = random_flip
        self.gaussian_noise_std = gaussian_noise_std
        self.probability = probability

    def apply_augmentation(
        self,
        image: torch.Tensor,
        label: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply augmentation to image and label pair

        Args:
            image: Input image tensor (C, D, H, W) or (D, H, W)
            label: Label tensor (D, H, W)

        Returns:
            Augmented image and label tensors
        """
        if not self.enabled or random.random() > self.probability:
            return image, label

        # Ensure consistent tensor format
        if len(image.shape) == 3:
            image = image.unsqueeze(0)  # Add channel dimension

        device = image.device
        dtype = image.dtype

        # Apply geometric transformations (both image and label)
        if random.random() < 0.7:  # 70% chance for geometric transforms
            image, label = self._apply_geometric_transforms(image, label)

        # Apply intensity transformations (image only)
        if random.random() < 0.6:  # 60% chance for intensity transforms
            image = self._apply_intensity_transforms(image)

        return image, label

    def _apply_geometric_transforms(
        self,
        image: torch.Tensor,
        label: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply geometric transformations to both image and label"""

        # Random rotation
        if random.random() < 0.5:
            angle = random.uniform(*self.rotation_range)
            image, label = self._rotate_3d(image, label, angle)

        # Random scaling
        if random.random() < 0.4:
            scale_factor = random.uniform(*self.scaling_range)
            image, label = self._scale_3d(image, label, scale_factor)

        # Random flipping
        if self.random_flip and random.random() < 0.5:
            image, label = self._random_flip_3d(image, label)

        return image, label

    def _apply_intensity_transforms(self, image: torch.Tensor) -> torch.Tensor:
        """Apply intensity transformations to image"""

        # Intensity shift
        if random.random() < 0.5:
            shift = random.uniform(*self.intensity_shift_range)
            image = image + shift

        # Intensity scaling
        if random.random() < 0.5:
            scale = random.uniform(*self.intensity_scale_range)
            image = image * scale

        # Gaussian noise
        if random.random() < 0.3 and self.gaussian_noise_std > 0:
            noise = torch.randn_like(image) * self.gaussian_noise_std
            image = image + noise

        return image

    def _rotate_3d(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
        angle: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply 3D rotation around random axis"""

        # For simplicity, rotate around z-axis (axial plane)
        angle_rad = np.deg2rad(angle)

        # Create rotation matrix
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        # Rotation matrix for 2D rotation in last two dimensions
        rotation_matrix = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0]
        ], dtype=torch.float32, device=image.device)

        # Create affine grid
        grid = F.affine_grid(
            rotation_matrix.unsqueeze(0),
            image.shape,
            align_corners=False
        )

        # Apply rotation
        rotated_image = F.grid_sample(
            image.unsqueeze(0),
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        ).squeeze(0)

        rotated_label = F.grid_sample(
            label.unsqueeze(0).unsqueeze(0).float(),
            grid,
            mode='nearest',
            padding_mode='zeros',
            align_corners=False
        ).squeeze(0).squeeze(0).long()

        return rotated_image, rotated_label

    def _scale_3d(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
        scale_factor: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply 3D scaling"""

        # Create scaling matrix
        scale_matrix = torch.tensor([
            [scale_factor, 0, 0],
            [0, scale_factor, 0]
        ], dtype=torch.float32, device=image.device)

        # Create affine grid
        grid = F.affine_grid(
            scale_matrix.unsqueeze(0),
            image.shape,
            align_corners=False
        )

        # Apply scaling
        scaled_image = F.grid_sample(
            image.unsqueeze(0),
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        ).squeeze(0)

        scaled_label = F.grid_sample(
            label.unsqueeze(0).unsqueeze(0).float(),
            grid,
            mode='nearest',
            padding_mode='border',
            align_corners=False
        ).squeeze(0).squeeze(0).long()

        return scaled_image, scaled_label

    def _random_flip_3d(
        self,
        image: torch.Tensor,
        label: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random flipping along random axes"""

        # Choose random axes to flip
        axes_to_flip = []
        for axis in [-1, -2, -3]:  # D, H, W
            if random.random() < 0.5:
                axes_to_flip.append(axis)

        # Apply flips
        for axis in axes_to_flip:
            image = torch.flip(image, dims=[axis])
            label = torch.flip(label, dims=[axis])

        return image, label

    def _elastic_deformation_3d(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
        alpha: float = 1000,
        sigma: float = 50
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply elastic deformation (computationally expensive, disabled by default)
        """
        # This is a simplified version - full implementation would be more complex
        # and memory-intensive for 3D volumes

        if not self.elastic_deformation:
            return image, label

        shape = image.shape[-3:]  # D, H, W
        device = image.device

        # Generate random displacement fields
        dx = torch.randn(shape, device=device) * alpha
        dy = torch.randn(shape, device=device) * alpha
        dz = torch.randn(shape, device=device) * alpha

        # Apply Gaussian smoothing (simplified)
        # In practice, you'd use proper Gaussian filtering
        dx = F.avg_pool3d(dx.unsqueeze(0).unsqueeze(0), 3, stride=1, padding=1).squeeze()
        dy = F.avg_pool3d(dy.unsqueeze(0).unsqueeze(0), 3, stride=1, padding=1).squeeze()
        dz = F.avg_pool3d(dz.unsqueeze(0).unsqueeze(0), 3, stride=1, padding=1).squeeze()

        # Create displacement grid
        d, h, w = shape
        grid_d, grid_h, grid_w = torch.meshgrid(
            torch.arange(d, device=device, dtype=torch.float32),
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing='ij'
        )

        # Apply displacement
        new_d = grid_d + dz
        new_h = grid_h + dy
        new_w = grid_w + dx

        # Normalize to [-1, 1]
        new_d = 2.0 * new_d / (d - 1) - 1.0
        new_h = 2.0 * new_h / (h - 1) - 1.0
        new_w = 2.0 * new_w / (w - 1) - 1.0

        # Stack to create grid
        grid = torch.stack([new_w, new_h, new_d], dim=-1).unsqueeze(0)

        # Apply deformation
        deformed_image = F.grid_sample(
            image.unsqueeze(0),
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        ).squeeze(0)

        deformed_label = F.grid_sample(
            label.unsqueeze(0).unsqueeze(0).float(),
            grid,
            mode='nearest',
            padding_mode='border',
            align_corners=False
        ).squeeze(0).squeeze(0).long()

        return deformed_image, deformed_label

    def get_augmentation_info(self) -> Dict[str, Union[bool, float, Tuple]]:
        """Get augmentation configuration info"""
        return {
            'enabled': self.enabled,
            'rotation_range': self.rotation_range,
            'scaling_range': self.scaling_range,
            'intensity_shift_range': self.intensity_shift_range,
            'intensity_scale_range': self.intensity_scale_range,
            'random_flip': self.random_flip,
            'gaussian_noise_std': self.gaussian_noise_std,
            'elastic_deformation': self.elastic_deformation,
            'probability': self.probability
        }


class CurriculumAugmentation:
    """Curriculum learning approach to augmentation"""

    def __init__(
        self,
        base_augmentation: AMOS22Augmentation,
        curriculum_epochs: int = 50,
        initial_intensity: float = 0.3,
        final_intensity: float = 1.0
    ):
        self.base_augmentation = base_augmentation
        self.curriculum_epochs = curriculum_epochs
        self.initial_intensity = initial_intensity
        self.final_intensity = final_intensity

    def get_augmentation_for_epoch(self, epoch: int) -> AMOS22Augmentation:
        """Get augmentation with intensity based on current epoch"""

        # Calculate current intensity
        if epoch >= self.curriculum_epochs:
            intensity = self.final_intensity
        else:
            progress = epoch / self.curriculum_epochs
            intensity = self.initial_intensity + (self.final_intensity - self.initial_intensity) * progress

        # Create adjusted augmentation
        adjusted_aug = AMOS22Augmentation(
            enabled=self.base_augmentation.enabled,
            rotation_range=self._scale_range(self.base_augmentation.rotation_range, intensity),
            scaling_range=self._scale_range(self.base_augmentation.scaling_range, intensity),
            intensity_shift_range=self._scale_range(self.base_augmentation.intensity_shift_range, intensity),
            intensity_scale_range=self._scale_range(self.base_augmentation.intensity_scale_range, intensity),
            elastic_deformation=self.base_augmentation.elastic_deformation,
            random_flip=self.base_augmentation.random_flip,
            gaussian_noise_std=self.base_augmentation.gaussian_noise_std * intensity,
            probability=self.base_augmentation.probability * intensity
        )

        return adjusted_aug

    def _scale_range(self, range_tuple: Tuple[float, float], intensity: float) -> Tuple[float, float]:
        """Scale augmentation range by intensity"""
        low, high = range_tuple
        center = (low + high) / 2
        half_range = (high - low) / 2

        scaled_half_range = half_range * intensity
        return (center - scaled_half_range, center + scaled_half_range)