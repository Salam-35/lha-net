#!/usr/bin/env python3
"""
Heavy data augmentation following nnU-Net methodology.
Applies 12+ augmentation types for robust training.
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import map_coordinates, gaussian_filter
import random
from typing import Tuple, Optional


class HeavyAugmentation:
    """
    nnU-Net style heavy augmentation pipeline.

    Augmentation Types:
    1. Spatial: rotation, scaling, elastic deformation, mirroring
    2. Intensity: gamma, brightness, contrast, noise, blur
    3. Resolution: simulate low resolution
    """

    def __init__(
        self,
        # Spatial augmentation params
        rotation_range: Tuple[float, float] = (-30, 30),
        scale_range: Tuple[float, float] = (0.85, 1.25),
        elastic_alpha: float = 900,
        elastic_sigma: float = 9,

        # Intensity augmentation params
        gamma_range: Tuple[float, float] = (0.7, 1.5),
        brightness_range: Tuple[float, float] = (0.7, 1.3),
        contrast_range: Tuple[float, float] = (0.65, 1.5),
        noise_variance: Tuple[float, float] = (0, 0.1),
        blur_sigma_range: Tuple[float, float] = (0.5, 1.0),

        # Probabilities
        p_rotation: float = 0.2,
        p_scale: float = 0.2,
        p_elastic: float = 0.2,
        p_mirror: float = 0.5,
        p_gamma: float = 0.3,
        p_brightness: float = 0.15,
        p_contrast: float = 0.15,
        p_noise: float = 0.1,
        p_blur: float = 0.2,
        p_low_res: float = 0.25,
    ):
        # Spatial params
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma

        # Intensity params
        self.gamma_range = gamma_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_variance = noise_variance
        self.blur_sigma_range = blur_sigma_range

        # Probabilities
        self.p_rotation = p_rotation
        self.p_scale = p_scale
        self.p_elastic = p_elastic
        self.p_mirror = p_mirror
        self.p_gamma = p_gamma
        self.p_brightness = p_brightness
        self.p_contrast = p_contrast
        self.p_noise = p_noise
        self.p_blur = p_blur
        self.p_low_res = p_low_res

    def __call__(
        self,
        image: np.ndarray,
        label: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply augmentation pipeline.

        Args:
            image: Input image (D, H, W) - already normalized
            label: Segmentation mask (D, H, W)

        Returns:
            Augmented image and label
        """
        # 1. Spatial augmentations (affect both image and label)

        # Mirror/Flip (all axes)
        if random.random() < self.p_mirror:
            image, label = self.random_mirror(image, label)

        # Rotation
        if random.random() < self.p_rotation:
            image, label = self.random_rotation(image, label)

        # Scaling
        if random.random() < self.p_scale:
            image, label = self.random_scale(image, label)

        # Elastic deformation
        if random.random() < self.p_elastic:
            image, label = self.elastic_deformation(image, label)

        # 2. Intensity augmentations (only affect image)

        # Gamma correction
        if random.random() < self.p_gamma:
            image = self.gamma_correction(image)

        # Brightness (multiplicative)
        if random.random() < self.p_brightness:
            image = self.brightness_augmentation(image)

        # Contrast
        if random.random() < self.p_contrast:
            image = self.contrast_augmentation(image)

        # Gaussian noise
        if random.random() < self.p_noise:
            image = self.add_gaussian_noise(image)

        # Gaussian blur
        if random.random() < self.p_blur:
            image = self.gaussian_blur(image)

        # Simulate low resolution
        if random.random() < self.p_low_res:
            image = self.simulate_low_resolution(image)

        # Ensure contiguous arrays
        image = np.ascontiguousarray(image)
        label = np.ascontiguousarray(label)

        return image, label

    # ==================== SPATIAL AUGMENTATIONS ====================

    def random_mirror(
        self,
        image: np.ndarray,
        label: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Random flip along each axis independently"""
        for axis in range(3):
            if random.random() < 0.5:
                image = np.flip(image, axis=axis).copy()
                label = np.flip(label, axis=axis).copy()
        return image, label

    def random_rotation(
        self,
        image: np.ndarray,
        label: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Random rotation around all three axes.
        Uses scipy.ndimage.rotate for proper interpolation.
        """
        # Random angles for each axis
        angle_x = random.uniform(*self.rotation_range)
        angle_y = random.uniform(*self.rotation_range)
        angle_z = random.uniform(*self.rotation_range)

        # Rotate around z-axis (axial plane)
        if abs(angle_z) > 1:
            image = ndimage.rotate(image, angle_z, axes=(1, 2),
                                   reshape=False, order=3, mode='nearest')
            label = ndimage.rotate(label, angle_z, axes=(1, 2),
                                   reshape=False, order=0, mode='nearest')

        # Rotate around y-axis (coronal plane)
        if abs(angle_y) > 1:
            image = ndimage.rotate(image, angle_y, axes=(0, 2),
                                   reshape=False, order=3, mode='nearest')
            label = ndimage.rotate(label, angle_y, axes=(0, 2),
                                   reshape=False, order=0, mode='nearest')

        # Rotate around x-axis (sagittal plane)
        if abs(angle_x) > 1:
            image = ndimage.rotate(image, angle_x, axes=(0, 1),
                                   reshape=False, order=3, mode='nearest')
            label = ndimage.rotate(label, angle_x, axes=(0, 1),
                                   reshape=False, order=0, mode='nearest')

        return image, label

    def random_scale(
        self,
        image: np.ndarray,
        label: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Random scaling with crop/pad to maintain size.
        """
        scale = random.uniform(*self.scale_range)
        original_shape = image.shape

        # Zoom
        image_scaled = ndimage.zoom(image, scale, order=3, mode='nearest')
        label_scaled = ndimage.zoom(label, scale, order=0, mode='nearest')

        # Crop or pad to original size
        image = self._crop_or_pad(image_scaled, original_shape)
        label = self._crop_or_pad(label_scaled, original_shape)

        return image, label

    def _crop_or_pad(
        self,
        volume: np.ndarray,
        target_shape: Tuple[int, int, int]
    ) -> np.ndarray:
        """Crop or pad volume to target shape (center crop/pad)"""
        result = np.zeros(target_shape, dtype=volume.dtype)

        # Calculate crop/pad amounts
        for i in range(3):
            if volume.shape[i] > target_shape[i]:
                # Crop (center)
                start = (volume.shape[i] - target_shape[i]) // 2
                if i == 0:
                    volume = volume[start:start+target_shape[i], :, :]
                elif i == 1:
                    volume = volume[:, start:start+target_shape[i], :]
                else:
                    volume = volume[:, :, start:start+target_shape[i]]

        # Pad if smaller
        pad_before = [(max(0, (t - s) // 2)) for s, t in zip(volume.shape, target_shape)]
        pad_after = [(max(0, t - s - pb)) for s, t, pb in zip(volume.shape, target_shape, pad_before)]

        if any(pb > 0 or pa > 0 for pb, pa in zip(pad_before, pad_after)):
            volume = np.pad(volume, list(zip(pad_before, pad_after)), mode='constant')

        # Final crop to exact size
        result = volume[:target_shape[0], :target_shape[1], :target_shape[2]]

        return result

    def elastic_deformation(
        self,
        image: np.ndarray,
        label: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Elastic deformation using random displacement fields.
        Based on Simard et al. (2003) and nnU-Net implementation.
        """
        shape = image.shape

        # Generate random displacement fields
        dx = gaussian_filter(
            (np.random.rand(*shape) * 2 - 1),
            self.elastic_sigma
        ) * self.elastic_alpha
        dy = gaussian_filter(
            (np.random.rand(*shape) * 2 - 1),
            self.elastic_sigma
        ) * self.elastic_alpha
        dz = gaussian_filter(
            (np.random.rand(*shape) * 2 - 1),
            self.elastic_sigma
        ) * self.elastic_alpha

        # Create coordinate grids
        z, y, x = np.meshgrid(
            np.arange(shape[0]),
            np.arange(shape[1]),
            np.arange(shape[2]),
            indexing='ij'
        )

        # Apply displacements
        indices = [
            np.clip(z + dz, 0, shape[0] - 1),
            np.clip(y + dy, 0, shape[1] - 1),
            np.clip(x + dx, 0, shape[2] - 1)
        ]

        # Interpolate
        image = map_coordinates(image, indices, order=3, mode='nearest')
        label = map_coordinates(label, indices, order=0, mode='nearest')

        return image, label

    # ==================== INTENSITY AUGMENTATIONS ====================

    def gamma_correction(self, image: np.ndarray) -> np.ndarray:
        """
        Random gamma correction.
        Preserves relative intensities while changing contrast.
        """
        gamma = random.uniform(*self.gamma_range)

        # Handle negative values (from z-score normalization)
        image_min = image.min()
        image_shifted = image - image_min + 1e-8

        # Apply gamma
        image_gamma = np.power(image_shifted, gamma)

        # Shift back
        image = image_gamma + image_min

        # 50% chance to invert gamma (dark <-> bright)
        if random.random() < 0.5:
            image_mean = image.mean()
            image = 2 * image_mean - image

        return image

    def brightness_augmentation(self, image: np.ndarray) -> np.ndarray:
        """Multiplicative brightness augmentation"""
        factor = random.uniform(*self.brightness_range)
        return image * factor

    def contrast_augmentation(self, image: np.ndarray) -> np.ndarray:
        """
        Contrast augmentation by scaling around mean.
        """
        factor = random.uniform(*self.contrast_range)
        mean = image.mean()
        return (image - mean) * factor + mean

    def add_gaussian_noise(self, image: np.ndarray) -> np.ndarray:
        """Add Gaussian noise"""
        variance = random.uniform(*self.noise_variance)
        noise = np.random.normal(0, variance ** 0.5, image.shape)
        return image + noise

    def gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur"""
        sigma = random.uniform(*self.blur_sigma_range)
        return gaussian_filter(image, sigma=sigma)

    def simulate_low_resolution(self, image: np.ndarray) -> np.ndarray:
        """
        Simulate low resolution by downsampling and upsampling.
        Helps model handle varying image qualities.
        """
        original_shape = image.shape

        # Random zoom factor (0.5 to 1.0)
        zoom_factor = random.uniform(0.5, 1.0)

        # Only apply to 2D (preserve z-resolution in CT)
        # Downsample
        downsampled = ndimage.zoom(
            image,
            (1, zoom_factor, zoom_factor),
            order=0,
            mode='nearest'
        )

        # Upsample back
        upsampled = ndimage.zoom(
            downsampled,
            (1, original_shape[1] / downsampled.shape[1],
             original_shape[2] / downsampled.shape[2]),
            order=3,
            mode='nearest'
        )

        # Ensure same shape
        return upsampled[:original_shape[0], :original_shape[1], :original_shape[2]]


# ==================== CONVENIENCE FUNCTIONS ====================

def get_training_augmentation() -> HeavyAugmentation:
    """Get default training augmentation (nnU-Net style)"""
    return HeavyAugmentation(
        # Spatial
        rotation_range=(-30, 30),
        scale_range=(0.85, 1.25),
        elastic_alpha=900,
        elastic_sigma=9,

        # Intensity
        gamma_range=(0.7, 1.5),
        brightness_range=(0.7, 1.3),
        contrast_range=(0.65, 1.5),
        noise_variance=(0, 0.1),
        blur_sigma_range=(0.5, 1.0),

        # Probabilities (nnU-Net defaults)
        p_rotation=0.2,
        p_scale=0.2,
        p_elastic=0.2,
        p_mirror=0.5,
        p_gamma=0.3,
        p_brightness=0.15,
        p_contrast=0.15,
        p_noise=0.1,
        p_blur=0.2,
        p_low_res=0.25,
    )


def get_light_augmentation() -> HeavyAugmentation:
    """Get lighter augmentation (for debugging or small datasets)"""
    return HeavyAugmentation(
        rotation_range=(-15, 15),
        scale_range=(0.9, 1.1),
        p_rotation=0.1,
        p_scale=0.1,
        p_elastic=0.0,  # Disable elastic
        p_gamma=0.15,
        p_brightness=0.1,
        p_contrast=0.1,
        p_noise=0.05,
        p_blur=0.1,
        p_low_res=0.1,
    )


def get_validation_augmentation() -> None:
    """No augmentation for validation"""
    return None
