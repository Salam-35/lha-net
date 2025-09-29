import torch
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import random
from scipy import ndimage


class SmartPatchSampler:
    """Smart patch sampling strategy biased toward small organ regions"""

    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (64, 128, 128),
        overlap_ratio: float = 0.5,
        small_organ_bias: float = 0.7,
        boundary_bias: float = 0.2,
        random_bias: float = 0.1,
        min_organ_voxels: int = 10
    ):
        self.patch_size = patch_size
        self.overlap_ratio = overlap_ratio
        self.small_organ_bias = small_organ_bias
        self.boundary_bias = boundary_bias
        self.random_bias = random_bias
        self.min_organ_voxels = min_organ_voxels

        # AMOS22 organ size categories
        self.small_organs = [9, 12]  # gallbladder, duodenum
        self.medium_organs = [7, 8]  # adrenal glands
        self.large_organs = [1, 2, 3, 4, 5, 6, 10, 11, 13]  # rest

        assert abs(small_organ_bias + boundary_bias + random_bias - 1.0) < 1e-6, \
            "Bias weights must sum to 1.0"

    def _get_stride(self) -> Tuple[int, int, int]:
        """Calculate stride based on overlap ratio"""
        return tuple(int(size * (1 - self.overlap_ratio)) for size in self.patch_size)

    def _find_organ_regions(self, label_volume: np.ndarray, organ_ids: List[int]) -> List[Tuple[int, int, int]]:
        """Find all regions containing specified organs"""
        regions = []

        for organ_id in organ_ids:
            organ_mask = (label_volume == organ_id)
            if not organ_mask.any():
                continue

            # Find connected components
            labeled_array, num_features = ndimage.label(organ_mask)

            for component_id in range(1, num_features + 1):
                component_mask = (labeled_array == component_id)
                if np.sum(component_mask) < self.min_organ_voxels:
                    continue

                # Get bounding box
                coords = np.where(component_mask)
                z_min, z_max = coords[0].min(), coords[0].max()
                y_min, y_max = coords[1].min(), coords[1].max()
                x_min, x_max = coords[2].min(), coords[2].max()

                # Add some padding around the organ
                pad_z, pad_y, pad_x = [size // 4 for size in self.patch_size]

                z_center = (z_min + z_max) // 2
                y_center = (y_min + y_max) // 2
                x_center = (x_min + x_max) // 2

                regions.append((z_center, y_center, x_center))

        return regions

    def _find_boundary_regions(self, label_volume: np.ndarray) -> List[Tuple[int, int, int]]:
        """Find boundary regions between different organs"""
        # Compute gradient to find boundaries
        grad_z = np.abs(np.diff(label_volume, axis=0))
        grad_y = np.abs(np.diff(label_volume, axis=1))
        grad_x = np.abs(np.diff(label_volume, axis=2))

        # Pad gradients to match original volume shape
        grad_z = np.pad(grad_z, ((0, 1), (0, 0), (0, 0)), mode='constant')
        grad_y = np.pad(grad_y, ((0, 0), (0, 1), (0, 0)), mode='constant')
        grad_x = np.pad(grad_x, ((0, 0), (0, 0), (0, 1)), mode='constant')

        # Combine gradients
        boundary_mask = (grad_z > 0) | (grad_y > 0) | (grad_x > 0)

        # Find boundary coordinates
        boundary_coords = np.where(boundary_mask)
        if len(boundary_coords[0]) == 0:
            return []

        # Subsample boundary points
        num_boundaries = min(len(boundary_coords[0]), 50)
        indices = np.random.choice(len(boundary_coords[0]), num_boundaries, replace=False)

        boundary_regions = [
            (boundary_coords[0][i], boundary_coords[1][i], boundary_coords[2][i])
            for i in indices
        ]

        return boundary_regions

    def _generate_random_positions(
        self,
        volume_shape: Tuple[int, int, int],
        num_positions: int
    ) -> List[Tuple[int, int, int]]:
        """Generate random sampling positions"""
        positions = []

        for _ in range(num_positions):
            z = random.randint(self.patch_size[0] // 2, volume_shape[0] - self.patch_size[0] // 2)
            y = random.randint(self.patch_size[1] // 2, volume_shape[1] - self.patch_size[1] // 2)
            x = random.randint(self.patch_size[2] // 2, volume_shape[2] - self.patch_size[2] // 2)
            positions.append((z, y, x))

        return positions

    def _is_valid_patch_center(
        self,
        center: Tuple[int, int, int],
        volume_shape: Tuple[int, int, int]
    ) -> bool:
        """Check if patch center results in valid patch within volume bounds"""
        z, y, x = center
        d, h, w = self.patch_size

        z_start, z_end = z - d // 2, z + d // 2
        y_start, y_end = y - h // 2, y + h // 2
        x_start, x_end = x - w // 2, x + w // 2

        return (0 <= z_start and z_end <= volume_shape[0] and
                0 <= y_start and y_end <= volume_shape[1] and
                0 <= x_start and x_end <= volume_shape[2])

    def _extract_patch(
        self,
        volume: np.ndarray,
        center: Tuple[int, int, int]
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int, int, int]]:
        """Extract patch from volume given center coordinates"""
        z, y, x = center
        d, h, w = self.patch_size

        z_start, z_end = z - d // 2, z + d // 2
        y_start, y_end = y - h // 2, y + h // 2
        x_start, x_end = x - w // 2, x + w // 2

        # Ensure we don't go out of bounds
        z_start = max(0, z_start)
        z_end = min(volume.shape[0], z_end)
        y_start = max(0, y_start)
        y_end = min(volume.shape[1], y_end)
        x_start = max(0, x_start)
        x_end = min(volume.shape[2], x_end)

        patch = volume[z_start:z_end, y_start:y_end, x_start:x_end]

        # Pad if necessary to reach desired patch size
        pad_z = self.patch_size[0] - patch.shape[0]
        pad_y = self.patch_size[1] - patch.shape[1]
        pad_x = self.patch_size[2] - patch.shape[2]

        if pad_z > 0 or pad_y > 0 or pad_x > 0:
            pad_z_before, pad_z_after = pad_z // 2, pad_z - pad_z // 2
            pad_y_before, pad_y_after = pad_y // 2, pad_y - pad_y // 2
            pad_x_before, pad_x_after = pad_x // 2, pad_x - pad_x // 2

            patch = np.pad(patch, ((pad_z_before, pad_z_after),
                                  (pad_y_before, pad_y_after),
                                  (pad_x_before, pad_x_after)), mode='constant')

        bbox = (z_start, z_end, y_start, y_end, x_start, x_end)
        return patch, bbox

    def sample_patches(
        self,
        image_volume: np.ndarray,
        label_volume: np.ndarray,
        num_patches: int = 16
    ) -> List[Dict[str, Union[np.ndarray, Tuple]]]:
        """
        Sample patches using smart sampling strategy

        Args:
            image_volume: Input image volume (D, H, W)
            label_volume: Label volume (D, H, W)
            num_patches: Number of patches to sample

        Returns:
            List of patch dictionaries containing image, label, center, and bbox
        """
        volume_shape = image_volume.shape
        patches = []

        # Calculate number of patches for each strategy
        num_small_patches = int(num_patches * self.small_organ_bias)
        num_boundary_patches = int(num_patches * self.boundary_bias)
        num_random_patches = num_patches - num_small_patches - num_boundary_patches

        # 1. Small organ focused patches
        small_organ_regions = self._find_organ_regions(label_volume, self.small_organs)
        if small_organ_regions and num_small_patches > 0:
            selected_regions = random.choices(small_organ_regions,
                                           k=min(num_small_patches, len(small_organ_regions)))
            for region in selected_regions:
                if self._is_valid_patch_center(region, volume_shape):
                    image_patch, bbox = self._extract_patch(image_volume, region)
                    label_patch, _ = self._extract_patch(label_volume, region)

                    patches.append({
                        'image': image_patch,
                        'label': label_patch,
                        'center': region,
                        'bbox': bbox,
                        'type': 'small_organ'
                    })

        # 2. Boundary focused patches
        boundary_regions = self._find_boundary_regions(label_volume)
        if boundary_regions and num_boundary_patches > 0:
            selected_boundaries = random.choices(boundary_regions,
                                               k=min(num_boundary_patches, len(boundary_regions)))
            for region in selected_boundaries:
                if self._is_valid_patch_center(region, volume_shape):
                    image_patch, bbox = self._extract_patch(image_volume, region)
                    label_patch, _ = self._extract_patch(label_volume, region)

                    patches.append({
                        'image': image_patch,
                        'label': label_patch,
                        'center': region,
                        'bbox': bbox,
                        'type': 'boundary'
                    })

        # 3. Random patches
        if num_random_patches > 0:
            random_regions = self._generate_random_positions(volume_shape, num_random_patches)
            for region in random_regions:
                if self._is_valid_patch_center(region, volume_shape):
                    image_patch, bbox = self._extract_patch(image_volume, region)
                    label_patch, _ = self._extract_patch(label_volume, region)

                    patches.append({
                        'image': image_patch,
                        'label': label_patch,
                        'center': region,
                        'bbox': bbox,
                        'type': 'random'
                    })

        # Fill remaining patches if we didn't get enough
        while len(patches) < num_patches:
            random_region = self._generate_random_positions(volume_shape, 1)[0]
            if self._is_valid_patch_center(random_region, volume_shape):
                image_patch, bbox = self._extract_patch(image_volume, random_region)
                label_patch, _ = self._extract_patch(label_volume, random_region)

                patches.append({
                    'image': image_patch,
                    'label': label_patch,
                    'center': random_region,
                    'bbox': bbox,
                    'type': 'fill_random'
                })

        return patches[:num_patches]


class OrganFocusedSampler:
    """Advanced sampler that focuses specifically on challenging organ categories"""

    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (64, 128, 128),
        difficulty_weights: Optional[Dict[str, float]] = None,
        adaptive_sampling: bool = True
    ):
        self.patch_size = patch_size
        self.adaptive_sampling = adaptive_sampling

        if difficulty_weights is None:
            self.difficulty_weights = {
                'small': 4.0,      # Highest weight for small organs
                'medium': 2.5,     # Medium weight for medium organs
                'large': 1.0,      # Base weight for large organs
                'boundary': 3.0,   # High weight for boundaries
                'background': 0.5  # Low weight for background-only patches
            }
        else:
            self.difficulty_weights = difficulty_weights

        self.base_sampler = SmartPatchSampler(patch_size=patch_size)

    def _assess_patch_difficulty(self, label_patch: np.ndarray) -> str:
        """Assess the difficulty/importance of a patch based on its content"""
        unique_labels = np.unique(label_patch)

        # Check for small organs
        small_organs = [9, 12]
        if any(label in small_organs for label in unique_labels):
            return 'small'

        # Check for medium organs
        medium_organs = [7, 8]
        if any(label in medium_organs for label in unique_labels):
            return 'medium'

        # Check for boundaries (multiple organs present)
        if len(unique_labels) > 2:  # More than background + one organ
            return 'boundary'

        # Check for large organs
        large_organs = [1, 2, 3, 4, 5, 6, 10, 11, 13]
        if any(label in large_organs for label in unique_labels):
            return 'large'

        return 'background'

    def sample_adaptive_patches(
        self,
        image_volume: np.ndarray,
        label_volume: np.ndarray,
        num_patches: int = 16,
        training_iteration: Optional[int] = None
    ) -> List[Dict[str, Union[np.ndarray, Tuple, str, float]]]:
        """
        Sample patches with adaptive difficulty-based weighting

        Args:
            image_volume: Input image volume
            label_volume: Label volume
            num_patches: Number of patches to sample
            training_iteration: Current training iteration for curriculum learning

        Returns:
            List of patch dictionaries with difficulty scores
        """
        # Generate initial patch pool (more than needed)
        pool_size = num_patches * 3
        patch_pool = self.base_sampler.sample_patches(
            image_volume, label_volume, num_patches=pool_size
        )

        # Assess difficulty for each patch
        assessed_patches = []
        for patch in patch_pool:
            difficulty_category = self._assess_patch_difficulty(patch['label'])
            difficulty_weight = self.difficulty_weights[difficulty_category]

            # Curriculum learning: gradually increase focus on difficult patches
            if training_iteration is not None and self.adaptive_sampling:
                curriculum_factor = min(1.0, training_iteration / 1000.0)  # Ramp up over 1000 iterations
                if difficulty_category in ['small', 'boundary']:
                    difficulty_weight *= (1.0 + curriculum_factor)

            assessed_patches.append({
                **patch,
                'difficulty_category': difficulty_category,
                'difficulty_weight': difficulty_weight
            })

        # Sample patches based on difficulty weights
        weights = [p['difficulty_weight'] for p in assessed_patches]
        total_weight = sum(weights)

        if total_weight > 0:
            probabilities = [w / total_weight for w in weights]
            selected_indices = np.random.choice(
                len(assessed_patches),
                size=min(num_patches, len(assessed_patches)),
                replace=False,
                p=probabilities
            )

            selected_patches = [assessed_patches[i] for i in selected_indices]
        else:
            selected_patches = assessed_patches[:num_patches]

        return selected_patches