"""
Dataset class for loading preprocessed data.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, Dict, Optional
import random

from .augmentation_heavy import get_training_augmentation, get_validation_augmentation


class PreprocessedAMOS22Dataset(Dataset):
    """
    Dataset for loading preprocessed AMOS22 data.

    Expects preprocessed data structure (same as original AMOS):
        data_dir/
            imagesTr/
                amos_0001.npz
            labelsTr/
                amos_0001.npz
            metadataTr/
                amos_0001.json
            imagesVal/
            labelsVal/
            metadataVal/
            preprocessing_config.json
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        patch_size: Tuple[int, int, int] = (96, 160, 160),
        num_patches_per_volume: int = 2,
        augmentation: bool = True
    ):
        """
        Args:
            data_dir: Path to preprocessed data directory
            split: Which split to use ('train', 'val', 'test')
            patch_size: Size of patches to extract (D, H, W)
            num_patches_per_volume: Number of patches per volume per epoch
            augmentation: Whether to apply augmentation
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.patch_size = patch_size
        self.num_patches_per_volume = num_patches_per_volume
        self.is_training = (split == 'train')
        self.augmentation = augmentation and self.is_training

        # Load preprocessing config
        config_path = self.data_dir / 'preprocessing_config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.preprocess_config = json.load(f)

        # Set directories based on split
        if split == 'train':
            self.image_dir = self.data_dir / 'imagesTr'
            self.label_dir = self.data_dir / 'labelsTr'
        elif split == 'val':
            self.image_dir = self.data_dir / 'imagesVal'
            self.label_dir = self.data_dir / 'labelsVal'
        elif split == 'test':
            self.image_dir = self.data_dir / 'imagesTs'
            self.label_dir = self.data_dir / 'labelsTs'
        else:
            raise ValueError(f"Unknown split: {split}")

        # Get list of cases
        self.case_names = sorted([
            p.stem for p in self.image_dir.glob("*.npz")
        ])

        if len(self.case_names) == 0:
            raise ValueError(f"No .npz files found in {self.image_dir}")

        # Setup heavy augmentation
        if self.augmentation:
            self.augmenter = get_training_augmentation()
            print(f"Loaded {len(self.case_names)} {split} cases from {data_dir}")
            print(f"Patch size: {patch_size}")
            print(f"Patches per volume: {num_patches_per_volume}")
            print(f"Heavy augmentation: ENABLED")
        else:
            self.augmenter = None
            print(f"Loaded {len(self.case_names)} {split} cases from {data_dir}")
            print(f"Patch size: {patch_size}")
            print(f"Patches per volume: {num_patches_per_volume}")
            print(f"Augmentation: DISABLED")

    def __len__(self):
        return len(self.case_names) * self.num_patches_per_volume

    def _load_case(self, case_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load preprocessed image and label"""
        image = np.load(self.image_dir / f'{case_name}.npz')['data']
        label = np.load(self.label_dir / f'{case_name}.npz')['data']
        return image, label

    def _sample_patch(
        self,
        image: np.ndarray,
        label: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample a patch from the volume.

        Strategy:
        - 50% centered on foreground
        - 50% random
        """
        d, h, w = image.shape
        pd, ph, pw = self.patch_size

        if self.is_training and random.random() < 0.5:
            # Sample centered on foreground
            foreground = np.where(label > 0)
            if len(foreground[0]) > 0:
                idx = random.randint(0, len(foreground[0]) - 1)
                center_d = foreground[0][idx]
                center_h = foreground[1][idx]
                center_w = foreground[2][idx]

                # Compute start indices
                start_d = max(0, min(center_d - pd // 2, d - pd))
                start_h = max(0, min(center_h - ph // 2, h - ph))
                start_w = max(0, min(center_w - pw // 2, w - pw))
            else:
                # Fallback to random
                start_d = random.randint(0, max(0, d - pd))
                start_h = random.randint(0, max(0, h - ph))
                start_w = random.randint(0, max(0, w - pw))
        else:
            # Random sampling
            start_d = random.randint(0, max(0, d - pd))
            start_h = random.randint(0, max(0, h - ph))
            start_w = random.randint(0, max(0, w - pw))

        # Extract patch
        image_patch = image[start_d:start_d+pd, start_h:start_h+ph, start_w:start_w+pw]
        label_patch = label[start_d:start_d+pd, start_h:start_h+ph, start_w:start_w+pw]

        # Pad if necessary
        if image_patch.shape != self.patch_size:
            image_patch = self._pad_to_size(image_patch, self.patch_size)
            label_patch = self._pad_to_size(label_patch, self.patch_size, mode='constant')

        return image_patch, label_patch

    def _pad_to_size(
        self,
        volume: np.ndarray,
        target_size: Tuple[int, int, int],
        mode: str = 'constant'
    ) -> np.ndarray:
        """Pad volume to target size"""
        pad_d = target_size[0] - volume.shape[0]
        pad_h = target_size[1] - volume.shape[1]
        pad_w = target_size[2] - volume.shape[2]

        padding = (
            (0, max(0, pad_d)),
            (0, max(0, pad_h)),
            (0, max(0, pad_w))
        )

        return np.pad(volume, padding, mode=mode)

    def _apply_augmentation(
        self,
        image: np.ndarray,
        label: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply heavy data augmentation using nnU-Net style pipeline"""
        if self.augmenter is not None:
            return self.augmenter(image, label)
        return image, label

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get case index
        case_idx = idx // self.num_patches_per_volume
        case_name = self.case_names[case_idx]

        # Load case
        image, label = self._load_case(case_name)

        # Sample patch
        image_patch, label_patch = self._sample_patch(image, label)

        # Apply augmentation
        if self.augmentation:
            image_patch, label_patch = self._apply_augmentation(
                image_patch, label_patch
            )

        # Convert to tensors
        image_tensor = torch.from_numpy(image_patch).float().unsqueeze(0)  # (1, D, H, W)
        label_tensor = torch.from_numpy(label_patch).long()  # (D, H, W)

        return {
            'image': image_tensor,
            'label': label_tensor,
            'case_name': case_name
        }
