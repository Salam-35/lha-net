#!/usr/bin/env python3
"""
Preprocess entire dataset with nnU-Net style preprocessing.
Saves preprocessed data to disk to avoid recomputing during training.

Usage:
    python scripts/preprocess_dataset.py \
        --data_dir data/AMOS22 \
        --stats_file dataset_statistics.json \
        --output_dir data/AMOS22_preprocessed
"""

import os
import json
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
from scipy import ndimage
import warnings

warnings.filterwarnings('ignore')


class NNUNetPreprocessor:
    """nnU-Net style preprocessing pipeline"""

    def __init__(self, stats_file: str):
        """
        Args:
            stats_file: Path to dataset_statistics.json
        """
        with open(stats_file, 'r') as f:
            self.stats = json.load(f)

        # Extract key values
        self.global_mean = self.stats['intensity']['global_mean']
        self.global_std = self.stats['intensity']['global_std']
        self.clip_low = self.stats['intensity']['percentile_0_5']
        self.clip_high = self.stats['intensity']['percentile_99_5']
        self.target_spacing = np.array(self.stats['spacing']['median'])

        print("Preprocessor initialized with:")
        print(f"  Global mean: {self.global_mean:.2f}")
        print(f"  Global std: {self.global_std:.2f}")
        print(f"  Clip range: [{self.clip_low:.2f}, {self.clip_high:.2f}]")
        print(f"  Target spacing: {self.target_spacing}")

    def resample_volume(
        self,
        volume: np.ndarray,
        original_spacing: np.ndarray,
        target_spacing: np.ndarray,
        order: int = 3,
        is_label: bool = False
    ) -> np.ndarray:
        """
        Resample volume to target spacing.

        Args:
            volume: Input volume
            original_spacing: Original voxel spacing
            target_spacing: Target voxel spacing
            order: Interpolation order (3 for image, 0 for label)
            is_label: Whether this is a label volume

        Returns:
            Resampled volume
        """
        # Compute zoom factors
        zoom_factors = original_spacing / target_spacing

        # Check if resampling is needed
        if np.allclose(zoom_factors, 1.0, atol=0.01):
            return volume

        if is_label:
            # For labels: use nearest neighbor
            resampled = ndimage.zoom(volume, zoom_factors, order=0, mode='nearest')
        else:
            # For images: use cubic spline
            resampled = ndimage.zoom(volume, zoom_factors, order=order, mode='nearest')

        return resampled

    def crop_to_nonzero(
        self,
        image: np.ndarray,
        label: np.ndarray = None,
        margin: int = 5
    ):
        """
        Crop volume to non-zero bounding box with margin.

        Returns:
            Cropped image, label (if provided), and bounding box coordinates
        """
        # Find non-zero region
        if label is not None:
            nonzero = np.where(label > 0)
        else:
            # Use intensity threshold for CT
            nonzero = np.where(image > -500)

        if len(nonzero[0]) == 0:
            # No foreground, return as is
            bbox = [(0, s) for s in image.shape]
            return image, label, bbox

        # Get bounding box
        bbox = []
        slices = []
        for i in range(3):
            min_idx = max(0, nonzero[i].min() - margin)
            max_idx = min(image.shape[i], nonzero[i].max() + margin + 1)
            bbox.append((min_idx, max_idx))
            slices.append(slice(min_idx, max_idx))

        # Crop
        image_cropped = image[slices[0], slices[1], slices[2]]

        if label is not None:
            label_cropped = label[slices[0], slices[1], slices[2]]
            return image_cropped, label_cropped, bbox

        return image_cropped, None, bbox

    def preprocess_case(
        self,
        image_path: str,
        label_path: str = None,
        crop: bool = True
    ):
        """
        Preprocess a single case.

        Pipeline:
        1. Load image and label
        2. Resample to target spacing
        3. Clip intensities (percentile-based)
        4. Z-score normalize with global statistics
        5. Crop to non-zero region (optional)

        Returns:
            Preprocessed image, label, and metadata
        """
        # Load image
        img_nib = nib.load(image_path)
        image = img_nib.get_fdata().astype(np.float32)
        original_spacing = np.array(img_nib.header.get_zooms()[:3])
        affine = img_nib.affine

        # Load label if provided
        if label_path and os.path.exists(label_path):
            label_nib = nib.load(label_path)
            label = label_nib.get_fdata().astype(np.int32)
        else:
            label = None

        # Store original shape
        original_shape = image.shape

        # 1. Resample to target spacing
        image = self.resample_volume(
            image,
            original_spacing,
            self.target_spacing,
            order=3,
            is_label=False
        )

        if label is not None:
            label = self.resample_volume(
                label,
                original_spacing,
                self.target_spacing,
                order=0,
                is_label=True
            )

        resampled_shape = image.shape

        # 2. Clip intensities (percentile-based)
        image = np.clip(image, self.clip_low, self.clip_high)

        # 3. Z-score normalization with GLOBAL statistics
        image = (image - self.global_mean) / (self.global_std + 1e-8)

        # 4. Crop to non-zero region
        bbox = None
        if crop:
            image, label, bbox = self.crop_to_nonzero(image, label, margin=5)

        final_shape = image.shape

        # Metadata
        metadata = {
            'original_spacing': original_spacing.tolist(),
            'target_spacing': self.target_spacing.tolist(),
            'original_shape': list(original_shape),
            'resampled_shape': list(resampled_shape),
            'final_shape': list(final_shape),
            'bbox': bbox,
            'affine': affine.tolist(),
        }

        return image, label, metadata

    def preprocess_dataset(
        self,
        data_dir: str,
        output_dir: str,
        split: str = 'train',
        crop: bool = True
    ):
        """
        Preprocess entire dataset and save to disk.

        Args:
            data_dir: Path to AMOS22 dataset root
            output_dir: Output directory for preprocessed data
            split: Which split to process ('train', 'val', 'test')
            crop: Whether to crop to non-zero region

        Output structure (maintains original structure):
            output_dir/
                imagesTr/
                    amos_0001.npz
                    ...
                labelsTr/
                    amos_0001.npz
                    ...
                metadataTr/
                    amos_0001.json
                    ...
                imagesVal/
                labelsVal/
                metadataVal/
                preprocessing_config.json
        """
        data_dir = Path(data_dir)
        output_dir = Path(output_dir)

        # Find image and label directories based on split
        if split == 'train':
            image_dir = data_dir / 'imagesTr'
            label_dir = data_dir / 'labelsTr'
            out_img_dir = output_dir / 'imagesTr'
            out_label_dir = output_dir / 'labelsTr'
            out_meta_dir = output_dir / 'metadataTr'
        elif split == 'val':
            image_dir = data_dir / 'imagesVal'
            label_dir = data_dir / 'labelsVal'
            out_img_dir = output_dir / 'imagesVal'
            out_label_dir = output_dir / 'labelsVal'
            out_meta_dir = output_dir / 'metadataVal'
        elif split == 'test':
            image_dir = data_dir / 'imagesTs'
            label_dir = data_dir / 'labelsTs'
            out_img_dir = output_dir / 'imagesTs'
            out_label_dir = output_dir / 'labelsTs'
            out_meta_dir = output_dir / 'metadataTs'
        else:
            raise ValueError(f"Unknown split: {split}")

        # Create output directories
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_label_dir.mkdir(parents=True, exist_ok=True)
        out_meta_dir.mkdir(parents=True, exist_ok=True)

        # Get all image files
        image_files = sorted(list(image_dir.glob("*.nii.gz")))

        print(f"\nPreprocessing {len(image_files)} cases...")
        print("="*60)

        for img_path in tqdm(image_files, desc="Preprocessing"):
            case_name = img_path.stem.replace('.nii', '')

            # Find corresponding label
            label_path = label_dir / img_path.name
            if not label_path.exists():
                label_path = None

            try:
                # Preprocess
                image, label, metadata = self.preprocess_case(
                    str(img_path),
                    str(label_path) if label_path else None,
                    crop=crop
                )

                # Save preprocessed image
                np.savez_compressed(
                    out_img_dir / f'{case_name}.npz',
                    data=image.astype(np.float32)
                )

                # Save preprocessed label
                if label is not None:
                    np.savez_compressed(
                        out_label_dir / f'{case_name}.npz',
                        data=label.astype(np.int16)
                    )

                # Save metadata
                with open(out_meta_dir / f'{case_name}.json', 'w') as f:
                    json.dump(metadata, f, indent=2)

            except Exception as e:
                print(f"\n❌ Error processing {case_name}: {e}")
                continue

        # Save preprocessing config
        config = {
            'global_mean': self.global_mean,
            'global_std': self.global_std,
            'clip_low': self.clip_low,
            'clip_high': self.clip_high,
            'target_spacing': self.target_spacing.tolist(),
            'crop': crop,
        }

        with open(output_dir / 'preprocessing_config.json', 'w') as f:
            json.dump(config, f, indent=2)

        print("\n" + "="*60)
        print(f"✅ Preprocessing complete!")
        print(f"   Output directory: {output_dir}")
        print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess dataset")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to AMOS22 dataset")
    parser.add_argument("--stats_file", type=str, required=True, help="Path to dataset_statistics.json")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], help="Which split to preprocess")
    parser.add_argument("--no_crop", action="store_true", help="Disable cropping")

    args = parser.parse_args()

    preprocessor = NNUNetPreprocessor(args.stats_file)
    preprocessor.preprocess_dataset(
        args.data_dir,
        args.output_dir,
        split=args.split,
        crop=not args.no_crop
    )
