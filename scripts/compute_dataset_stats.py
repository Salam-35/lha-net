#!/usr/bin/env python3
"""
Compute global dataset statistics for nnU-Net style preprocessing.
Run this ONCE before training.

Usage:
    python scripts/compute_dataset_stats.py \
        --data_dir data/AMOS22/imagesTr \
        --label_dir data/AMOS22/labelsTr \
        --output dataset_statistics.json
"""

import os
import json
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


def compute_dataset_statistics(data_dir: str, label_dir: str, output_path: str):
    """
    Compute global statistics following nnU-Net methodology.

    Statistics computed:
    1. Global foreground mean and std (for z-score normalization)
    2. Intensity percentiles (for clipping)
    3. Median spacing (for resampling target)
    4. Per-organ volume statistics
    """

    data_dir = Path(data_dir)
    label_dir = Path(label_dir)

    # Collect all image files
    image_files = sorted(list(data_dir.glob("*.nii.gz")))

    if len(image_files) == 0:
        raise ValueError(f"No .nii.gz files found in {data_dir}")

    print(f"Found {len(image_files)} images")
    print("="*60)

    # Storage for statistics
    all_foreground_intensities = []
    all_spacings = []
    all_sizes = []
    organ_volumes = defaultdict(list)

    # Process each image
    for img_path in tqdm(image_files, desc="Computing statistics"):
        # Load image
        img_nib = nib.load(img_path)
        image = img_nib.get_fdata().astype(np.float32)
        spacing = np.array(img_nib.header.get_zooms()[:3])

        # Load corresponding label
        label_name = img_path.name  # Assumes same naming
        label_path = label_dir / label_name

        if label_path.exists():
            label_nib = nib.load(label_path)
            label = label_nib.get_fdata().astype(np.int32)
        else:
            # If no label, use intensity-based foreground
            label = (image > -500).astype(np.int32)

        # Get foreground mask (any organ present)
        foreground_mask = label > 0

        if foreground_mask.sum() > 0:
            # Collect foreground intensities
            foreground_intensities = image[foreground_mask]

            # Subsample to avoid memory issues (max 10M voxels per image)
            if len(foreground_intensities) > 10_000_000:
                indices = np.random.choice(
                    len(foreground_intensities),
                    10_000_000,
                    replace=False
                )
                foreground_intensities = foreground_intensities[indices]

            all_foreground_intensities.append(foreground_intensities)

        # Collect spacing
        all_spacings.append(spacing)
        all_sizes.append(image.shape)

        # Compute per-organ volumes
        if label_path.exists():
            unique_organs = np.unique(label)
            for organ_id in unique_organs:
                if organ_id == 0:  # Skip background
                    continue
                organ_mask = label == organ_id
                # Volume in mm³
                voxel_volume = np.prod(spacing)
                organ_volume_mm3 = organ_mask.sum() * voxel_volume
                organ_volumes[int(organ_id)].append(organ_volume_mm3)

    # Concatenate all foreground intensities
    print("\nComputing intensity statistics...")
    all_intensities = np.concatenate(all_foreground_intensities)

    # Compute statistics
    statistics = {
        # Intensity statistics
        "intensity": {
            "global_mean": float(np.mean(all_intensities)),
            "global_std": float(np.std(all_intensities)),
            "global_median": float(np.median(all_intensities)),
            "percentile_0_5": float(np.percentile(all_intensities, 0.5)),
            "percentile_99_5": float(np.percentile(all_intensities, 99.5)),
            "percentile_1": float(np.percentile(all_intensities, 1)),
            "percentile_99": float(np.percentile(all_intensities, 99)),
            "min": float(np.min(all_intensities)),
            "max": float(np.max(all_intensities)),
        },

        # Spacing statistics
        "spacing": {
            "median": [float(x) for x in np.median(all_spacings, axis=0)],
            "mean": [float(x) for x in np.mean(all_spacings, axis=0)],
            "min": [float(x) for x in np.min(all_spacings, axis=0)],
            "max": [float(x) for x in np.max(all_spacings, axis=0)],
            "percentile_10": [float(x) for x in np.percentile(all_spacings, 10, axis=0)],
        },

        # Size statistics
        "size": {
            "median": [int(x) for x in np.median(all_sizes, axis=0)],
            "mean": [int(x) for x in np.mean(all_sizes, axis=0)],
            "min": [int(x) for x in np.min(all_sizes, axis=0)],
            "max": [int(x) for x in np.max(all_sizes, axis=0)],
        },

        # Organ volume statistics (mm³)
        "organ_volumes": {
            str(organ_id): {
                "mean": float(np.mean(volumes)),
                "std": float(np.std(volumes)),
                "median": float(np.median(volumes)),
                "min": float(np.min(volumes)),
                "max": float(np.max(volumes)),
            }
            for organ_id, volumes in organ_volumes.items()
        },

        # Dataset info
        "dataset_info": {
            "num_cases": len(image_files),
            "num_foreground_voxels": int(len(all_intensities)),
        }
    }

    # Save statistics
    with open(output_path, 'w') as f:
        json.dump(statistics, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("DATASET STATISTICS SUMMARY")
    print("="*60)

    print(f"\n📊 Intensity Statistics:")
    print(f"   Global Mean: {statistics['intensity']['global_mean']:.2f}")
    print(f"   Global Std: {statistics['intensity']['global_std']:.2f}")
    print(f"   Clip Range (0.5-99.5%): [{statistics['intensity']['percentile_0_5']:.2f}, {statistics['intensity']['percentile_99_5']:.2f}]")

    print(f"\n📏 Spacing Statistics:")
    print(f"   Median Spacing: {statistics['spacing']['median']}")
    print(f"   → Use this as target_spacing!")

    print(f"\n📦 Size Statistics:")
    print(f"   Median Size: {statistics['size']['median']}")

    print(f"\n✅ Statistics saved to: {output_path}")
    print("="*60)

    return statistics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute dataset statistics")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to images")
    parser.add_argument("--label_dir", type=str, required=True, help="Path to labels")
    parser.add_argument("--output", type=str, default="dataset_statistics.json", help="Output file")

    args = parser.parse_args()

    compute_dataset_statistics(args.data_dir, args.label_dir, args.output)
