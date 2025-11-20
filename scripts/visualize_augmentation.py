#!/usr/bin/env python3
"""
Visualize augmentation effects on sample patches.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import argparse

sys.path.insert(0, 'src')

from data.augmentation_heavy import get_training_augmentation


def visualize_augmentation(
    data_dir: str = "/media/salam/projects/amos22/data/amos_preprocessed",
    case_name: str = "amos_0001",
    num_augmentations: int = 6,
    output_path: str = "augmentation_visualization.png"
):
    """Generate visualization of augmentation effects"""

    # Load sample
    image_path = Path(data_dir) / "imagesTr" / f"{case_name}.npz"
    label_path = Path(data_dir) / "labelsTr" / f"{case_name}.npz"

    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}")
        print("\nPlease run preprocessing first:")
        print("  python scripts/compute_dataset_stats.py ...")
        print("  python scripts/preprocess_dataset.py ...")
        return

    image = np.load(image_path)['data']
    label = np.load(label_path)['data']

    print(f"Loaded case: {case_name}")
    print(f"  Image shape: {image.shape}")
    print(f"  Label shape: {label.shape}")

    # Get middle slice
    mid = image.shape[0] // 2

    # Extract a patch
    patch_size = (64, 128, 128)
    start_d = max(0, mid - patch_size[0] // 2)
    start_h = (image.shape[1] - patch_size[1]) // 2
    start_w = (image.shape[2] - patch_size[2]) // 2

    image_patch = image[start_d:start_d+patch_size[0],
                        start_h:start_h+patch_size[1],
                        start_w:start_w+patch_size[2]]
    label_patch = label[start_d:start_d+patch_size[0],
                        start_h:start_h+patch_size[1],
                        start_w:start_w+patch_size[2]]

    print(f"\nExtracted patch:")
    print(f"  Patch shape: {image_patch.shape}")

    # Get augmenter
    augmenter = get_training_augmentation()
    print(f"\nApplying heavy augmentation with:")
    print(f"  Rotation: ±30°")
    print(f"  Scaling: 0.85-1.25")
    print(f"  Elastic deformation: enabled")
    print(f"  Gamma, brightness, contrast: enabled")
    print(f"  Noise, blur, low-res simulation: enabled")

    # Create figure
    fig, axes = plt.subplots(2, num_augmentations + 1, figsize=(3*(num_augmentations+1), 6))

    # Original
    mid_patch = patch_size[0] // 2
    axes[0, 0].imshow(image_patch[mid_patch], cmap='gray')
    axes[0, 0].set_title('Original Image', fontsize=10)
    axes[0, 0].axis('off')

    axes[1, 0].imshow(label_patch[mid_patch], cmap='nipy_spectral', vmin=0, vmax=15)
    axes[1, 0].set_title('Original Label', fontsize=10)
    axes[1, 0].axis('off')

    # Augmented versions
    print(f"\nGenerating {num_augmentations} augmented versions...")
    for i in range(num_augmentations):
        aug_image, aug_label = augmenter(image_patch.copy(), label_patch.copy())

        axes[0, i+1].imshow(aug_image[mid_patch], cmap='gray')
        axes[0, i+1].set_title(f'Augmented {i+1}', fontsize=10)
        axes[0, i+1].axis('off')

        axes[1, i+1].imshow(aug_label[mid_patch], cmap='nipy_spectral', vmin=0, vmax=15)
        axes[1, i+1].set_title(f'Augmented {i+1}', fontsize=10)
        axes[1, i+1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved visualization to: {output_path}")

    # Also show if running interactively
    try:
        plt.show()
    except:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize augmentation effects")
    parser.add_argument("--data_dir", type=str,
                        default="/media/salam/projects/amos22/data/amos_preprocessed",
                        help="Path to preprocessed data")
    parser.add_argument("--case_name", type=str, default="amos_0001",
                        help="Case name to visualize")
    parser.add_argument("--num_augmentations", type=int, default=6,
                        help="Number of augmented versions to show")
    parser.add_argument("--output", type=str, default="augmentation_visualization.png",
                        help="Output file path")

    args = parser.parse_args()

    visualize_augmentation(
        data_dir=args.data_dir,
        case_name=args.case_name,
        num_augmentations=args.num_augmentations,
        output_path=args.output
    )
