#!/usr/bin/env python3
"""
LHA-Net Inference Script
Run predictions on new medical images using trained LHA-Net model
"""

import os
import sys
import yaml
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
import argparse
from tqdm import tqdm
from typing import Union, List, Tuple
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.lha_net import LHANet
from src.utils.memory_utils import MemoryMonitor


class LHANetPredictor:
    """Inference engine for LHA-Net"""

    def __init__(self, config_path: str, checkpoint_path: str, device: str = 'auto'):
        """
        Initialize predictor

        Args:
            config_path: Path to configuration YAML file
            checkpoint_path: Path to trained model checkpoint
            device: Device to use ('cuda', 'cpu', or 'auto')
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Setup device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        print(f"Using device: {self.device}")

        # Load model
        self.model = self.build_model()
        self.load_checkpoint(checkpoint_path)

        # Get preprocessing parameters
        self.target_spacing = self.config['data']['preprocessing']['target_spacing']
        self.clip_range = self.config['data']['preprocessing']['clip_range']
        self.patch_size = self.config['data']['patch_size']
        self.overlap_ratio = self.config['data']['overlap_ratio']

        # Organ names for labeling
        self.organ_names = [
            'background', 'liver', 'right_kidney', 'spleen', 'pancreas',
            'aorta', 'ivc', 'right_adrenal', 'left_adrenal', 'gallbladder',
            'esophagus', 'stomach', 'duodenum', 'left_kidney', 'class_14', 'class_15'
        ]

        # Memory monitor
        self.memory_monitor = MemoryMonitor(self.device)

    def build_model(self) -> torch.nn.Module:
        """Build LHA-Net model"""
        model_config = self.config['model']

        model = LHANet(
            in_channels=model_config['in_channels'],
            num_classes=model_config['num_classes'],
            base_channels=model_config['base_channels'],
            use_lightweight=model_config['use_lightweight'],
            use_deep_supervision=model_config['use_deep_supervision'],
            pmsa_scales=model_config['pmsa_scales'],
            memory_efficient=model_config['memory_efficient']
        )

        model = model.to(self.device)
        model.eval()

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model: LHA-Net ({model_config['type']})")
        print(f"Total parameters: {total_params:,}")

        return model

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        print(f"\nLoading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Print checkpoint info
        if 'epoch' in checkpoint:
            print(f"Checkpoint epoch: {checkpoint['epoch']}")
        if 'best_dice' in checkpoint:
            print(f"Checkpoint best Dice: {checkpoint['best_dice']:.4f}")

        print("Model loaded successfully!\n")

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess medical image

        Args:
            image: Input image array (D, H, W)

        Returns:
            Preprocessed image array
        """
        # Clip intensity values
        image = np.clip(image, self.clip_range[0], self.clip_range[1])

        # Z-score normalization
        mean = np.mean(image)
        std = np.std(image)
        if std > 0:
            image = (image - mean) / std
        else:
            image = image - mean

        return image

    def extract_patches(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[Tuple[int, ...]]]:
        """
        Extract overlapping patches from image

        Args:
            image: Input image (D, H, W)

        Returns:
            patches: List of patches
            positions: List of patch positions (d_start, h_start, w_start)
        """
        d, h, w = image.shape
        pd, ph, pw = self.patch_size

        # Calculate stride based on overlap ratio
        stride_d = int(pd * (1 - self.overlap_ratio))
        stride_h = int(ph * (1 - self.overlap_ratio))
        stride_w = int(pw * (1 - self.overlap_ratio))

        patches = []
        positions = []

        # Extract patches with overlap
        for d_start in range(0, d - pd + 1, stride_d):
            for h_start in range(0, h - ph + 1, stride_h):
                for w_start in range(0, w - pw + 1, stride_w):
                    patch = image[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
                    patches.append(patch)
                    positions.append((d_start, h_start, w_start))

        # Handle remaining regions (edge cases)
        # Add patches for remaining depth
        if d % stride_d != 0 and d > pd:
            d_start = d - pd
            for h_start in range(0, h - ph + 1, stride_h):
                for w_start in range(0, w - pw + 1, stride_w):
                    patch = image[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
                    patches.append(patch)
                    positions.append((d_start, h_start, w_start))

        # Add patches for remaining height
        if h % stride_h != 0 and h > ph:
            h_start = h - ph
            for d_start in range(0, d - pd + 1, stride_d):
                for w_start in range(0, w - pw + 1, stride_w):
                    patch = image[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
                    patches.append(patch)
                    positions.append((d_start, h_start, w_start))

        # Add patches for remaining width
        if w % stride_w != 0 and w > pw:
            w_start = w - pw
            for d_start in range(0, d - pd + 1, stride_d):
                for h_start in range(0, h - ph + 1, stride_h):
                    patch = image[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
                    patches.append(patch)
                    positions.append((d_start, h_start, w_start))

        return patches, positions

    def reconstruct_from_patches(self,
                                  predictions: List[np.ndarray],
                                  positions: List[Tuple[int, ...]],
                                  image_shape: Tuple[int, ...],
                                  num_classes: int) -> np.ndarray:
        """
        Reconstruct full segmentation from patch predictions using weighted averaging

        Args:
            predictions: List of patch predictions (each is [C, D, H, W])
            positions: List of patch positions
            image_shape: Original image shape (D, H, W)
            num_classes: Number of classes

        Returns:
            Full segmentation mask (D, H, W)
        """
        d, h, w = image_shape
        pd, ph, pw = self.patch_size

        # Initialize accumulation arrays
        prediction_sum = np.zeros((num_classes, d, h, w), dtype=np.float32)
        weight_sum = np.zeros((d, h, w), dtype=np.float32)

        # Create Gaussian weight map for smooth blending
        weight_map = self._create_gaussian_weight_map(self.patch_size)

        # Accumulate predictions with weights
        for pred, (d_start, h_start, w_start) in zip(predictions, positions):
            prediction_sum[:, d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw] += pred * weight_map
            weight_sum[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw] += weight_map

        # Avoid division by zero
        weight_sum = np.maximum(weight_sum, 1e-8)

        # Compute weighted average
        final_prediction = prediction_sum / weight_sum[np.newaxis, :, :, :]

        # Get final segmentation
        segmentation = np.argmax(final_prediction, axis=0).astype(np.uint8)

        return segmentation

    def _create_gaussian_weight_map(self, patch_size: Tuple[int, ...]) -> np.ndarray:
        """Create Gaussian weight map for patch blending"""
        pd, ph, pw = patch_size

        # Create 1D Gaussian curves
        def gaussian_1d(length):
            sigma = length / 4
            x = np.arange(length)
            gaussian = np.exp(-((x - length/2) ** 2) / (2 * sigma ** 2))
            return gaussian

        # Create 3D Gaussian weight map
        weight_d = gaussian_1d(pd)
        weight_h = gaussian_1d(ph)
        weight_w = gaussian_1d(pw)

        weight_map = weight_d[:, np.newaxis, np.newaxis] * \
                     weight_h[np.newaxis, :, np.newaxis] * \
                     weight_w[np.newaxis, np.newaxis, :]

        return weight_map

    def predict(self, image: np.ndarray, batch_size: int = 4) -> np.ndarray:
        """
        Run prediction on full image

        Args:
            image: Input image (D, H, W)
            batch_size: Batch size for patch processing

        Returns:
            Segmentation mask (D, H, W)
        """
        original_shape = image.shape

        # Preprocess
        image = self.preprocess_image(image)

        # Extract patches
        patches, positions = self.extract_patches(image)
        print(f"Extracted {len(patches)} patches from image")

        # Run inference on patches
        all_predictions = []

        with torch.no_grad():
            for i in tqdm(range(0, len(patches), batch_size), desc="Processing patches"):
                batch_patches = patches[i:i+batch_size]

                # Convert to tensor
                batch_tensor = torch.from_numpy(
                    np.array(batch_patches)[:, np.newaxis, :, :, :]  # Add channel dimension
                ).float().to(self.device)

                # Forward pass
                outputs = self.model(batch_tensor)

                # Get predictions
                if isinstance(outputs, dict):
                    predictions = outputs['final_prediction']
                else:
                    predictions = outputs

                # Convert to numpy and store
                predictions = predictions.cpu().numpy()
                all_predictions.extend([predictions[j] for j in range(predictions.shape[0])])

        # Reconstruct full segmentation
        segmentation = self.reconstruct_from_patches(
            all_predictions,
            positions,
            original_shape,
            self.config['model']['num_classes']
        )

        return segmentation

    def predict_from_file(self,
                          image_path: str,
                          output_path: str = None,
                          save_probabilities: bool = False) -> np.ndarray:
        """
        Run prediction on NIfTI file

        Args:
            image_path: Path to input NIfTI file
            output_path: Path to save output segmentation (optional)
            save_probabilities: Whether to save probability maps

        Returns:
            Segmentation mask
        """
        print(f"\nProcessing: {image_path}")

        # Load image
        nii_img = nib.load(image_path)
        image = nii_img.get_fdata()
        print(f"Image shape: {image.shape}")

        # Run prediction
        segmentation = self.predict(image, batch_size=self.config['training']['batch_size'])

        # Save output
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save segmentation
            seg_nii = nib.Nifti1Image(segmentation, nii_img.affine, nii_img.header)
            nib.save(seg_nii, output_path)
            print(f"Saved segmentation: {output_path}")

            # Save organ statistics
            stats = self.compute_segmentation_stats(segmentation)
            stats_path = output_path.parent / (output_path.stem + '_stats.json')
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"Saved statistics: {stats_path}")

        return segmentation

    def compute_segmentation_stats(self, segmentation: np.ndarray) -> dict:
        """Compute statistics from segmentation"""
        stats = {
            'volume_shape': list(segmentation.shape),
            'organ_volumes': {},
            'organ_voxel_counts': {}
        }

        # Compute volume for each organ
        for i, organ_name in enumerate(self.organ_names):
            if i >= self.config['model']['num_classes']:
                break

            voxel_count = np.sum(segmentation == i)
            # Assuming isotropic spacing for volume calculation
            spacing = np.array(self.target_spacing)
            volume_mm3 = voxel_count * np.prod(spacing)

            stats['organ_voxel_counts'][organ_name] = int(voxel_count)
            stats['organ_volumes'][organ_name] = float(volume_mm3)

        return stats

    def predict_batch(self,
                      image_paths: List[str],
                      output_dir: str,
                      save_probabilities: bool = False):
        """
        Run prediction on multiple files

        Args:
            image_paths: List of input file paths
            output_dir: Output directory
            save_probabilities: Whether to save probability maps
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing {len(image_paths)} files...")
        print("="*80)

        for image_path in image_paths:
            try:
                # Generate output path
                image_name = Path(image_path).stem
                output_path = output_dir / f"{image_name}_segmentation.nii.gz"

                # Run prediction
                self.predict_from_file(
                    image_path,
                    output_path,
                    save_probabilities=save_probabilities
                )

                print("-"*80)

            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue

        print("\nBatch processing completed!")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Run LHA-Net inference')
    parser.add_argument('--config', type=str, default='configs/lha_net_config.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                        help='Input image file or directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Output file or directory')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                        help='Device to use for inference')
    parser.add_argument('--save-probabilities', action='store_true',
                        help='Save probability maps in addition to segmentation')
    parser.add_argument('--batch-mode', action='store_true',
                        help='Process all NIfTI files in input directory')

    args = parser.parse_args()

    # Create predictor
    predictor = LHANetPredictor(args.config, args.checkpoint, device=args.device)

    # Run inference
    if args.batch_mode:
        # Process directory
        input_dir = Path(args.input)
        image_paths = list(input_dir.glob('*.nii.gz')) + list(input_dir.glob('*.nii'))

        if not image_paths:
            print(f"No NIfTI files found in {input_dir}")
            return

        predictor.predict_batch(
            [str(p) for p in image_paths],
            args.output,
            save_probabilities=args.save_probabilities
        )
    else:
        # Process single file
        predictor.predict_from_file(
            args.input,
            args.output,
            save_probabilities=args.save_probabilities
        )


if __name__ == '__main__':
    main()
