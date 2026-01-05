#!/usr/bin/env python3
"""
Test/Evaluate model from checkpoint
"""

import os
import sys
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.lha_net_adaptive import LHANetAdaptive
from src.data.preprocessing import AMOSDataset
from torch.utils.data import DataLoader
from src.evaluation.metrics import SegmentationMetrics, compute_dice_score


def load_checkpoint(checkpoint_path, device='cuda'):
    """Load model from checkpoint"""

    print("="*80)
    print(f"LOADING CHECKPOINT: {checkpoint_path}")
    print("="*80)

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get config
    config = checkpoint['config']
    model_config = config['model']

    # Build model
    print("\nBuilding model...")
    model = LHANetAdaptive(
        in_channels=model_config['in_channels'],
        num_classes=model_config['num_classes'],
        backbone_type=model_config['backbone_type'],
        use_lightweight=model_config.get('use_lightweight', True),
        base_channels=model_config.get('base_channels', 32),
        scale_bank=model_config.get('scale_bank', [0.5, 0.75, 1.0, 1.5, 2.0]),
        num_active_scales=model_config.get('num_active_scales', 3),
        share_scale_selection=model_config.get('share_scale_selection', False),
        use_deep_supervision=model_config.get('use_deep_supervision', True)
    ).to(device)

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✓ Model loaded from epoch {checkpoint['epoch']}")
    print(f"✓ Best Dice Score: {checkpoint.get('best_dice', 0):.4f}")

    return model, config


def evaluate_model(checkpoint_path, data_root=None, split='val', device='cuda', save_predictions=False):
    """Evaluate model on validation/test set"""

    # Load model
    model, config = load_checkpoint(checkpoint_path, device)

    # Setup data path
    if data_root is None:
        data_root = config['paths']['data_root']

    print(f"\n{'='*80}")
    print(f"EVALUATING ON {split.upper()} SET")
    print('='*80)

    # Load dataset
    print(f"\nLoading {split} dataset from: {data_root}")
    dataset = AMOSDataset(
        data_root=data_root,
        split=split,
        patch_size=config['data']['patch_size'],
        num_classes=config['model']['num_classes'],
        augmentation=False  # No augmentation for testing
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Use batch size 1 for testing
        shuffle=False,
        num_workers=config['system'].get('num_workers', 2),
        pin_memory=config['system'].get('pin_memory', True)
    )

    print(f"Number of samples: {len(dataset)}")
    print(f"Number of batches: {len(dataloader)}")

    # Initialize metrics
    num_classes = config['model']['num_classes']

    # Get organ names from config
    organ_names = []
    if 'labels' in config:
        label_dict = config['labels']
        # Sort by key to get proper order
        sorted_labels = sorted(label_dict.items(), key=lambda x: int(x[0]))
        organ_names = [v for k, v in sorted_labels]
    else:
        # Default names
        organ_names = [f'class_{i}' for i in range(num_classes)]

    metrics_calculator = SegmentationMetrics(
        num_classes=num_classes,
        organ_names=organ_names
    )

    # Storage for predictions
    all_predictions = []
    all_labels = []

    # Evaluation loop
    print("\nRunning evaluation...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            outputs = model(images)

            # Get predictions
            if isinstance(outputs, dict):
                predictions = outputs['final_prediction']
            else:
                predictions = outputs

            # Convert to predictions (argmax)
            pred_masks = torch.argmax(predictions, dim=1)

            # Store for metrics
            all_predictions.append(pred_masks.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Concatenate all predictions
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    print(f"\nTotal samples evaluated: {all_predictions.shape[0]}")

    # Compute metrics
    print("\nComputing metrics...")

    # Compute Dice scores
    dice_scores = []
    per_organ_dice = {organ: [] for organ in organ_names}

    for i in range(all_predictions.shape[0]):
        pred = all_predictions[i]
        label = all_labels[i]

        # Per-organ dice
        sample_dice_scores = []
        for c in range(num_classes):
            pred_c = (pred == c).astype(np.float32)
            label_c = (label == c).astype(np.float32)

            # Only compute if this organ is present in ground truth
            if np.sum(label_c) > 0:
                organ_dice = compute_dice_score(pred_c, label_c, ignore_background=False)
                per_organ_dice[organ_names[c]].append(organ_dice)

                # Also add to sample scores (only for present organs, excluding background)
                if c > 0:  # Skip background
                    sample_dice_scores.append(organ_dice)

        # Overall dice: average only over organs present in this sample
        if len(sample_dice_scores) > 0:
            dice = np.mean(sample_dice_scores)
            dice_scores.append(dice)

    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)

    mean_dice = np.mean(dice_scores)
    print(f"\nOverall Mean Dice Score: {mean_dice:.4f} ± {np.std(dice_scores):.4f}")

    print(f"\nPer-Organ Dice Scores:")
    print(f"  {'Organ':<20} {'Mean':<12} {'Std':<12} {'Samples':<12}")
    print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12}")

    for organ in organ_names:
        if organ != 'background' and per_organ_dice[organ]:
            scores = per_organ_dice[organ]
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            num_samples = len(scores)
            print(f"  {organ:<20} {mean_score:.4f}       {std_score:.4f}       {num_samples:<12}")

    print("="*80)

    # Save predictions if requested
    if save_predictions:
        output_dir = Path(checkpoint_path).parent / 'predictions'
        output_dir.mkdir(exist_ok=True, parents=True)

        pred_path = output_dir / f'{split}_predictions.npz'
        np.savez_compressed(
            pred_path,
            predictions=all_predictions,
            labels=all_labels,
            dice_scores=dice_scores
        )
        print(f"\n✓ Predictions saved to: {pred_path}")

    return mean_dice, per_organ_dice


def main():
    parser = argparse.ArgumentParser(description='Test/Evaluate model from checkpoint')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--data-root', type=str, default=None, help='Path to data root (overrides config)')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate on')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--save-predictions', action='store_true', help='Save predictions to file')

    args = parser.parse_args()

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    # Run evaluation
    mean_dice, per_organ_dice = evaluate_model(
        args.checkpoint,
        data_root=args.data_root,
        split=args.split,
        device=args.device,
        save_predictions=args.save_predictions
    )

    print(f"\n✓ Evaluation complete!")
    print(f"  Mean Dice: {mean_dice:.4f}")


if __name__ == "__main__":
    main()
