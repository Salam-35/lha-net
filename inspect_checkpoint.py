#!/usr/bin/env python3
"""
Inspect checkpoint file contents and extract training history
"""

import torch
import argparse
import json
from pathlib import Path
import numpy as np


def inspect_checkpoint(checkpoint_path):
    """Load and inspect checkpoint contents"""

    print("="*80)
    print(f"CHECKPOINT INSPECTION: {checkpoint_path}")
    print("="*80)

    if not Path(checkpoint_path).exists():
        print(f"\n✗ Checkpoint file not found: {checkpoint_path}")
        return None

    # Load checkpoint
    print("\nLoading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Basic info
    print("\n" + "="*80)
    print("BASIC INFORMATION")
    print("="*80)
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Best Dice Score: {checkpoint.get('best_dice', 'N/A'):.4f}")

    # Keys in checkpoint
    print("\n" + "="*80)
    print("CHECKPOINT KEYS")
    print("="*80)
    for key in checkpoint.keys():
        if key in ['model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict']:
            print(f"  ✓ {key}: <state dict>")
        elif key == 'config':
            print(f"  ✓ {key}: <configuration dict>")
        elif key == 'metrics_history':
            print(f"  ✓ {key}: <training history>")
        else:
            print(f"  ✓ {key}: {checkpoint[key]}")

    # Model info
    if 'model_state_dict' in checkpoint:
        print("\n" + "="*80)
        print("MODEL INFORMATION")
        print("="*80)
        state_dict = checkpoint['model_state_dict']
        total_params = sum(p.numel() for p in state_dict.values())
        print(f"Total parameters: {total_params:,}")
        print(f"Model size: {total_params * 4 / (1024**3):.2f} GB (float32)")

    # Config info
    if 'config' in checkpoint:
        config = checkpoint['config']
        print("\n" + "="*80)
        print("TRAINING CONFIGURATION")
        print("="*80)
        print(f"Model name: {config.get('model', {}).get('name', 'N/A')}")
        print(f"Number of classes: {config.get('model', {}).get('num_classes', 'N/A')}")
        print(f"Batch size: {config.get('training', {}).get('batch_size', 'N/A')}")
        print(f"Learning rate: {config.get('training', {}).get('learning_rate', 'N/A')}")
        print(f"Patch size: {config.get('data', {}).get('patch_size', 'N/A')}")

    # Metrics history
    if 'metrics_history' in checkpoint:
        metrics = checkpoint['metrics_history']

        print("\n" + "="*80)
        print("TRAINING HISTORY")
        print("="*80)

        # Training metrics
        if 'train' in metrics and metrics['train'].get('epochs'):
            train_epochs = metrics['train']['epochs']
            train_losses = metrics['train']['losses']
            print(f"\nTraining Epochs: {len(train_epochs)}")
            print(f"  First epoch: {train_epochs[0] if train_epochs else 'N/A'}")
            print(f"  Last epoch: {train_epochs[-1] if train_epochs else 'N/A'}")
            print(f"  Initial loss: {train_losses[0]:.4f}" if train_losses else "")
            print(f"  Final loss: {train_losses[-1]:.4f}" if train_losses else "")

        # Validation metrics
        if 'val' in metrics and metrics['val'].get('epochs'):
            val_epochs = metrics['val']['epochs']
            val_losses = metrics['val']['losses']
            val_dice = metrics['val']['dice_scores']
            print(f"\nValidation Epochs: {len(val_epochs)}")
            print(f"  First epoch: {val_epochs[0] if val_epochs else 'N/A'}")
            print(f"  Last epoch: {val_epochs[-1] if val_epochs else 'N/A'}")
            print(f"  Initial Dice: {val_dice[0]:.4f}" if val_dice else "")
            print(f"  Best Dice: {max(val_dice):.4f}" if val_dice else "")
            print(f"  Final Dice: {val_dice[-1]:.4f}" if val_dice else "")

    # Current epoch metrics
    if 'current_epoch_metrics' in checkpoint and checkpoint['current_epoch_metrics']:
        print("\n" + "="*80)
        print("LATEST EPOCH METRICS")
        print("="*80)
        current = checkpoint['current_epoch_metrics']
        for key, value in current.items():
            if isinstance(value, dict):
                print(f"\n{key}:")
                for k, v in value.items():
                    if isinstance(v, (int, float)):
                        print(f"  {k}: {v:.4f}")
                    else:
                        print(f"  {k}: {v}")
            elif isinstance(value, (int, float)):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")

    print("\n" + "="*80)

    return checkpoint


def extract_training_history(checkpoint_path, output_dir=None):
    """Extract training history to JSON and CSV files"""

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'metrics_history' not in checkpoint:
        print("No training history found in checkpoint")
        return

    metrics = checkpoint['metrics_history']

    # Determine output directory
    if output_dir is None:
        output_dir = Path(checkpoint_path).parent / 'training_history'
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"\nExtracting training history to: {output_dir}")

    # Save as JSON
    json_path = output_dir / 'training_history.json'

    # Convert numpy arrays to lists for JSON serialization
    serializable_metrics = {}
    for split in ['train', 'val']:
        if split in metrics:
            serializable_metrics[split] = {}
            for key, value in metrics[split].items():
                if isinstance(value, (list, np.ndarray)):
                    serializable_metrics[split][key] = [
                        float(v) if not np.isnan(v) else None
                        for v in value
                    ]
                else:
                    serializable_metrics[split][key] = value

    with open(json_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)
    print(f"  ✓ Saved JSON: {json_path}")

    # Save as CSV for easy viewing
    if 'train' in metrics and metrics['train'].get('epochs'):
        csv_path = output_dir / 'training_metrics.csv'
        with open(csv_path, 'w') as f:
            # Header
            f.write("epoch,train_loss,val_loss,val_dice\n")

            # Get all training epochs
            train_epochs = metrics['train']['epochs']
            train_losses = metrics['train']['losses']

            # Create dict for easy lookup
            val_data = {}
            if 'val' in metrics and metrics['val'].get('epochs'):
                val_epochs = metrics['val']['epochs']
                val_losses = metrics['val']['losses']
                val_dice = metrics['val']['dice_scores']
                for e, l, d in zip(val_epochs, val_losses, val_dice):
                    val_data[e] = (l, d)

            # Write rows
            for epoch, train_loss in zip(train_epochs, train_losses):
                val_loss, val_dice = val_data.get(epoch, ('', ''))
                f.write(f"{epoch},{train_loss:.6f},{val_loss},{val_dice}\n")

        print(f"  ✓ Saved CSV: {csv_path}")

    print(f"\n✓ Training history extracted successfully!")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description='Inspect checkpoint and extract training history')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--extract', action='store_true', help='Extract training history to files')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for extracted history')

    args = parser.parse_args()

    # Inspect checkpoint
    checkpoint = inspect_checkpoint(args.checkpoint)

    # Extract history if requested
    if args.extract and checkpoint is not None:
        print("\n" + "="*80)
        extract_training_history(args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
