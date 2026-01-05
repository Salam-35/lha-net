#!/usr/bin/env python3
"""
Plot training history from checkpoint
"""

import torch
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(checkpoint_path, output_dir=None, show=False):
    """Plot training curves from checkpoint"""

    print("="*80)
    print(f"PLOTTING TRAINING HISTORY: {checkpoint_path}")
    print("="*80)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'metrics_history' not in checkpoint:
        print("✗ No training history found in checkpoint")
        return

    metrics = checkpoint['metrics_history']

    # Determine output directory
    if output_dir is None:
        output_dir = Path(checkpoint_path).parent / 'plots'
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(exist_ok=True, parents=True)

    # Get data
    train_epochs = metrics.get('train', {}).get('epochs', [])
    train_losses = metrics.get('train', {}).get('losses', [])
    val_epochs = metrics.get('val', {}).get('epochs', [])
    val_losses = metrics.get('val', {}).get('losses', [])
    val_dice = metrics.get('val', {}).get('dice_scores', [])

    if not train_epochs:
        print("✗ No training data to plot")
        return

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot 1: Loss curves
    ax1 = axes[0]
    ax1.plot(train_epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    if val_epochs:
        ax1.plot(val_epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Dice score
    ax2 = axes[1]
    if val_dice:
        ax2.plot(val_epochs, val_dice, 'g-', label='Val Dice', linewidth=2, marker='o', markersize=4)
        ax2.axhline(y=max(val_dice), color='r', linestyle='--', alpha=0.5, label=f'Best: {max(val_dice):.4f}')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Dice Score', fontsize=12)
    ax2.set_title('Validation Dice Score', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

    plt.tight_layout()

    # Save figure
    plot_path = output_dir / 'training_curves.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved plot: {plot_path}")

    if show:
        plt.show()
    else:
        plt.close()

    # Create detailed figure with loss components
    if 'train' in metrics and 'loss_components' in metrics['train']:
        fig2, ax = plt.subplots(figsize=(12, 6))

        loss_components = metrics['train']['loss_components']

        # Plot each component
        for component, values in loss_components.items():
            if values and any(v > 0 for v in values):
                ax.plot(train_epochs, values, label=component, linewidth=2, marker='o', markersize=3)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss Value', fontsize=12)
        ax.set_title('Training Loss Components', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure
        plot_path = output_dir / 'loss_components.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved plot: {plot_path}")

        if show:
            plt.show()
        else:
            plt.close()

    print(f"\n✓ All plots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Plot training history from checkpoint')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for plots')
    parser.add_argument('--show', action='store_true', help='Show plots interactively')

    args = parser.parse_args()

    plot_training_history(args.checkpoint, args.output_dir, args.show)


if __name__ == "__main__":
    main()
