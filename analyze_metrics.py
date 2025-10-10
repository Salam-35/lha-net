#!/usr/bin/env python3
"""
Analyze and visualize training metrics from checkpoint files
"""

import torch
import argparse
import json
from pathlib import Path
import numpy as np


def load_checkpoint_metrics(checkpoint_path):
    """Load metrics from a checkpoint file"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return checkpoint.get('metrics_history', None), checkpoint.get('current_epoch_metrics', None)


def print_metrics_summary(checkpoint_path):
    """Print comprehensive metrics summary from checkpoint"""
    print(f"\nAnalyzing checkpoint: {checkpoint_path}")
    print("="*100)

    metrics_history, current_epoch_metrics = load_checkpoint_metrics(checkpoint_path)

    if metrics_history is None:
        print("No metrics history found in checkpoint!")
        return

    # Training metrics
    if metrics_history['train']['epochs']:
        print("\n" + "="*100)
        print("TRAINING METRICS HISTORY")
        print("="*100)
        print(f"\n{'Epoch':<8} {'Total Loss':<12} {'Focal Loss':<12} {'Dice Loss':<12} {'Size Loss':<12} {'Routing Loss':<12}")
        print("-"*100)

        for i, epoch in enumerate(metrics_history['train']['epochs']):
            total_loss = metrics_history['train']['losses'][i]
            loss_comp = metrics_history['train']['loss_components'][i]

            focal = loss_comp.get('focal_loss', 0.0)
            dice = loss_comp.get('dice_loss', 0.0)
            size = loss_comp.get('size_loss', 0.0)
            routing = loss_comp.get('routing_loss', 0.0)

            print(f"{epoch:<8} {total_loss:<12.4f} {focal:<12.4f} {dice:<12.4f} {size:<12.4f} {routing:<12.4f}")

    # Validation metrics
    if metrics_history['val']['epochs']:
        print("\n" + "="*100)
        print("VALIDATION METRICS HISTORY")
        print("="*100)
        print(f"\n{'Epoch':<8} {'Total Loss':<12} {'Mean Dice':<12}")
        print("-"*100)

        for i, epoch in enumerate(metrics_history['val']['epochs']):
            val_loss = metrics_history['val']['losses'][i]
            val_dice = metrics_history['val']['dice_scores'][i]

            print(f"{epoch:<8} {val_loss:<12.4f} {val_dice:<12.4f}")

        # Per-organ metrics for each validation epoch
        print("\n" + "="*100)
        print("PER-ORGAN METRICS (for each validation epoch)")
        print("="*100)

        for i, epoch in enumerate(metrics_history['val']['epochs']):
            organ_metrics = metrics_history['val']['per_organ_metrics'][i]

            print(f"\nEpoch {epoch}:")
            print(f"  {'Organ':<20} {'Dice (mean±std)':<20} {'HD95 (mean±std)':<20} {'NSD (mean±std)':<20}")
            print(f"  {'-'*20} {'-'*20} {'-'*20} {'-'*20}")

            for organ, metrics in organ_metrics.items():
                if organ == 'background':
                    continue

                dice_mean = metrics.get('dice_mean', 0.0)
                dice_std = metrics.get('dice_std', 0.0)
                dice_str = f"{dice_mean:.4f}±{dice_std:.4f}"

                if 'hd95_mean' in metrics:
                    hd95_mean = metrics['hd95_mean']
                    hd95_std = metrics['hd95_std']
                    hd95_str = f"{hd95_mean:.2f}±{hd95_std:.2f}"
                else:
                    hd95_str = "N/A"

                if 'nsd_mean' in metrics:
                    nsd_mean = metrics['nsd_mean']
                    nsd_std = metrics['nsd_std']
                    nsd_str = f"{nsd_mean:.4f}±{nsd_std:.4f}"
                else:
                    nsd_str = "N/A"

                print(f"  {organ:<20} {dice_str:<20} {hd95_str:<20} {nsd_str:<20}")

        # Best performance summary
        best_epoch_idx = np.argmax(metrics_history['val']['dice_scores'])
        best_epoch = metrics_history['val']['epochs'][best_epoch_idx]
        best_dice = metrics_history['val']['dice_scores'][best_epoch_idx]
        best_organ_metrics = metrics_history['val']['per_organ_metrics'][best_epoch_idx]

        print("\n" + "="*100)
        print(f"BEST PERFORMANCE (Epoch {best_epoch})")
        print("="*100)
        print(f"Mean Dice Score: {best_dice:.4f}")

        print(f"\nPer-Organ Performance (sorted by Dice score):")
        print(f"  {'Organ':<20} {'Dice':<20} {'HD95 (mm)':<20} {'NSD':<20}")
        print(f"  {'-'*20} {'-'*20} {'-'*20} {'-'*20}")

        sorted_organs = sorted(best_organ_metrics.items(),
                              key=lambda x: x[1].get('dice_mean', 0),
                              reverse=True)

        for organ, metrics in sorted_organs:
            if organ == 'background':
                continue

            dice_mean = metrics.get('dice_mean', 0.0)
            dice_std = metrics.get('dice_std', 0.0)
            dice_str = f"{dice_mean:.4f}±{dice_std:.4f}"

            if 'hd95_mean' in metrics:
                hd95_mean = metrics['hd95_mean']
                hd95_std = metrics['hd95_std']
                hd95_str = f"{hd95_mean:.2f}±{hd95_std:.2f}"
            else:
                hd95_str = "N/A"

            if 'nsd_mean' in metrics:
                nsd_mean = metrics['nsd_mean']
                nsd_std = metrics['nsd_std']
                nsd_str = f"{nsd_mean:.4f}±{nsd_std:.4f}"
            else:
                nsd_str = "N/A"

            print(f"  {organ:<20} {dice_str:<20} {hd95_str:<20} {nsd_str:<20}")


def export_metrics_to_json(checkpoint_path, output_path):
    """Export metrics to JSON file"""
    metrics_history, current_epoch_metrics = load_checkpoint_metrics(checkpoint_path)

    if metrics_history is None:
        print("No metrics history found in checkpoint!")
        return

    # Convert numpy types to native Python types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        return obj

    metrics_data = {
        'metrics_history': convert_to_native(metrics_history),
        'current_epoch_metrics': convert_to_native(current_epoch_metrics)
    }

    with open(output_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)

    print(f"Metrics exported to: {output_path}")


def compare_checkpoints(checkpoint_paths):
    """Compare metrics from multiple checkpoints"""
    print("\n" + "="*100)
    print("CHECKPOINT COMPARISON")
    print("="*100)

    results = []
    for path in checkpoint_paths:
        metrics_history, _ = load_checkpoint_metrics(path)
        if metrics_history and metrics_history['val']['dice_scores']:
            best_dice = max(metrics_history['val']['dice_scores'])
            last_dice = metrics_history['val']['dice_scores'][-1]
            results.append({
                'path': path,
                'best_dice': best_dice,
                'last_dice': last_dice,
                'num_epochs': len(metrics_history['train']['epochs'])
            })

    if not results:
        print("No valid metrics found in checkpoints!")
        return

    print(f"\n{'Checkpoint':<40} {'Best Dice':<15} {'Last Dice':<15} {'Epochs':<10}")
    print("-"*100)

    for result in sorted(results, key=lambda x: x['best_dice'], reverse=True):
        path_str = str(Path(result['path']).name)
        print(f"{path_str:<40} {result['best_dice']:<15.4f} {result['last_dice']:<15.4f} {result['num_epochs']:<10}")


def main():
    parser = argparse.ArgumentParser(description='Analyze training metrics from checkpoint')
    parser.add_argument('checkpoint', type=str, nargs='+',
                       help='Path to checkpoint file(s)')
    parser.add_argument('--export', type=str, default=None,
                       help='Export metrics to JSON file')
    parser.add_argument('--compare', action='store_true',
                       help='Compare multiple checkpoints')

    args = parser.parse_args()

    if args.compare and len(args.checkpoint) > 1:
        compare_checkpoints(args.checkpoint)
    else:
        checkpoint_path = args.checkpoint[0]
        print_metrics_summary(checkpoint_path)

        if args.export:
            export_metrics_to_json(checkpoint_path, args.export)


if __name__ == '__main__':
    main()
