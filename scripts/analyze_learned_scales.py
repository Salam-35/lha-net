#!/usr/bin/env python3
"""
Analyze learned scales after training
"""

import torch
import sys
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.models.lha_net_adaptive import create_lha_net_adaptive
from src.utils.scale_analysis import ScaleAnalyzer, analyze_scale_organ_correlation


def main():
    parser = argparse.ArgumentParser(description='Analyze learned scales')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--save_dir', type=str, default='analysis/final_scales',
                       help='Directory to save analysis results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--num_classes', type=int, default=16,
                       help='Number of classes in the model')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load model
    print("Loading model...")
    model = create_lha_net_adaptive(
        config_type="lightweight",
        num_classes=args.num_classes,
        use_adaptive_pmsa=True
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    if 'val_dice' in checkpoint:
        print(f"Validation Dice: {checkpoint['val_dice']:.4f}")

    # Create analyzer
    analyzer = ScaleAnalyzer(model, save_dir=args.save_dir)

    # Print summary
    print("\n" + "="*70)
    analyzer.print_scale_summary()

    # Compare with baseline
    baseline_scales = [0.5, 0.75, 1.0, 1.5, 2.0]  # Previous hardcoded scales
    analyzer.compare_with_baseline(baseline_scales=baseline_scales)

    # Visualize
    print("\nGenerating visualizations...")
    analyzer.plot_scale_distribution(epoch=checkpoint.get('epoch', None), show=False)

    # If scale history is available in checkpoint
    if 'scale_history' in checkpoint and checkpoint['scale_history']:
        print("Plotting scale evolution...")
        analyzer.plot_scale_evolution(checkpoint['scale_history'], show=False)

    print(f"\n✓ Analysis saved to {args.save_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
