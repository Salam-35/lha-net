#!/usr/bin/env python3
"""
LHA-Net Testing/Evaluation Script
Evaluates trained LHA-Net model on test/validation data with comprehensive metrics
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
from datetime import datetime
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.lha_net import LHANet, create_lha_net
from src.data.preprocessing import AMOSDataset
from src.evaluation.metrics import compute_dice_score, SegmentationMetrics
from src.utils.memory_utils import MemoryMonitor


class ModelEvaluator:
    """Comprehensive model evaluation for LHA-Net"""

    def __init__(self, config_path, checkpoint_path, output_dir=None):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Setup device
        self.device = torch.device(self.config['system']['device'] if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Setup output directory
        if output_dir is None:
            output_dir = Path(self.config['paths']['output_dir']) / 'evaluation'
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load model and checkpoint
        self.model = self.build_model()
        self.load_checkpoint(checkpoint_path)

        # Initialize metrics calculator
        self.organ_names = [
            'background', 'liver', 'right_kidney', 'spleen', 'pancreas',
            'aorta', 'ivc', 'right_adrenal', 'left_adrenal', 'gallbladder',
            'esophagus', 'stomach', 'duodenum', 'left_kidney', 'class_14', 'class_15'
        ]
        self.metrics_calculator = SegmentationMetrics(
            num_classes=self.config['model']['num_classes'],
            spacing=tuple(self.config['data']['preprocessing']['target_spacing']),
            organ_names=self.organ_names
        )

        # Memory monitor
        self.memory_monitor = MemoryMonitor(self.device)

    def build_model(self):
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

        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model: LHA-Net ({model_config['type']})")
        print(f"Total parameters: {total_params:,}")

        return model

    def load_checkpoint(self, checkpoint_path):
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

        self.model.eval()
        print("Model loaded successfully!\n")

    def build_dataloader(self, split='val'):
        """Build dataloader for evaluation"""
        data_root = Path(self.config['paths']['data_root'])

        # Dataset
        dataset = AMOSDataset(
            data_root=data_root,
            split=split,
            patch_size=self.config['data']['patch_size'],
            num_classes=self.config['model']['num_classes'],
            augmentation=False  # No augmentation for testing
        )

        # DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['system']['num_workers'],
            pin_memory=self.config['system']['pin_memory']
        )

        print(f"Test samples: {len(dataset)}")
        print(f"Test batches: {len(dataloader)}\n")

        return dataloader

    def evaluate(self, dataloader, save_predictions=False):
        """Comprehensive evaluation with all metrics"""
        print("="*80)
        print("STARTING EVALUATION")
        print("="*80)

        # Storage for results
        all_results = []
        per_organ_dice = {organ: [] for organ in self.organ_names}
        per_organ_hd95 = {organ: [] for organ in self.organ_names[1:]}  # Exclude background
        per_organ_nsd = {organ: [] for organ in self.organ_names[1:]}
        per_organ_vs = {organ: [] for organ in self.organ_names[1:]}

        # For storing predictions if requested
        predictions_list = [] if save_predictions else None

        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Evaluating")

            for batch_idx, batch in enumerate(pbar):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)

                # Get case IDs if available
                case_ids = batch.get('case_id', [f'case_{batch_idx}_{i}' for i in range(images.shape[0])])

                # Forward pass
                outputs = self.model(images)

                # Get predictions
                if isinstance(outputs, dict):
                    predictions = outputs['final_prediction']
                else:
                    predictions = outputs

                # Get predicted masks
                pred_masks = torch.argmax(predictions, dim=1)

                # Evaluate each sample in batch
                for b in range(pred_masks.shape[0]):
                    case_id = case_ids[b] if isinstance(case_ids, list) else case_ids[b].item()

                    try:
                        # Compute all metrics
                        metrics = self.metrics_calculator.compute_all_metrics(
                            pred_masks[b],
                            labels[b],
                            case_id=str(case_id)
                        )

                        # Store per-organ Dice scores
                        for organ, dice_val in metrics['dice_scores'].items():
                            if organ in per_organ_dice:
                                per_organ_dice[organ].append(dice_val)

                        # Store HD95 scores (excluding inf values)
                        for organ, hd95_val in metrics['hd95_scores'].items():
                            if organ in per_organ_hd95 and not np.isinf(hd95_val):
                                per_organ_hd95[organ].append(hd95_val)

                        # Store NSD scores
                        for organ, nsd_val in metrics['nsd_scores'].items():
                            if organ in per_organ_nsd:
                                per_organ_nsd[organ].append(nsd_val)

                        # Store Volume Similarity scores
                        for organ, vs_val in metrics['volume_similarity_scores'].items():
                            if organ in per_organ_vs:
                                per_organ_vs[organ].append(vs_val)

                        # Store case result
                        case_result = {
                            'case_id': str(case_id),
                            'mean_dice': metrics['mean_dice'],
                            'median_dice': metrics['median_dice'],
                            **{f'dice_{organ}': metrics['dice_scores'].get(organ, 0.0)
                               for organ in self.organ_names},
                            **{f'hd95_{organ}': metrics['hd95_scores'].get(organ, np.inf)
                               for organ in self.organ_names[1:]},
                            **{f'nsd_{organ}': metrics['nsd_scores'].get(organ, 0.0)
                               for organ in self.organ_names[1:]},
                        }
                        all_results.append(case_result)

                        # Save predictions if requested
                        if save_predictions:
                            predictions_list.append({
                                'case_id': str(case_id),
                                'prediction': pred_masks[b].cpu().numpy(),
                                'ground_truth': labels[b].cpu().numpy()
                            })

                    except Exception as e:
                        print(f"\nWarning: Metrics calculation failed for {case_id}: {e}")
                        continue

                    # Update progress bar
                    pbar.set_postfix({'case': str(case_id)})

        # Aggregate results
        results_summary = self.aggregate_results(
            all_results, per_organ_dice, per_organ_hd95, per_organ_nsd, per_organ_vs
        )

        # Save results
        self.save_results(results_summary, all_results, predictions_list if save_predictions else None)

        # Print summary
        self.print_results(results_summary)

        return results_summary

    def aggregate_results(self, all_results, per_organ_dice, per_organ_hd95, per_organ_nsd, per_organ_vs):
        """Aggregate all evaluation results"""
        summary = {
            'num_cases': len(all_results),
            'overall_metrics': {},
            'per_organ_metrics': {},
            'organ_groups': {}
        }

        # Overall metrics
        if all_results:
            all_mean_dice = [r['mean_dice'] for r in all_results]
            summary['overall_metrics'] = {
                'mean_dice': float(np.mean(all_mean_dice)),
                'std_dice': float(np.std(all_mean_dice)),
                'median_dice': float(np.median(all_mean_dice)),
                'min_dice': float(np.min(all_mean_dice)),
                'max_dice': float(np.max(all_mean_dice))
            }

        # Per-organ metrics
        for organ in self.organ_names:
            if organ == 'background':
                continue

            organ_metrics = {}

            # Dice scores
            if organ in per_organ_dice and per_organ_dice[organ]:
                organ_metrics['dice_mean'] = float(np.mean(per_organ_dice[organ]))
                organ_metrics['dice_std'] = float(np.std(per_organ_dice[organ]))
                organ_metrics['dice_median'] = float(np.median(per_organ_dice[organ]))
                organ_metrics['dice_min'] = float(np.min(per_organ_dice[organ]))
                organ_metrics['dice_max'] = float(np.max(per_organ_dice[organ]))

            # HD95 scores
            if organ in per_organ_hd95 and per_organ_hd95[organ]:
                organ_metrics['hd95_mean'] = float(np.mean(per_organ_hd95[organ]))
                organ_metrics['hd95_std'] = float(np.std(per_organ_hd95[organ]))
                organ_metrics['hd95_median'] = float(np.median(per_organ_hd95[organ]))

            # NSD scores
            if organ in per_organ_nsd and per_organ_nsd[organ]:
                organ_metrics['nsd_mean'] = float(np.mean(per_organ_nsd[organ]))
                organ_metrics['nsd_std'] = float(np.std(per_organ_nsd[organ]))
                organ_metrics['nsd_median'] = float(np.median(per_organ_nsd[organ]))

            # Volume Similarity scores
            if organ in per_organ_vs and per_organ_vs[organ]:
                organ_metrics['vs_mean'] = float(np.mean(per_organ_vs[organ]))
                organ_metrics['vs_std'] = float(np.std(per_organ_vs[organ]))

            if organ_metrics:
                summary['per_organ_metrics'][organ] = organ_metrics

        # Organ group analysis (small, medium, large)
        organ_groups_config = self.config['evaluation']['organ_groups']
        for group_name, organ_indices in organ_groups_config.items():
            group_organs = [self.organ_names[i] for i in organ_indices if i < len(self.organ_names)]
            group_dice_scores = []

            for organ in group_organs:
                if organ in per_organ_dice and per_organ_dice[organ]:
                    group_dice_scores.extend(per_organ_dice[organ])

            if group_dice_scores:
                summary['organ_groups'][group_name] = {
                    'dice_mean': float(np.mean(group_dice_scores)),
                    'dice_std': float(np.std(group_dice_scores)),
                    'organs': group_organs
                }

        return summary

    def save_results(self, summary, all_results, predictions=None):
        """Save evaluation results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save summary as JSON
        summary_path = self.output_dir / f'evaluation_summary_{timestamp}.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved summary: {summary_path}")

        # Save detailed results as CSV
        if all_results:
            df = pd.DataFrame(all_results)
            csv_path = self.output_dir / f'detailed_results_{timestamp}.csv'
            df.to_csv(csv_path, index=False)
            print(f"Saved detailed results: {csv_path}")

        # Save predictions if available
        if predictions is not None:
            predictions_path = self.output_dir / f'predictions_{timestamp}.npz'
            np.savez_compressed(
                predictions_path,
                **{f"{p['case_id']}_pred": p['prediction'] for p in predictions},
                **{f"{p['case_id']}_gt": p['ground_truth'] for p in predictions}
            )
            print(f"Saved predictions: {predictions_path}")

    def print_results(self, summary):
        """Print evaluation results summary"""
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)

        # Overall metrics
        print(f"\nOVERALL METRICS (n={summary['num_cases']} cases):")
        overall = summary['overall_metrics']
        print(f"  Mean Dice Score: {overall['mean_dice']:.4f} ± {overall['std_dice']:.4f}")
        print(f"  Median Dice Score: {overall['median_dice']:.4f}")
        print(f"  Range: [{overall['min_dice']:.4f}, {overall['max_dice']:.4f}]")

        # Organ group metrics
        if summary['organ_groups']:
            print(f"\nORGAN GROUP METRICS:")
            for group_name, group_metrics in summary['organ_groups'].items():
                print(f"  {group_name.upper()} organs: {group_metrics['dice_mean']:.4f} ± {group_metrics['dice_std']:.4f}")
                print(f"    Organs: {', '.join(group_metrics['organs'])}")

        # Per-organ metrics
        print(f"\nPER-ORGAN METRICS:")
        print(f"  {'Organ':<20} {'Dice':<15} {'HD95 (mm)':<15} {'NSD':<15}")
        print(f"  {'-'*20} {'-'*15} {'-'*15} {'-'*15}")

        # Sort organs by dice score
        sorted_organs = sorted(
            summary['per_organ_metrics'].items(),
            key=lambda x: x[1].get('dice_mean', 0),
            reverse=True
        )

        for organ, metrics in sorted_organs:
            dice_str = f"{metrics['dice_mean']:.4f}±{metrics['dice_std']:.4f}"
            hd95_str = f"{metrics['hd95_mean']:.2f}±{metrics['hd95_std']:.2f}" if 'hd95_mean' in metrics else "N/A"
            nsd_str = f"{metrics['nsd_mean']:.4f}±{metrics['nsd_std']:.4f}" if 'nsd_mean' in metrics else "N/A"

            print(f"  {organ:<20} {dice_str:<15} {hd95_str:<15} {nsd_str:<15}")

        print("="*80 + "\n")

    def evaluate_single_case(self, image_path, label_path=None, save_prediction=True):
        """Evaluate a single case (useful for debugging or inference)"""
        print(f"\nEvaluating single case: {image_path}")

        # TODO: Implement single case loading and evaluation
        # This would load a single NIfTI file, run inference, and optionally compare to ground truth
        pass


def main():
    parser = argparse.ArgumentParser(description='Evaluate LHA-Net')
    parser.add_argument('--config', type=str, default='configs/lha_net_config.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results (default: from config)')
    parser.add_argument('--save-predictions', action='store_true',
                        help='Save prediction masks')
    parser.add_argument('--single-case', type=str, default=None,
                        help='Evaluate a single case (path to image file)')

    args = parser.parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Create evaluator
    evaluator = ModelEvaluator(args.config, args.checkpoint, args.output_dir)

    # Evaluate
    if args.single_case:
        evaluator.evaluate_single_case(args.single_case, save_prediction=args.save_predictions)
    else:
        dataloader = evaluator.build_dataloader(split=args.split)
        evaluator.evaluate(dataloader, save_predictions=args.save_predictions)


if __name__ == '__main__':
    main()
