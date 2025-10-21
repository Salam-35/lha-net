#!/usr/bin/env python3
"""
LHA-Net Training Script
Trains the Lightweight Hierarchical Attention Network for medical image segmentation
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
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.lha_net import LHANet
from src.data.preprocessing import AMOSDataset
from src.losses.combo_loss import ComboLoss
from src.training.optimizer import create_optimizer
from src.training.scheduler import create_scheduler
from src.training.mixed_precision import MixedPrecisionTraining
from src.evaluation.metrics import compute_dice_score, SegmentationMetrics
from src.utils.memory_utils import MemoryMonitor


class Trainer:
    """Training manager for LHA-Net"""

    def __init__(self, config_path):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Setup device
        self.device = torch.device(self.config['system']['device'] if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Create output directories
        self.setup_directories()

        # Initialize model
        self.model = self.build_model()

        # Initialize data loaders
        self.train_loader, self.val_loader = self.build_dataloaders()

        # Initialize loss function
        self.criterion = self.build_loss()

        # Initialize optimizer and scheduler
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()

        # Initialize mixed precision trainer
        self.mp_trainer = MixedPrecisionTraining(
            enabled=self.config['training']['mixed_precision']['enabled'],
            init_scale=self.config['training']['mixed_precision']['init_scale']
        )

        # Initialize comprehensive metrics calculator
        # Default names (used if config['labels'] not provided)
        default_organ_names = [
            'background', 'liver', 'right_kidney', 'spleen', 'pancreas',
            'aorta', 'ivc', 'right_adrenal', 'left_adrenal', 'gallbladder',
            'esophagus', 'stomach', 'duodenum', 'left_kidney', 'class_14', 'class_15'
        ]
        num_classes = self.config['model']['num_classes']
        labels_map = self.config.get('labels', None)
        if labels_map:
            # Build ordered list by class index 0..num_classes-1
            ordered = []
            for i in range(num_classes):
                key = str(i)
                if key in labels_map:
                    ordered.append(labels_map[key])
                else:
                    # Fallback to default or generic
                    if i < len(default_organ_names):
                        ordered.append(default_organ_names[i])
                    else:
                        ordered.append(f'class_{i}')
            self.organ_names = ordered
        else:
            # Fallback to defaults, pad/truncate to match num_classes
            if len(default_organ_names) < num_classes:
                default_organ_names = list(default_organ_names) + [f'class_{i}' for i in range(len(default_organ_names), num_classes)]
            elif len(default_organ_names) > num_classes:
                default_organ_names = default_organ_names[:num_classes]
            self.organ_names = default_organ_names
        self.metrics_calculator = SegmentationMetrics(
            num_classes=self.config['model']['num_classes'],
            spacing=tuple(self.config['data']['preprocessing']['target_spacing']),
            organ_names=self.organ_names
        )

        # Initialize memory monitor
        self.memory_monitor = MemoryMonitor(self.device)

        # Training state
        self.current_epoch = 0
        self.best_dice = 0.0
        self.global_step = 0

        # Metrics history storage
        self.metrics_history = {
            'train': {'epochs': [], 'losses': [], 'loss_components': []},
            'val': {'epochs': [], 'losses': [], 'dice_scores': [], 'per_organ_metrics': []}
        }

    def setup_directories(self):
        """Create necessary directories"""
        self.output_dir = Path(self.config['paths']['output_dir'])
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.log_dir = self.output_dir / 'logs'

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        print(f"Output directory: {self.output_dir}")
        print(f"Checkpoint directory: {self.checkpoint_dir}")

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
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model: LHA-Net ({model_config['type']})")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size: {total_params * 4 / 1024**2:.2f} MB")

        return model

    def build_dataloaders(self):
        """Build training and validation dataloaders"""
        data_root = Path(self.config['paths']['data_root'])

        # Training dataset
        train_dataset = AMOSDataset(
            data_root=data_root,
            split='train',
            patch_size=self.config['data']['patch_size'],
            num_classes=self.config['model']['num_classes'],
            augmentation=self.config['data']['augmentation']['enabled']
        )

        # Validation dataset
        val_dataset = AMOSDataset(
            data_root=data_root,
            split='val',
            patch_size=self.config['data']['patch_size'],
            num_classes=self.config['model']['num_classes'],
            augmentation=False
        )

        # Training loader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['system']['num_workers'],
            pin_memory=self.config['system']['pin_memory'],
            drop_last=True
        )

        # Validation loader
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['system']['num_workers'],
            pin_memory=self.config['system']['pin_memory']
        )

        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")

        return train_loader, val_loader

    def build_loss(self):
        """Build loss function"""
        loss_config = self.config['loss']

        criterion = ComboLoss(
            num_classes=self.config['model']['num_classes'],
            focal_weight=loss_config['focal_weight'],
            dice_weight=loss_config['dice_weight'],
            size_weight=loss_config['size_weight'],
            deep_supervision_weight=loss_config['deep_supervision_weight']
        )

        return criterion

    def build_optimizer(self):
        """Build optimizer"""
        opt_config = self.config['training']['optimizer']

        optimizer = create_optimizer(
            self.model,
            optimizer_type=opt_config['type'],
            learning_rate=self.config['training']['learning_rate'],
            weight_decay=opt_config.get('weight_decay', 1e-4),
            betas=tuple(opt_config.get('betas', [0.9, 0.999])),
            eps=opt_config.get('eps', 1e-8),
            differential_lr=opt_config.get('differential_lr', True),
            backbone_lr_factor=opt_config.get('backbone_lr_factor', 0.1)
        )
        return optimizer

    def build_scheduler(self):
        """Build learning rate scheduler"""
        sched_config = self.config['training']['scheduler']

        scheduler = create_scheduler(
            self.optimizer,
            scheduler_type=sched_config['type'],
            num_epochs=self.config['training']['num_epochs'],
            warmup_epochs=sched_config.get('warmup_epochs', 5),
            min_lr=sched_config.get('min_lr', 1e-6)
        )
        return scheduler

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        num_batches = len(self.train_loader)

        # Track loss components
        loss_components = {
            'total_loss': 0.0,
            'focal_loss': 0.0,
            'dice_loss': 0.0,
            'size_loss': 0.0,
            'routing_loss': 0.0
        }

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")

        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)

            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=self.mp_trainer.enabled):
                outputs = self.model(images)

                # Calculate loss
                if isinstance(outputs, dict):
                    loss_dict = self.criterion(outputs['final_prediction'], labels)
                else:
                    loss_dict = self.criterion(outputs, labels)

                # Extract loss components
                if isinstance(loss_dict, dict):
                    loss = loss_dict['total_loss']
                    # Accumulate loss components
                    for key in loss_components.keys():
                        if key in loss_dict:
                            loss_components[key] += loss_dict[key].item() if torch.is_tensor(loss_dict[key]) else loss_dict[key]
                else:
                    loss = loss_dict
                    loss_components['total_loss'] += loss.item()

            # Backward pass with gradient scaling
            self.optimizer.zero_grad()
            self.mp_trainer.backward(loss)

            # Gradient clipping
            if self.config['training']['max_grad_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['max_grad_norm']
                )

            # Optimizer step
            self.mp_trainer.step_optimizer(self.optimizer)

            # Update stats
            epoch_loss += loss.item()
            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{epoch_loss / (batch_idx + 1):.4f}"
            })

            # Log GPU memory periodically
            if batch_idx % self.config['system']['memory_log_interval'] == 0:
                if torch.cuda.is_available():
                    memory_stats = self.memory_monitor.get_memory_stats()
                    pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'gpu_mb': f"{memory_stats.get('gpu_allocated_mb', 0):.0f}"
                    })

        # Average losses
        avg_loss = epoch_loss / num_batches
        avg_loss_components = {k: v / num_batches for k, v in loss_components.items()}

        return avg_loss, avg_loss_components

    def validate(self, compute_detailed_metrics=True):
        """Validate the model with comprehensive metrics

        Args:
            compute_detailed_metrics: If True, compute HD95 and NSD (slower).
                                     If False, only compute Dice scores (faster).
        """
        self.model.eval()
        val_loss = 0.0
        all_dice_scores = []

        # Comprehensive metrics storage
        per_organ_dice = {organ: [] for organ in self.organ_names}
        per_organ_hd95 = {organ: [] for organ in self.organ_names[1:]} if compute_detailed_metrics else {}
        per_organ_nsd = {organ: [] for organ in self.organ_names[1:]} if compute_detailed_metrics else {}

        # Loss components
        loss_components = {
            'total_loss': 0.0,
            'focal_loss': 0.0,
            'dice_loss': 0.0,
            'size_loss': 0.0,
            'routing_loss': 0.0
        }

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")

            for batch_idx, batch in enumerate(pbar):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)

                # Forward pass
                outputs = self.model(images)

                # Get predictions
                if isinstance(outputs, dict):
                    predictions = outputs['final_prediction']
                else:
                    predictions = outputs

                # Calculate loss
                loss_dict = self.criterion(predictions, labels)
                if isinstance(loss_dict, dict):
                    loss = loss_dict['total_loss']
                    # Accumulate loss components
                    for key in loss_components.keys():
                        if key in loss_dict:
                            loss_components[key] += loss_dict[key].item() if torch.is_tensor(loss_dict[key]) else loss_dict[key]
                else:
                    loss = loss_dict
                    loss_components['total_loss'] += loss.item()

                val_loss += loss.item()

                # Calculate comprehensive metrics per sample in batch
                pred_masks = torch.argmax(predictions, dim=1)

                for b in range(pred_masks.shape[0]):
                    # Compute metrics for this sample
                    try:
                        if compute_detailed_metrics:
                            # Compute all metrics including HD95 and NSD (slower)
                            metrics = self.metrics_calculator.compute_all_metrics(
                                pred_masks[b],
                                labels[b],
                                case_id=f"val_batch{batch_idx}_sample{b}"
                            )

                            # Store per-organ dice scores
                            for organ, dice_val in metrics['dice_scores'].items():
                                if organ in per_organ_dice:
                                    per_organ_dice[organ].append(dice_val)

                            # Store HD95 scores
                            for organ, hd95_val in metrics['hd95_scores'].items():
                                if organ in per_organ_hd95 and not np.isinf(hd95_val):
                                    per_organ_hd95[organ].append(hd95_val)

                            # Store NSD scores
                            for organ, nsd_val in metrics['nsd_scores'].items():
                                if organ in per_organ_nsd:
                                    per_organ_nsd[organ].append(nsd_val)

                            # Overall dice (excluding background)
                            all_dice_scores.append(metrics['mean_dice'])
                        else:
                            # Fast mode: Only compute Dice scores (much faster)
                            # Convert to one-hot for per-class dice computation
                            pred_one_hot = np.zeros((self.config['model']['num_classes'],) + pred_masks[b].shape, dtype=np.float32)
                            labels_one_hot = np.zeros((self.config['model']['num_classes'],) + labels[b].shape, dtype=np.float32)

                            for c in range(self.config['model']['num_classes']):
                                pred_one_hot[c] = (pred_masks[b].cpu().numpy() == c).astype(np.float32)
                                labels_one_hot[c] = (labels[b].cpu().numpy() == c).astype(np.float32)

                            # Compute per-organ dice
                            for c, organ in enumerate(self.organ_names):
                                dice = compute_dice_score(pred_one_hot[c], labels_one_hot[c], ignore_background=False)
                                per_organ_dice[organ].append(dice)

                            # Overall dice (excluding background)
                            dice = compute_dice_score(pred_masks[b], labels[b], ignore_background=True)
                            if isinstance(dice, np.ndarray):
                                all_dice_scores.append(dice.mean())
                            else:
                                all_dice_scores.append(dice)

                    except Exception as e:
                        # Fallback to simple dice if metrics calculation fails
                        print(f"Warning: Metrics calculation failed for batch {batch_idx}, sample {b}: {e}")
                        dice = compute_dice_score(pred_masks[b], labels[b], ignore_background=True)
                        if isinstance(dice, np.ndarray):
                            all_dice_scores.append(dice.mean())
                        else:
                            all_dice_scores.append(dice)

                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # Calculate averages
        avg_loss = val_loss / len(self.val_loader)
        avg_dice = np.mean(all_dice_scores) if all_dice_scores else 0.0
        avg_loss_components = {k: v / len(self.val_loader) for k, v in loss_components.items()}

        # Aggregate per-organ metrics
        organ_metrics = {}
        for organ in self.organ_names:
            if organ in per_organ_dice and per_organ_dice[organ]:
                organ_metrics[organ] = {
                    'dice_mean': np.mean(per_organ_dice[organ]),
                    'dice_std': np.std(per_organ_dice[organ])
                }

                # Add HD95 if available (not for background)
                if organ in per_organ_hd95 and per_organ_hd95[organ]:
                    # Filter out nan values
                    valid_hd95 = [v for v in per_organ_hd95[organ] if not np.isnan(v)]
                    if valid_hd95:
                        organ_metrics[organ]['hd95_mean'] = np.mean(valid_hd95)
                        organ_metrics[organ]['hd95_std'] = np.std(valid_hd95)

                # Add NSD if available (not for background)
                if organ in per_organ_nsd and per_organ_nsd[organ]:
                    # Filter out nan values
                    valid_nsd = [v for v in per_organ_nsd[organ] if not np.isnan(v)]
                    if valid_nsd:
                        organ_metrics[organ]['nsd_mean'] = np.mean(valid_nsd)
                        organ_metrics[organ]['nsd_std'] = np.std(valid_nsd)

        return avg_loss, avg_dice, avg_loss_components, organ_metrics

    def save_checkpoint(self, is_best=False, epoch_metrics=None):
        """Save model checkpoint with comprehensive metrics history"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_dice': self.best_dice,
            'config': self.config,
            'metrics_history': self.metrics_history,
            'current_epoch_metrics': epoch_metrics
        }

        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_checkpoint.pth'
            torch.save(checkpoint, best_path)
            print(f"Saved best checkpoint: {best_path} (Dice: {self.best_dice:.4f})")

    def print_metrics_summary(self, epoch, train_loss, train_loss_components, val_loss=None,
                             val_dice=None, val_loss_components=None, organ_metrics=None):
        """Print comprehensive metrics summary"""
        print("\n" + "="*80)
        print(f"EPOCH {epoch + 1}/{self.config['training']['num_epochs']} SUMMARY")
        print("="*80)

        # Training metrics
        print(f"\nTRAINING METRICS:")
        print(f"  Total Loss: {train_loss:.4f}")
        if train_loss_components:
            print(f"  Loss Components:")
            for key, value in train_loss_components.items():
                if value > 0:
                    print(f"    {key}: {value:.4f}")

        # Validation metrics
        if val_loss is not None:
            print(f"\nVALIDATION METRICS:")
            print(f"  Total Loss: {val_loss:.4f}")
            print(f"  Mean Dice Score: {val_dice:.4f}")

            if val_loss_components:
                print(f"  Loss Components:")
                for key, value in val_loss_components.items():
                    if value > 0:
                        print(f"    {key}: {value:.4f}")

            # Per-organ metrics
            if organ_metrics:
                print(f"\nPER-ORGAN METRICS:")
                print(f"  {'Organ':<20} {'Dice':<12} {'HD95 (mm)':<15} {'NSD':<12}")
                print(f"  {'-'*20} {'-'*12} {'-'*15} {'-'*12}")

                for organ, metrics in organ_metrics.items():
                    if organ == 'background':
                        continue

                    dice_str = f"{metrics['dice_mean']:.4f}±{metrics['dice_std']:.4f}"

                    # Handle HD95 with nan values
                    if 'hd95_mean' in metrics and not np.isnan(metrics['hd95_mean']):
                        hd95_str = f"{metrics['hd95_mean']:.2f}±{metrics['hd95_std']:.2f}"
                    else:
                        hd95_str = "N/A"

                    # Handle NSD with nan values
                    if 'nsd_mean' in metrics and not np.isnan(metrics['nsd_mean']):
                        nsd_str = f"{metrics['nsd_mean']:.4f}±{metrics['nsd_std']:.4f}"
                    else:
                        nsd_str = "N/A"

                    print(f"  {organ:<20} {dice_str:<12} {hd95_str:<15} {nsd_str:<12}")

        # Learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f"\nLearning Rate: {current_lr:.6f}")
        print("="*80 + "\n")

    def print_training_summary(self):
        """Print final training summary with metrics history"""
        print("\n" + "="*80)
        print("TRAINING SUMMARY")
        print("="*80)

        # Training progression
        if self.metrics_history['train']['epochs']:
            print("\nTraining Loss Progression:")
            for i, epoch in enumerate(self.metrics_history['train']['epochs']):
                loss = self.metrics_history['train']['losses'][i]
                print(f"  Epoch {epoch}: {loss:.4f}")

        # Validation progression
        if self.metrics_history['val']['epochs']:
            print("\nValidation Metrics Progression:")
            for i, epoch in enumerate(self.metrics_history['val']['epochs']):
                val_loss = self.metrics_history['val']['losses'][i]
                val_dice = self.metrics_history['val']['dice_scores'][i]
                print(f"  Epoch {epoch}: Loss={val_loss:.4f}, Dice={val_dice:.4f}")

        # Best performance
        if self.metrics_history['val']['dice_scores']:
            best_epoch_idx = np.argmax(self.metrics_history['val']['dice_scores'])
            best_epoch = self.metrics_history['val']['epochs'][best_epoch_idx]
            best_dice = self.metrics_history['val']['dice_scores'][best_epoch_idx]
            best_organ_metrics = self.metrics_history['val']['per_organ_metrics'][best_epoch_idx]

            print(f"\nBest Performance (Epoch {best_epoch}):")
            print(f"  Mean Dice Score: {best_dice:.4f}")

            if best_organ_metrics:
                print(f"\n  Best Per-Organ Dice Scores:")
                sorted_organs = sorted(best_organ_metrics.items(),
                                      key=lambda x: x[1].get('dice_mean', 0),
                                      reverse=True)
                for organ, metrics in sorted_organs:
                    if organ != 'background':
                        dice_mean = metrics.get('dice_mean', 0)
                        dice_std = metrics.get('dice_std', 0)
                        print(f"    {organ:<20}: {dice_mean:.4f}±{dice_std:.4f}")

        print("="*80)

    def train(self):
        """Main training loop with comprehensive metrics tracking"""
        print("\n" + "="*50)
        print("Starting Training")
        print("="*50)
        print(f"Total epochs: {self.config['training']['num_epochs']}")
        print(f"Batch size: {self.config['training']['batch_size']}")
        print(f"Learning rate: {self.config['training']['learning_rate']}")
        print("="*50 + "\n")

        for epoch in range(self.config['training']['num_epochs']):
            self.current_epoch = epoch

            # Train epoch
            train_loss, train_loss_components = self.train_epoch()

            # Store training metrics
            self.metrics_history['train']['epochs'].append(epoch + 1)
            self.metrics_history['train']['losses'].append(train_loss)
            self.metrics_history['train']['loss_components'].append(train_loss_components)

            # Update learning rate
            if self.scheduler:
                self.scheduler.step()

            # Validation
            val_loss = None
            val_dice = None
            val_loss_components = None
            organ_metrics = None

            if (epoch + 1) % self.config['training']['validation_freq'] == 0:
                # Check if we should compute detailed metrics (HD95, NSD) this epoch
                detailed_freq = self.config['training'].get('detailed_metrics_freq', 5)
                compute_detailed = (epoch + 1) % detailed_freq == 0

                if not compute_detailed:
                    print("Running fast validation (Dice only)...")
                else:
                    print("Running detailed validation (Dice + HD95 + NSD)...")

                val_loss, val_dice, val_loss_components, organ_metrics = self.validate(
                    compute_detailed_metrics=compute_detailed
                )

                # Store validation metrics
                self.metrics_history['val']['epochs'].append(epoch + 1)
                self.metrics_history['val']['losses'].append(val_loss)
                self.metrics_history['val']['dice_scores'].append(val_dice)
                self.metrics_history['val']['per_organ_metrics'].append(organ_metrics)

                # Save best model
                is_best = val_dice > self.best_dice
                if is_best:
                    self.best_dice = val_dice

                # Prepare epoch metrics for checkpoint
                epoch_metrics = {
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'train_loss_components': train_loss_components,
                    'val_loss': val_loss,
                    'val_dice': val_dice,
                    'val_loss_components': val_loss_components,
                    'organ_metrics': organ_metrics
                }

                # Save checkpoint with metrics
                if is_best:
                    self.save_checkpoint(is_best=True, epoch_metrics=epoch_metrics)

            # Print comprehensive metrics summary
            self.print_metrics_summary(
                epoch, train_loss, train_loss_components,
                val_loss, val_dice, val_loss_components, organ_metrics
            )

            # Save checkpoint periodically
            if (epoch + 1) % self.config['training']['save_freq'] == 0:
                epoch_metrics = {
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'train_loss_components': train_loss_components,
                }
                self.save_checkpoint(epoch_metrics=epoch_metrics)

            # Clear GPU cache periodically
            if torch.cuda.is_available() and (epoch + 1) % self.config['training']['memory_optimization']['empty_cache_freq'] == 0:
                torch.cuda.empty_cache()

        print("\n" + "="*50)
        print("Training Completed!")
        print(f"Best Dice Score: {self.best_dice:.4f}")
        print("="*50)

        # Print final summary
        self.print_training_summary()


def main():
    parser = argparse.ArgumentParser(description='Train LHA-Net')
    parser.add_argument('--config', type=str, default='configs/lha_net_config.yaml',
                        help='Path to config file')
    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Create trainer and start training
    trainer = Trainer(args.config)
    trainer.train()


if __name__ == '__main__':
    main()
