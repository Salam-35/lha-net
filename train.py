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
from src.evaluation.metrics import compute_dice_score
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

        # Metrics will be computed inline using compute_dice_score

        # Initialize memory monitor
        self.memory_monitor = MemoryMonitor(self.device)

        # Training state
        self.current_epoch = 0
        self.best_dice = 0.0
        self.global_step = 0

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
                    loss = loss_dict['total_loss'] if isinstance(loss_dict, dict) else loss_dict
                else:
                    loss_dict = self.criterion(outputs, labels)
                    loss = loss_dict['total_loss'] if isinstance(loss_dict, dict) else loss_dict

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

        avg_loss = epoch_loss / num_batches
        return avg_loss

    def validate(self):
        """Validate the model"""
        self.model.eval()
        val_loss = 0.0
        all_dice_scores = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")

            for batch in pbar:
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
                loss = self.criterion(predictions, labels)
                val_loss += loss.item()

                # Calculate metrics
                pred_masks = torch.argmax(predictions, dim=1)

                # Compute Dice score per sample in batch
                for b in range(pred_masks.shape[0]):
                    dice = compute_dice_score(pred_masks[b], labels[b], ignore_background=True)
                    if isinstance(dice, np.ndarray):
                        all_dice_scores.append(dice.mean())
                    else:
                        all_dice_scores.append(dice)

                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = val_loss / len(self.val_loader)
        avg_dice = np.mean(all_dice_scores)

        return avg_loss, avg_dice

    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_dice': self.best_dice,
            'config': self.config
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

    def train(self):
        """Main training loop"""
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
            train_loss = self.train_epoch()

            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
            else:
                current_lr = self.config['training']['learning_rate']

            print(f"\nEpoch {epoch + 1}/{self.config['training']['num_epochs']}")
            print(f"Train Loss: {train_loss:.4f}, LR: {current_lr:.6f}")

            # Validation
            if (epoch + 1) % self.config['training']['validation_freq'] == 0:
                val_loss, val_dice = self.validate()
                print(f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")

                # Save best model
                is_best = val_dice > self.best_dice
                if is_best:
                    self.best_dice = val_dice
                    self.save_checkpoint(is_best=True)

            # Save checkpoint periodically
            if (epoch + 1) % self.config['training']['save_freq'] == 0:
                self.save_checkpoint()

            # Clear GPU cache periodically
            if torch.cuda.is_available() and (epoch + 1) % self.config['training']['memory_optimization']['empty_cache_freq'] == 0:
                torch.cuda.empty_cache()

        print("\n" + "="*50)
        print("Training Completed!")
        print(f"Best Dice Score: {self.best_dice:.4f}")
        print("="*50)


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
