#!/usr/bin/env python3
"""
Training script for LHA-Net with Adaptive PMSA

Usage:
    python train_adaptive.py --config configs/lha_net_adaptive_config.yaml
"""

import os
import sys
import argparse
import yaml
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.models.lha_net_adaptive import create_lha_net_adaptive
from src.losses.combo_loss import LHANetLoss
from src.training.optimizer import create_optimizer
from src.training.scheduler import create_scheduler
from src.training.adaptive_trainer import AdaptivePMSATrainer
from src.data.preprocessed_dataset import AMOS22Dataset
from torch.utils.data import DataLoader


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dataloaders(config: dict):
    """Create training and validation dataloaders"""

    # Training dataset
    train_dataset = AMOS22Dataset(
        data_dir=config['paths']['data_root'],
        split='train',
        patch_size=config['data']['patch_size'],
        num_patches_per_volume=config['data']['patches_per_volume'],
        augmentation=config['data']['augmentation']['enabled']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['system']['num_workers'],
        pin_memory=config['system']['pin_memory']
    )

    # Validation dataset
    val_dataset = AMOS22Dataset(
        data_dir=config['paths']['data_root'],
        split='val',
        patch_size=config['data']['patch_size'],
        num_patches_per_volume=config['data']['patches_per_volume'],
        augmentation=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['system']['num_workers'],
        pin_memory=config['system']['pin_memory']
    )

    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description='Train LHA-Net with Adaptive PMSA')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    print("\nCreating model...")
    model = create_lha_net_adaptive(
        config_type=config['model']['type'],
        num_classes=config['model']['num_classes'],
        scale_bank=config['model']['scale_bank'],
        num_active_scales=config['model']['num_active_scales'],
        base_channels=config['model']['base_channels'],
        use_adaptive_pmsa=config['model']['use_adaptive_pmsa'],
        share_scale_selection=config['model'].get('share_scale_selection', False)
    )
    model = model.to(device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Print scale configuration
    if hasattr(model, 'hierarchical_pmsa'):
        scale_bank = model.hierarchical_pmsa.pmsa_modules['level_0'].scale_bank
        print(f"\nScale bank: {scale_bank}")
        print(f"Active scales: {config['model']['num_active_scales']}")

    # Create loss function
    print("\nCreating loss function...")
    loss_fn = LHANetLoss(
        primary_loss_weight=config['loss']['primary_loss_weight'],
        deep_supervision_weight=config['loss']['deep_supervision_weight'],
        size_prediction_weight=config['loss']['size_prediction_weight'],
        focal_weight=config['loss']['focal_weight'],
        dice_weight=config['loss']['dice_weight']
    ).to(device)

    # Create optimizer
    print("Creating optimizer...")
    optimizer = create_optimizer(
        model=model,
        optimizer_type=config['training']['optimizer']['type'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['optimizer']['weight_decay']
    )

    # Create scheduler
    scheduler = create_scheduler(
        optimizer=optimizer,
        scheduler_type=config['training']['scheduler']['type'],
        num_epochs=config['training']['num_epochs'],
        warmup_epochs=config['training']['scheduler']['warmup_epochs']
    )

    # Create dataloaders
    print("Loading data...")
    train_loader, val_loader = create_dataloaders(config)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Create trainer
    trainer = AdaptivePMSATrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=config['paths']['checkpoint_dir'],
        log_scale_freq=config['training'].get('log_scale_freq', 5),
        visualize_scale_freq=config['training'].get('visualize_scale_freq', 10)
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Resumed from epoch {checkpoint['epoch']}")

    # Train
    trainer.train(
        num_epochs=config['training']['num_epochs'],
        scheduler=scheduler
    )


if __name__ == '__main__':
    main()
