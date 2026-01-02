#!/usr/bin/env python3
"""
Test script to verify training pipeline with dummy data
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.lha_net_adaptive import LHANetAdaptive
from src.training.optimizer import create_optimizer
from src.training.scheduler import create_scheduler
from src.training.mixed_precision import MixedPrecisionTraining
from src.losses.combo_loss import ComboLoss


def test_training_pipeline():
    """Test training pipeline with dummy data"""

    print("=" * 60)
    print("Testing LHA-Net Training Pipeline with Dummy Data")
    print("=" * 60)

    # Load config
    config_path = "configs/lha_net_adaptive_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n1. Device: {device}")

    # Build model
    print("\n2. Building model...")
    model_config = config['model']

    model = LHANetAdaptive(
        in_channels=model_config['in_channels'],
        num_classes=model_config['num_classes'],
        backbone_type=model_config['backbone_type'],
        use_lightweight=model_config['use_lightweight'],
        scale_bank=model_config['scale_bank'],
        num_active_scales=model_config['num_active_scales'],
        share_scale_selection=model_config['share_scale_selection'],
        use_deep_supervision=model_config['use_deep_supervision']
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    # Build optimizer
    print("\n3. Building optimizer...")
    opt_config = config['training']['optimizer']

    optimizer = create_optimizer(
        model,
        optimizer_type=opt_config['type'],
        learning_rate=float(config['training']['learning_rate']),
        weight_decay=float(opt_config.get('weight_decay', 1e-4)),
        betas=tuple(opt_config.get('betas', [0.9, 0.999])),
        eps=float(opt_config.get('eps', 1e-8)),
        differential_lr=opt_config.get('differential_lr', True),
        backbone_lr_factor=float(opt_config.get('backbone_lr_factor', 0.1))
    )
    print(f"   Optimizer: {opt_config['type']}")
    print(f"   Learning rate: {config['training']['learning_rate']}")
    print(f"   Parameter groups: {len(optimizer.param_groups)}")

    # Build scheduler
    print("\n4. Building scheduler...")
    sched_config = config['training']['scheduler']

    scheduler = create_scheduler(
        optimizer,
        scheduler_type=sched_config['type'],
        num_epochs=config['training']['num_epochs'],
        warmup_epochs=sched_config.get('warmup_epochs', 5),
        min_lr=float(sched_config.get('min_lr', 1e-6))
    )
    print(f"   Scheduler: {sched_config['type']}")
    print(f"   Warmup epochs: {sched_config.get('warmup_epochs', 5)}")

    # Build loss function
    print("\n5. Building loss function...")
    loss_config = config['loss']

    criterion = ComboLoss(
        num_classes=model_config['num_classes'],
        focal_weight=loss_config['focal_weight'],
        dice_weight=loss_config['dice_weight'],
        size_weight=loss_config['size_weight']
    )
    print(f"   Loss: ComboLoss")
    print(f"   Focal weight: {loss_config['focal_weight']}")
    print(f"   Dice weight: {loss_config['dice_weight']}")

    # Initialize mixed precision
    print("\n6. Initializing mixed precision...")
    mp_config = config['training'].get('mixed_precision', {})
    mp_trainer = MixedPrecisionTraining(
        enabled=mp_config.get('enabled', False),
        init_scale=mp_config.get('init_scale', 2**16)
    )
    print(f"   Enabled: {mp_config.get('enabled', False)}")

    # Create dummy data
    print("\n7. Creating dummy data...")
    batch_size = config['training']['batch_size']
    patch_size = config['data']['patch_size']
    num_classes = model_config['num_classes']

    # Dummy input: [B, C, D, H, W]
    dummy_input = torch.randn(
        batch_size,
        model_config['in_channels'],
        patch_size[0],
        patch_size[1],
        patch_size[2]
    ).to(device)

    # Dummy target: [B, D, H, W] with class indices
    dummy_target = torch.randint(
        0, num_classes,
        (batch_size, patch_size[0], patch_size[1], patch_size[2])
    ).to(device)

    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Target shape: {dummy_target.shape}")

    # Test forward pass
    print("\n8. Testing forward pass...")
    model.train()

    try:
        with torch.cuda.amp.autocast(enabled=mp_trainer.enabled):
            outputs = model(dummy_input)

        if isinstance(outputs, dict):
            main_output = outputs.get('final_prediction', outputs.get('output'))
            print(f"   Output shape: {main_output.shape}")
            deep_outputs = outputs.get('deep_supervision_outputs', outputs.get('deep_outputs', []))
            print(f"   Deep supervision outputs: {len(deep_outputs)}")
        else:
            main_output = outputs
            print(f"   Output shape: {main_output.shape}")

        print("   ✓ Forward pass successful")
    except Exception as e:
        print(f"   ✗ Forward pass failed: {e}")
        raise

    # Test loss computation
    print("\n9. Testing loss computation...")

    try:
        # Extract prediction from model output
        if isinstance(outputs, dict):
            prediction = outputs['final_prediction']
        else:
            prediction = outputs

        loss_result = criterion(prediction, dummy_target)

        if isinstance(loss_result, dict):
            total_loss = loss_result.get('total_loss', loss_result.get('loss'))
            print(f"   Total loss: {total_loss.item():.4f}")
            for k, v in loss_result.items():
                if k not in ['total_loss', 'loss'] and isinstance(v, torch.Tensor):
                    print(f"   {k}: {v.item():.4f}")
        else:
            total_loss = loss_result
            print(f"   Total loss: {total_loss.item():.4f}")

        print("   ✓ Loss computation successful")
    except Exception as e:
        print(f"   ✗ Loss computation failed: {e}")
        raise

    # Test backward pass
    print("\n10. Testing backward pass...")

    try:
        optimizer.zero_grad()

        if mp_trainer.enabled:
            mp_trainer.scaler.scale(total_loss).backward()
            mp_trainer.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            mp_trainer.scaler.step(optimizer)
            mp_trainer.scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

        print("   ✓ Backward pass successful")
    except Exception as e:
        print(f"   ✗ Backward pass failed: {e}")
        raise

    # Test scheduler step
    print("\n11. Testing scheduler step...")

    try:
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"   Current LR: {current_lr:.6f}")
        print("   ✓ Scheduler step successful")
    except Exception as e:
        print(f"   ✗ Scheduler step failed: {e}")
        raise

    # Run a few training iterations
    print("\n12. Running 3 training iterations...")

    try:
        for iteration in range(3):
            model.train()
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=mp_trainer.enabled):
                outputs = model(dummy_input)

                # Extract prediction from model output
                if isinstance(outputs, dict):
                    prediction = outputs['final_prediction']
                else:
                    prediction = outputs

                loss_result = criterion(prediction, dummy_target)

                if isinstance(loss_result, dict):
                    loss = loss_result.get('total_loss', loss_result.get('loss'))
                else:
                    loss = loss_result

            if mp_trainer.enabled:
                mp_trainer.scaler.scale(loss).backward()
                mp_trainer.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                mp_trainer.scaler.step(optimizer)
                mp_trainer.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

            print(f"   Iteration {iteration + 1}/3: Loss = {loss.item():.4f}")

        print("   ✓ Training iterations successful")
    except Exception as e:
        print(f"   ✗ Training iterations failed: {e}")
        raise

    # Memory info
    if torch.cuda.is_available():
        print("\n13. GPU Memory Usage:")
        print(f"   Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"   Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    print("\n" + "=" * 60)
    print("✓ All tests passed! Training pipeline is working correctly.")
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_training_pipeline()
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)
