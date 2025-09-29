#!/usr/bin/env python3
"""
Dummy Training Script for LHA-Net
Tests the complete training pipeline with synthetic data before using real AMOS22 data
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.lha_net import create_lha_net, LHANetWithAuxiliaryLoss
from losses.combo_loss import LHANetLoss
from training.optimizer import create_optimizer
from training.scheduler import create_scheduler
from training.mixed_precision import MixedPrecisionTraining, GradientAccumulation, MemoryOptimization
from evaluation.metrics import SegmentationMetrics
from utils.memory_utils import MemoryMonitor
from data.smart_sampling import SmartPatchSampler


class SyntheticDataGenerator:
    """Generate synthetic medical volumes for testing"""

    def __init__(
        self,
        volume_shape: Tuple[int, int, int] = (128, 256, 256),
        patch_size: Tuple[int, int, int] = (64, 128, 128),
        num_classes: int = 14
    ):
        self.volume_shape = volume_shape
        self.patch_size = patch_size
        self.num_classes = num_classes

    def create_synthetic_volume(self, case_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create a synthetic medical volume with realistic organ distribution"""
        d, h, w = self.volume_shape

        # Create image volume (simulating CT scan)
        image = np.random.normal(-200, 100, self.volume_shape).astype(np.float32)

        # Create label volume
        labels = np.zeros(self.volume_shape, dtype=np.uint8)

        # Add organs with AMOS22-like distribution
        np.random.seed(case_id)  # Reproducible per case

        # Large organs
        # Liver (class 1) - large central organ
        liver_center = (d//2, h//2, w//2 + 20)
        liver_size = (30, 60, 80)
        self._add_organ(labels, image, 1, liver_center, liver_size, intensity=50)

        # Kidneys (classes 2, 13) - medium size, bilateral
        kidney_r_center = (d//2, h//2 - 30, w//2 + 40)
        kidney_l_center = (d//2, h//2 + 30, w//2 + 40)
        kidney_size = (20, 25, 25)
        self._add_organ(labels, image, 2, kidney_r_center, kidney_size, intensity=30)
        self._add_organ(labels, image, 13, kidney_l_center, kidney_size, intensity=30)

        # Spleen (class 3) - medium size
        spleen_center = (d//2, h//2 + 50, w//2)
        spleen_size = (15, 30, 20)
        self._add_organ(labels, image, 3, spleen_center, spleen_size, intensity=40)

        # Medium organs
        # Adrenal glands (classes 7, 8) - small
        adrenal_r_center = (d//2 - 15, h//2 - 35, w//2 + 35)
        adrenal_l_center = (d//2 - 15, h//2 + 35, w//2 + 35)
        adrenal_size = (5, 8, 8)
        self._add_organ(labels, image, 7, adrenal_r_center, adrenal_size, intensity=35)
        self._add_organ(labels, image, 8, adrenal_l_center, adrenal_size, intensity=35)

        # Small organs (challenging for segmentation)
        # Gallbladder (class 9) - very small
        gb_center = (d//2, h//2 - 10, w//2 + 25)
        gb_size = (8, 6, 6)
        self._add_organ(labels, image, 9, gb_center, gb_size, intensity=0)  # Fluid-like

        # Duodenum (class 12) - small, elongated
        duodenum_center = (d//2 + 5, h//2, w//2 + 15)
        duodenum_size = (12, 8, 25)
        self._add_organ(labels, image, 12, duodenum_center, duodenum_size, intensity=20)

        # Add realistic noise and artifacts
        self._add_imaging_artifacts(image)

        return image, labels

    def _add_organ(
        self,
        labels: np.ndarray,
        image: np.ndarray,
        class_id: int,
        center: Tuple[int, int, int],
        size: Tuple[int, int, int],
        intensity: float
    ):
        """Add an organ with realistic shape to the volume"""
        d, h, w = self.volume_shape
        cd, ch, cw = center
        sd, sh, sw = size

        # Create ellipsoidal organ
        z, y, x = np.ogrid[:d, :h, :w]
        mask = (((z - cd) / (sd/2))**2 +
                ((y - ch) / (sh/2))**2 +
                ((x - cw) / (sw/2))**2) <= 1

        # Add some randomness to shape
        noise = np.random.normal(0, 0.1, mask.shape)
        mask = mask & (noise > -0.3)

        # Set label
        labels[mask] = class_id

        # Set image intensity
        image[mask] = np.random.normal(intensity, 10, mask.sum())

    def _add_imaging_artifacts(self, image: np.ndarray):
        """Add realistic CT imaging artifacts"""
        # Add Gaussian noise
        noise = np.random.normal(0, 5, image.shape)
        image += noise

        # Add some streak artifacts (simplified)
        for _ in range(3):
            streak_pos = np.random.randint(0, image.shape[1])
            image[:, streak_pos, :] += np.random.normal(0, 20, (image.shape[0], image.shape[2]))

        # Clip to realistic HU range
        image = np.clip(image, -1024, 3000)


class DummyDataset:
    """Dummy dataset that generates synthetic data on-the-fly"""

    def __init__(
        self,
        num_cases: int = 20,
        patch_size: Tuple[int, int, int] = (64, 128, 128),
        patches_per_volume: int = 8
    ):
        self.num_cases = num_cases
        self.patch_size = patch_size
        self.patches_per_volume = patches_per_volume
        self.generator = SyntheticDataGenerator(patch_size=patch_size)
        self.sampler = SmartPatchSampler(patch_size=patch_size)

    def __len__(self):
        return self.num_cases * self.patches_per_volume

    def __getitem__(self, idx):
        # Determine which case and which patch
        case_id = idx // self.patches_per_volume
        patch_id = idx % self.patches_per_volume

        # Generate volume for this case
        image, labels = self.generator.create_synthetic_volume(case_id)

        # Sample patch
        patches = self.sampler.sample_patches(image, labels, num_patches=1)
        patch = patches[0]

        # Convert to tensors
        image_tensor = torch.from_numpy(patch['image']).unsqueeze(0).float()  # Add channel dim
        label_tensor = torch.from_numpy(patch['label']).long()

        return image_tensor, label_tensor


def dummy_training_loop(
    model: torch.nn.Module,
    dataset: DummyDataset,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    num_epochs: int = 5,
    gradient_accumulation_steps: int = 4
) -> Dict:
    """Run dummy training loop"""

    # Initialize components
    memory_monitor = MemoryMonitor(device)
    metrics_calc = SegmentationMetrics()
    mixed_precision = MixedPrecisionTraining(enabled=(device.type == 'cuda'))
    grad_accumulation = GradientAccumulation(accumulation_steps=gradient_accumulation_steps)
    memory_opt = MemoryOptimization()

    # Apply memory optimizations
    model = memory_opt.optimize_model(model)

    training_history = {
        'losses': [],
        'dice_scores': [],
        'memory_usage': [],
        'learning_rates': []
    }

    print(f"Starting dummy training on {device}")
    print(f"Dataset size: {len(dataset)} patches")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    model.train()

    for epoch in range(num_epochs):
        epoch_losses = []
        epoch_dice_scores = []

        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 30)

        # Reset memory stats
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)

        for step in range(len(dataset)):
            # Get batch (single sample for simplicity)
            try:
                image, target = dataset[step]
                image = image.unsqueeze(0).to(device)  # Add batch dimension
                target = target.unsqueeze(0).to(device)

                # Forward pass with mixed precision
                with mixed_precision.get_context():
                    if isinstance(model, LHANetWithAuxiliaryLoss):
                        output = model(image)
                        loss_dict = loss_fn(output, target)
                        loss = loss_dict['total_loss']
                    else:
                        predictions = model(image)
                        if isinstance(predictions, dict):
                            predictions = predictions['final_prediction']
                        loss = F.cross_entropy(predictions, target)

                # Gradient accumulation
                should_step = grad_accumulation.accumulate_gradients(
                    loss, model, mixed_precision
                )

                if should_step:
                    # Optimizer step
                    mixed_precision.step_optimizer(optimizer)
                    optimizer.zero_grad()

                    # Scheduler step
                    scheduler.step()

                # Log metrics
                epoch_losses.append(loss.item())

                # Compute Dice score for monitoring
                if step % 10 == 0:
                    with torch.no_grad():
                        if isinstance(model, LHANetWithAuxiliaryLoss):
                            model.eval()
                            test_output = model(image)
                            if isinstance(test_output, dict):
                                test_predictions = test_output['final_prediction']
                            else:
                                test_predictions = test_output
                            model.train()
                        else:
                            test_predictions = predictions

                        pred_labels = torch.argmax(test_predictions, dim=1)
                        metrics = metrics_calc.compute_all_metrics(
                            pred_labels[0].cpu().numpy(),
                            target[0].cpu().numpy()
                        )
                        epoch_dice_scores.append(metrics['mean_dice'])

                        # Memory usage
                        memory_stats = memory_monitor.get_memory_stats()
                        training_history['memory_usage'].append(memory_stats)

                        # Progress logging
                        if step % 20 == 0:
                            lr = optimizer.param_groups[0]['lr']
                            print(f"  Step {step:3d}: Loss={loss.item():.4f}, "
                                  f"Dice={metrics['mean_dice']:.3f}, LR={lr:.2e}")

                            if device.type == 'cuda':
                                gpu_mem = memory_stats.get('gpu_allocated_gb', 0)
                                print(f"              GPU Memory: {gpu_mem:.2f}GB")

                # Memory management
                memory_opt.step_memory_management()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM at step {step}, clearing cache and continuing...")
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e

        # Epoch summary
        avg_loss = np.mean(epoch_losses) if epoch_losses else float('inf')
        avg_dice = np.mean(epoch_dice_scores) if epoch_dice_scores else 0.0
        current_lr = optimizer.param_groups[0]['lr']

        training_history['losses'].append(avg_loss)
        training_history['dice_scores'].append(avg_dice)
        training_history['learning_rates'].append(current_lr)

        print(f"Epoch {epoch + 1} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Average Dice: {avg_dice:.3f}")
        print(f"  Learning Rate: {current_lr:.2e}")

        if device.type == 'cuda':
            peak_memory = torch.cuda.max_memory_allocated(device) / 1e9
            print(f"  Peak GPU Memory: {peak_memory:.2f}GB")

    return training_history


def test_model_components():
    """Test individual model components before full training"""
    print("Testing Model Components...")
    print("=" * 40)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test model creation
    print("1. Testing model creation...")
    try:
        model = create_lha_net(config_type="lightweight", num_classes=14)
        model = model.to(device)
        print("   ✅ Model created successfully")

        # Get model info
        model_info = model.get_model_size()
        print(f"   Parameters: {model_info['total_parameters']:,}")
    except Exception as e:
        print(f"   ❌ Model creation failed: {e}")
        return False

    # Test forward pass
    print("2. Testing forward pass...")
    try:
        batch_size = 2
        test_input = torch.randn(batch_size, 1, 64, 128, 128, device=device)

        model.eval()
        with torch.no_grad():
            output = model(test_input)

        if isinstance(output, dict):
            output_shape = output['final_prediction'].shape
        else:
            output_shape = output.shape

        expected_shape = (batch_size, 14, 64, 128, 128)
        assert output_shape == expected_shape, f"Expected {expected_shape}, got {output_shape}"
        print("   ✅ Forward pass successful")
    except Exception as e:
        print(f"   ❌ Forward pass failed: {e}")
        return False

    # Test loss function
    print("3. Testing loss function...")
    try:
        loss_fn = LHANetLoss()
        target = torch.randint(0, 14, (batch_size, 64, 128, 128), device=device)

        model.train()
        output = model(test_input, return_features=True)
        loss_dict = loss_fn(output, target)

        assert 'total_loss' in loss_dict
        assert torch.isfinite(loss_dict['total_loss'])
        print(f"   ✅ Loss computation successful: {loss_dict['total_loss'].item():.4f}")
    except Exception as e:
        print(f"   ❌ Loss computation failed: {e}")
        return False

    # Test optimizer and scheduler
    print("4. Testing optimizer and scheduler...")
    try:
        optimizer = create_optimizer(model, optimizer_type="adamw", learning_rate=1e-3)
        scheduler = create_scheduler(optimizer, scheduler_type="cosine_warmup", num_epochs=10)

        # Test backward pass
        loss_dict['total_loss'].backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        print("   ✅ Optimizer and scheduler working")
    except Exception as e:
        print(f"   ❌ Optimizer/scheduler failed: {e}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Dummy training for LHA-Net")
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--num-cases', type=int, default=10, help='Number of synthetic cases')
    parser.add_argument('--patches-per-volume', type=int, default=8, help='Patches per volume')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--config', type=str, default='lightweight',
                        choices=['lightweight', 'standard'], help='Model configuration')

    args = parser.parse_args()

    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print("LHA-Net Dummy Training Script")
    print("=" * 50)
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    # Test components first
    if not test_model_components():
        print("\n❌ Component tests failed!")
        return 1

    print("\n" + "=" * 50)
    print("Starting Dummy Training...")

    # Create model
    if args.config == 'lightweight':
        model = create_lha_net(config_type="lightweight", num_classes=14)
    else:
        model = LHANetWithAuxiliaryLoss(
            num_classes=14,
            backbone_type="resnet18",
            use_lightweight=False,
            base_channels=64
        )

    model = model.to(device)

    # Create dataset
    dataset = DummyDataset(
        num_cases=args.num_cases,
        patches_per_volume=args.patches_per_volume
    )

    # Create loss function
    loss_fn = LHANetLoss()

    # Create optimizer and scheduler
    optimizer = create_optimizer(
        model,
        optimizer_type="adamw",
        learning_rate=1e-3,
        weight_decay=1e-4
    )

    scheduler = create_scheduler(
        optimizer,
        scheduler_type="cosine_warmup",
        num_epochs=args.epochs,
        warmup_epochs=1
    )

    # Run training
    try:
        history = dummy_training_loop(
            model=model,
            dataset=dataset,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_epochs=args.epochs
        )

        # Print final results
        print("\n" + "=" * 50)
        print("Training Complete!")
        print(f"Final Loss: {history['losses'][-1]:.4f}")
        print(f"Final Dice: {history['dice_scores'][-1]:.3f}")

        if device.type == 'cuda' and history['memory_usage']:
            final_memory = history['memory_usage'][-1]
            print(f"Peak GPU Memory: {final_memory.get('gpu_max_allocated_gb', 0):.2f}GB")

        print("\n✅ Dummy training completed successfully!")
        print("The model and training pipeline are working correctly.")
        print("You can now proceed with real AMOS22 data training.")

        return 0

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)