"""
Training loop with adaptive PMSA monitoring
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable
from pathlib import Path
import time
from tqdm import tqdm
import numpy as np

from ..utils.scale_analysis import ScaleAnalyzer


class AdaptivePMSATrainer:
    """
    Extended trainer with scale analysis capabilities.

    Monitors and logs scale selection during training to ensure
    the adaptive mechanism is learning properly.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        save_dir: str = 'checkpoints',
        log_scale_freq: int = 5,  # Log scale stats every N epochs
        visualize_scale_freq: int = 10,  # Visualize scales every N epochs
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.log_scale_freq = log_scale_freq
        self.visualize_scale_freq = visualize_scale_freq

        # Scale analyzer
        if hasattr(model, 'get_scale_statistics'):
            self.scale_analyzer = ScaleAnalyzer(
                model,
                save_dir=str(self.save_dir / 'scale_analysis')
            )
            self.use_scale_analysis = True
        else:
            self.use_scale_analysis = False
            print("Warning: Model does not support scale analysis")

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.scale_history = {}

        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()

        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")

        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            # Get model output
            if self.model.training and hasattr(self.model, 'use_deep_supervision'):
                output = self.model(images, return_features=True)
            else:
                output = self.model(images)

            # Compute loss
            loss_dict = self.loss_fn(output, labels)
            loss = loss_dict['total_loss']

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Track loss
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/num_batches:.4f}'
            })

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        # Import metrics
        from ..evaluation.metrics import SegmentationMetrics
        metrics_calculator = SegmentationMetrics(num_classes=self.model.num_classes)

        all_dice_scores = []

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")

        with torch.no_grad():
            for batch in pbar:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)

                # Forward pass
                output = self.model(images)

                if isinstance(output, dict):
                    predictions = output['final_prediction']
                else:
                    predictions = output

                # Compute loss
                loss_dict = self.loss_fn(predictions, labels)
                loss = loss_dict['total_loss']

                total_loss += loss.item()
                num_batches += 1

                # Compute metrics
                pred_labels = torch.argmax(predictions, dim=1)

                for pred, target in zip(pred_labels, labels):
                    metrics = metrics_calculator.compute_all_metrics(pred, target)
                    all_dice_scores.append(metrics['mean_dice'])

                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'dice': f'{np.mean(all_dice_scores):.4f}' if all_dice_scores else 'N/A'
                })

        avg_loss = total_loss / num_batches
        avg_dice = np.mean(all_dice_scores) if all_dice_scores else 0.0

        return {
            'val_loss': avg_loss,
            'val_dice': avg_dice
        }

    def train(
        self,
        num_epochs: int,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ):
        """Main training loop"""

        print("\n" + "="*70)
        print("STARTING TRAINING WITH ADAPTIVE PMSA")
        print("="*70)

        if self.use_scale_analysis:
            print("✓ Scale analysis enabled")
            print(f"  - Logging scale stats every {self.log_scale_freq} epochs")
            print(f"  - Visualizing scales every {self.visualize_scale_freq} epochs")

        print("\n")

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()

            # Train
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)

            # Validate
            val_metrics = self.validate(epoch)
            val_loss = val_metrics['val_loss']
            val_dice = val_metrics['val_dice']
            self.val_losses.append(val_loss)

            # Learning rate scheduling
            if scheduler is not None:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
            else:
                current_lr = self.optimizer.param_groups[0]['lr']

            epoch_time = time.time() - epoch_start

            # Print epoch summary
            print(f"\nEpoch {epoch}/{num_epochs} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Dice: {val_dice:.4f}")
            print(f"  LR: {current_lr:.6f}")
            print(f"  Time: {epoch_time:.1f}s")

            # Scale analysis
            if self.use_scale_analysis and epoch % self.log_scale_freq == 0:
                print("\n  Scale Analysis:")
                self.scale_analyzer.print_scale_summary()

                # Save scale history
                self.scale_history = self.scale_analyzer.save_scale_history(
                    epoch,
                    self.scale_history
                )

            # Visualize scales
            if self.use_scale_analysis and epoch % self.visualize_scale_freq == 0:
                self.scale_analyzer.plot_scale_distribution(epoch=epoch)

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch

                save_path = self.save_dir / 'best_model.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_dice': val_dice,
                }, save_path)

                print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")

            # Save checkpoint
            if epoch % 10 == 0:
                save_path = self.save_dir / f'checkpoint_epoch_{epoch}.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'scale_history': self.scale_history,
                }, save_path)

            print("-" * 70)

        # Final analysis
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"\nBest validation loss: {self.best_val_loss:.4f} (epoch {self.best_epoch})")

        if self.use_scale_analysis:
            print("\nFinal Scale Configuration:")
            self.scale_analyzer.print_scale_summary()

            # Compare with baseline
            self.scale_analyzer.compare_with_baseline()

            # Plot scale evolution
            if self.scale_history:
                self.scale_analyzer.plot_scale_evolution(self.scale_history)

        print("="*70)
