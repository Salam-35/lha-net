import torch
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, Optional, Any
import warnings


class MixedPrecisionTraining:
    """Mixed precision training utilities for memory optimization"""

    def __init__(
        self,
        enabled: bool = True,
        init_scale: float = 65536.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        enabled_max_scale: float = 65536.0
    ):
        self.enabled = enabled and torch.cuda.is_available()

        if self.enabled:
            self.scaler = GradScaler(
                init_scale=init_scale,
                growth_factor=growth_factor,
                backoff_factor=backoff_factor,
                growth_interval=growth_interval,
                enabled=True
            )
        else:
            self.scaler = None

        self.autocast_context = autocast if self.enabled else self._dummy_context

    def _dummy_context(self):
        """Dummy context manager for when mixed precision is disabled"""
        class DummyContext:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        return DummyContext()

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for backward pass"""
        if self.enabled and self.scaler is not None:
            return self.scaler.scale(loss)
        return loss

    def step_optimizer(self, optimizer: torch.optim.Optimizer) -> bool:
        """Step optimizer with proper scaling"""
        if self.enabled and self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
            return True
        else:
            optimizer.step()
            return True

    def backward(self, loss: torch.Tensor) -> None:
        """Backward pass with scaling"""
        if self.enabled and self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def get_context(self):
        """Get autocast context manager"""
        return self.autocast_context()

    def get_scale(self) -> float:
        """Get current loss scale"""
        if self.enabled and self.scaler is not None:
            return self.scaler.get_scale()
        return 1.0

    def state_dict(self) -> Dict[str, Any]:
        """Get state dict for checkpointing"""
        if self.enabled and self.scaler is not None:
            return self.scaler.state_dict()
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dict from checkpoint"""
        if self.enabled and self.scaler is not None:
            self.scaler.load_state_dict(state_dict)


class GradientAccumulation:
    """Gradient accumulation for effective larger batch sizes"""

    def __init__(
        self,
        accumulation_steps: int = 8,
        max_grad_norm: Optional[float] = 1.0,
        normalize_grad_by_steps: bool = True
    ):
        self.accumulation_steps = accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.normalize_grad_by_steps = normalize_grad_by_steps
        self.current_step = 0

    def accumulate_gradients(
        self,
        loss: torch.Tensor,
        model: torch.nn.Module,
        mixed_precision: Optional[MixedPrecisionTraining] = None
    ) -> bool:
        """
        Accumulate gradients and return True if optimizer step should be taken

        Args:
            loss: Current batch loss
            model: Model to accumulate gradients for
            mixed_precision: Mixed precision training handler

        Returns:
            True if gradients should be stepped, False otherwise
        """
        # Normalize loss by accumulation steps if required
        if self.normalize_grad_by_steps:
            loss = loss / self.accumulation_steps

        # Backward pass
        if mixed_precision is not None:
            mixed_precision.backward(loss)
        else:
            loss.backward()

        self.current_step += 1

        # Check if we should step
        should_step = (self.current_step % self.accumulation_steps) == 0

        if should_step:
            # Clip gradients if specified
            if self.max_grad_norm is not None:
                if mixed_precision is not None and mixed_precision.enabled:
                    mixed_precision.scaler.unscale_(mixed_precision.scaler._optimizers[0] if mixed_precision.scaler._optimizers else None)

                torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)

        return should_step

    def reset(self):
        """Reset accumulation counter"""
        self.current_step = 0


class MemoryOptimization:
    """Memory optimization utilities"""

    def __init__(
        self,
        enable_checkpointing: bool = True,
        empty_cache_freq: int = 10,
        pin_memory: bool = True
    ):
        self.enable_checkpointing = enable_checkpointing
        self.empty_cache_freq = empty_cache_freq
        self.pin_memory = pin_memory
        self._step_count = 0

    def optimize_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply memory optimizations to model"""
        if self.enable_checkpointing:
            # Apply gradient checkpointing to save memory
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
            else:
                # Manual gradient checkpointing for custom models
                self._apply_gradient_checkpointing(model)

        return model

    def _apply_gradient_checkpointing(self, model: torch.nn.Module):
        """Apply gradient checkpointing to specific modules"""
        from torch.utils.checkpoint import checkpoint

        # Apply checkpointing to memory-intensive modules
        for name, module in model.named_modules():
            if 'pmsa' in name.lower() or 'attention' in name.lower():
                # Wrap forward method with checkpointing
                original_forward = module.forward

                def checkpointed_forward(*args, **kwargs):
                    return checkpoint(original_forward, *args, **kwargs)

                module.forward = checkpointed_forward

    def step_memory_management(self):
        """Memory management step (call after each training step)"""
        self._step_count += 1

        if self._step_count % self.empty_cache_freq == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics"""
        if not torch.cuda.is_available():
            return {'cuda_available': False}

        stats = {
            'cuda_available': True,
            'allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
            'reserved_mb': torch.cuda.memory_reserved() / 1024 / 1024,
            'max_allocated_mb': torch.cuda.max_memory_allocated() / 1024 / 1024,
            'max_reserved_mb': torch.cuda.max_memory_reserved() / 1024 / 1024
        }

        return stats

    def reset_peak_memory_stats(self):
        """Reset peak memory statistics"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()


class DynamicBatchSizing:
    """Dynamic batch sizing based on available memory"""

    def __init__(
        self,
        initial_batch_size: int = 2,
        min_batch_size: int = 1,
        max_batch_size: int = 8,
        memory_threshold: float = 0.85,  # 85% of GPU memory
        increase_factor: float = 1.2,
        decrease_factor: float = 0.8
    ):
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.memory_threshold = memory_threshold
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor

        self.consecutive_successful_steps = 0
        self.last_oom_step = -1000

    def adjust_batch_size(self, step: int, memory_stats: Dict[str, float]) -> int:
        """Adjust batch size based on memory usage"""
        if not memory_stats.get('cuda_available', False):
            return self.current_batch_size

        # Get total GPU memory
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024  # MB
        memory_usage_ratio = memory_stats['allocated_mb'] / total_memory

        # Decrease batch size if memory usage is high
        if memory_usage_ratio > self.memory_threshold:
            new_batch_size = max(
                self.min_batch_size,
                int(self.current_batch_size * self.decrease_factor)
            )
            if new_batch_size != self.current_batch_size:
                print(f"Decreasing batch size from {self.current_batch_size} to {new_batch_size} "
                      f"(memory usage: {memory_usage_ratio:.2%})")
                self.current_batch_size = new_batch_size
                self.consecutive_successful_steps = 0

        # Increase batch size if memory usage is low and we haven't had OOM recently
        elif (memory_usage_ratio < 0.6 and
              self.consecutive_successful_steps > 50 and
              step - self.last_oom_step > 100):

            new_batch_size = min(
                self.max_batch_size,
                int(self.current_batch_size * self.increase_factor)
            )
            if new_batch_size != self.current_batch_size:
                print(f"Increasing batch size from {self.current_batch_size} to {new_batch_size} "
                      f"(memory usage: {memory_usage_ratio:.2%})")
                self.current_batch_size = new_batch_size
                self.consecutive_successful_steps = 0

        self.consecutive_successful_steps += 1
        return self.current_batch_size

    def handle_oom(self, step: int) -> int:
        """Handle out-of-memory error"""
        self.last_oom_step = step
        self.consecutive_successful_steps = 0

        new_batch_size = max(self.min_batch_size, self.current_batch_size - 1)
        print(f"OOM detected! Reducing batch size from {self.current_batch_size} to {new_batch_size}")
        self.current_batch_size = new_batch_size

        return self.current_batch_size