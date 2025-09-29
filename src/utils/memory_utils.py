import torch
import gc
import psutil
import os
from typing import Dict, Optional, Any
import time
import warnings


class MemoryMonitor:
    """Monitor and manage memory usage during training"""

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_cuda = self.device.type == 'cuda'

        # Get system memory info
        self.system_memory_gb = psutil.virtual_memory().total / (1024**3)

        # Get GPU memory info if available
        if self.is_cuda:
            self.gpu_memory_gb = torch.cuda.get_device_properties(self.device).total_memory / (1024**3)
        else:
            self.gpu_memory_gb = None

        self.peak_memory_usage = 0.0
        self.memory_history = []

    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics"""
        stats = {}

        # System memory
        system_mem = psutil.virtual_memory()
        stats['system_memory_used_gb'] = system_mem.used / (1024**3)
        stats['system_memory_percent'] = system_mem.percent
        stats['system_memory_available_gb'] = system_mem.available / (1024**3)

        # Process memory
        process = psutil.Process(os.getpid())
        process_memory = process.memory_info()
        stats['process_memory_rss_gb'] = process_memory.rss / (1024**3)
        stats['process_memory_vms_gb'] = process_memory.vms / (1024**3)

        # GPU memory
        if self.is_cuda:
            gpu_memory = torch.cuda.memory_stats(self.device)
            stats['gpu_allocated_gb'] = torch.cuda.memory_allocated(self.device) / (1024**3)
            stats['gpu_reserved_gb'] = torch.cuda.memory_reserved(self.device) / (1024**3)
            stats['gpu_max_allocated_gb'] = torch.cuda.max_memory_allocated(self.device) / (1024**3)
            stats['gpu_max_reserved_gb'] = torch.cuda.max_memory_reserved(self.device) / (1024**3)

            # Update peak usage
            current_peak = stats['gpu_max_allocated_gb']
            if current_peak > self.peak_memory_usage:
                self.peak_memory_usage = current_peak

            # Memory fragmentation
            if stats['gpu_reserved_gb'] > 0:
                fragmentation = 1 - (stats['gpu_allocated_gb'] / stats['gpu_reserved_gb'])
                stats['gpu_fragmentation'] = fragmentation

        return stats

    def log_memory_usage(self, prefix: str = ""):
        """Log current memory usage"""
        stats = self.get_memory_stats()

        print(f"{prefix}Memory Usage:")
        print(f"  System: {stats['system_memory_used_gb']:.2f}GB ({stats['system_memory_percent']:.1f}%)")
        print(f"  Process: {stats['process_memory_rss_gb']:.2f}GB")

        if self.is_cuda:
            print(f"  GPU Allocated: {stats['gpu_allocated_gb']:.2f}GB")
            print(f"  GPU Reserved: {stats['gpu_reserved_gb']:.2f}GB")
            if 'gpu_fragmentation' in stats:
                print(f"  GPU Fragmentation: {stats['gpu_fragmentation']:.2%}")

    def check_memory_threshold(self, threshold: float = 0.85) -> bool:
        """Check if memory usage exceeds threshold"""
        if self.is_cuda:
            current_usage = torch.cuda.memory_allocated(self.device) / torch.cuda.get_device_properties(self.device).total_memory
            return current_usage > threshold
        else:
            system_mem = psutil.virtual_memory()
            return system_mem.percent / 100.0 > threshold

    def cleanup_memory(self):
        """Perform memory cleanup"""
        if self.is_cuda:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        gc.collect()

    def get_memory_summary(self) -> Dict[str, Any]:
        """Get comprehensive memory summary"""
        stats = self.get_memory_stats()

        summary = {
            'current_stats': stats,
            'peak_gpu_usage_gb': self.peak_memory_usage,
            'total_gpu_memory_gb': self.gpu_memory_gb,
            'total_system_memory_gb': self.system_memory_gb,
            'gpu_utilization_percent': (stats.get('gpu_allocated_gb', 0) / self.gpu_memory_gb * 100) if self.gpu_memory_gb else 0
        }

        return summary


def optimize_memory_usage(model: torch.nn.Module, enable_checkpointing: bool = True) -> torch.nn.Module:
    """Apply memory optimizations to a model"""

    # 1. Enable gradient checkpointing
    if enable_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()

    # 2. Convert BatchNorm to SyncBatchNorm if using distributed training
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    return model


class MemoryEfficientDataLoader:
    """Memory-efficient data loading utilities"""

    @staticmethod
    def pin_memory_if_available(tensor: torch.Tensor) -> torch.Tensor:
        """Pin memory if CUDA is available and tensor is on CPU"""
        if torch.cuda.is_available() and tensor.device.type == 'cpu':
            return tensor.pin_memory()
        return tensor

    @staticmethod
    def optimize_dataloader_memory(dataloader_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize DataLoader kwargs for memory efficiency"""
        optimized_kwargs = dataloader_kwargs.copy()

        # Enable pin memory for CUDA
        if torch.cuda.is_available():
            optimized_kwargs['pin_memory'] = True

        # Adjust num_workers based on available CPU cores
        available_cores = os.cpu_count()
        if 'num_workers' not in optimized_kwargs:
            # Use fewer workers to reduce memory overhead
            optimized_kwargs['num_workers'] = min(4, available_cores // 2)

        # Enable prefetch_factor for PyTorch >= 1.8
        if hasattr(torch.utils.data.DataLoader, 'prefetch_factor'):
            optimized_kwargs['prefetch_factor'] = 2

        return optimized_kwargs


class GradientAccumulationManager:
    """Manage gradient accumulation for memory efficiency"""

    def __init__(self, accumulation_steps: int = 4):
        self.accumulation_steps = accumulation_steps
        self.current_step = 0

    def should_update(self) -> bool:
        """Check if gradients should be updated"""
        self.current_step += 1
        return (self.current_step % self.accumulation_steps) == 0

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for gradient accumulation"""
        return loss / self.accumulation_steps

    def reset(self):
        """Reset step counter"""
        self.current_step = 0


def estimate_model_memory(model: torch.nn.Module, input_shape: tuple, dtype: torch.dtype = torch.float32) -> Dict[str, float]:
    """Estimate memory usage of a model"""

    # Calculate parameter memory
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters())

    # Calculate buffer memory
    buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers())

    # Estimate activation memory (very rough approximation)
    dummy_input = torch.randn(input_shape, dtype=dtype)
    activation_memory = dummy_input.numel() * dummy_input.element_size()

    # Factor in intermediate activations (rough estimate)
    # Assume each layer roughly doubles the activation memory temporarily
    num_layers = len(list(model.modules()))
    estimated_activation_memory = activation_memory * num_layers * 2

    # Convert to MB
    param_memory_mb = param_memory / (1024 * 1024)
    buffer_memory_mb = buffer_memory / (1024 * 1024)
    activation_memory_mb = estimated_activation_memory / (1024 * 1024)

    total_memory_mb = param_memory_mb + buffer_memory_mb + activation_memory_mb

    return {
        'parameter_memory_mb': param_memory_mb,
        'buffer_memory_mb': buffer_memory_mb,
        'estimated_activation_memory_mb': activation_memory_mb,
        'total_estimated_memory_mb': total_memory_mb,
        'parameter_count': sum(p.numel() for p in model.parameters())
    }


class MemoryProfiler:
    """Profile memory usage during training"""

    def __init__(self, log_interval: int = 100):
        self.log_interval = log_interval
        self.step_count = 0
        self.memory_log = []

    def step(self, prefix: str = ""):
        """Log memory at current step"""
        self.step_count += 1

        if self.step_count % self.log_interval == 0:
            if torch.cuda.is_available():
                current_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
                max_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB

                log_entry = {
                    'step': self.step_count,
                    'current_memory_mb': current_memory,
                    'max_memory_mb': max_memory,
                    'timestamp': time.time()
                }

                self.memory_log.append(log_entry)

                if prefix:
                    print(f"{prefix} Step {self.step_count}: {current_memory:.1f}MB current, {max_memory:.1f}MB peak")

    def get_memory_log(self) -> list:
        """Get the memory log"""
        return self.memory_log

    def reset_peak_memory(self):
        """Reset peak memory statistics"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()


# Utility functions for memory management
def get_gpu_memory_info() -> Dict[str, float]:
    """Get GPU memory information"""
    if not torch.cuda.is_available():
        return {'cuda_available': False}

    device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)  # GB
    allocated_memory = torch.cuda.memory_allocated(device) / (1024**3)  # GB
    reserved_memory = torch.cuda.memory_reserved(device) / (1024**3)  # GB

    return {
        'cuda_available': True,
        'device_id': device,
        'total_memory_gb': total_memory,
        'allocated_memory_gb': allocated_memory,
        'reserved_memory_gb': reserved_memory,
        'free_memory_gb': total_memory - reserved_memory,
        'utilization_percent': (allocated_memory / total_memory) * 100
    }


def clear_gpu_cache():
    """Clear GPU cache and run garbage collection"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def set_memory_growth(enabled: bool = True):
    """Enable/disable memory growth for better memory management"""
    if torch.cuda.is_available() and hasattr(torch.cuda, 'set_memory_growth'):
        try:
            torch.cuda.set_memory_growth(enabled)
        except AttributeError:
            # Not available in all PyTorch versions
            pass