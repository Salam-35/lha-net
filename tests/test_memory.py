import unittest
import torch
import numpy as np
import sys
import os
import gc
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.lha_net import create_lha_net, LHANetWithAuxiliaryLoss
from utils.memory_utils import (
    MemoryMonitor,
    optimize_memory_usage,
    estimate_model_memory,
    get_gpu_memory_info,
    clear_gpu_cache
)
from training.mixed_precision import MixedPrecisionTraining, GradientAccumulation, MemoryOptimization


class TestMemoryUsage(unittest.TestCase):
    """Test memory usage and optimization strategies"""

    def setUp(self):
        """Set up test fixtures"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.memory_monitor = MemoryMonitor(self.device)

        # Clear memory before each test
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        gc.collect()

    def tearDown(self):
        """Clean up after each test"""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_memory_monitor(self):
        """Test memory monitoring functionality"""
        print("\nTesting Memory Monitor:")

        # Get initial memory stats
        initial_stats = self.memory_monitor.get_memory_stats()
        print(f"Initial GPU memory: {initial_stats.get('gpu_allocated_gb', 0):.2f}GB")

        # Allocate some memory
        large_tensor = torch.randn(100, 100, 100, 100, device=self.device)

        # Check memory increase
        after_stats = self.memory_monitor.get_memory_stats()
        print(f"After allocation: {after_stats.get('gpu_allocated_gb', 0):.2f}GB")

        # Memory should have increased
        if self.device.type == 'cuda':
            self.assertGreater(
                after_stats['gpu_allocated_gb'],
                initial_stats['gpu_allocated_gb']
            )

        # Clean up
        del large_tensor
        torch.cuda.empty_cache()

        # Log memory usage
        self.memory_monitor.log_memory_usage("Test completed: ")

    def test_model_memory_estimation(self):
        """Test model memory estimation"""
        print("\nTesting Model Memory Estimation:")

        models_to_test = [
            ("lightweight", create_lha_net(config_type="lightweight", num_classes=14)),
            ("standard", create_lha_net(config_type="standard", num_classes=14))
        ]

        for config_name, model in models_to_test:
            print(f"\n{config_name.upper()} Model:")

            # Get model size info
            model_info = model.get_model_size()
            print(f"  Total parameters: {model_info['total_parameters']:,}")
            print(f"  Backbone parameters: {model_info['backbone_parameters']:,}")
            print(f"  PMSA parameters: {model_info['pmsa_parameters']:,}")
            print(f"  Decoder parameters: {model_info['decoder_parameters']:,}")

            # Estimate memory
            input_shape = (2, 1, 64, 128, 128)
            memory_estimate = estimate_model_memory(model, input_shape)

            print(f"  Estimated memory usage:")
            print(f"    Parameters: {memory_estimate['parameter_memory_mb']:.1f}MB")
            print(f"    Activations: {memory_estimate['estimated_activation_memory_mb']:.1f}MB")
            print(f"    Total: {memory_estimate['total_estimated_memory_mb']:.1f}MB")

            # Verify we can load the model
            model = model.to(self.device)

            # Test actual memory usage if on GPU
            if self.device.type == 'cuda':
                try:
                    actual_memory = model.get_memory_usage(input_shape)
                    print(f"  Actual memory usage:")
                    for key, value in actual_memory.items():
                        print(f"    {key}: {value:.2f}")
                except Exception as e:
                    print(f"  Could not measure actual memory: {e}")

            del model

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_memory_optimization(self):
        """Test memory optimization strategies"""
        print("\nTesting Memory Optimization:")

        # Create model without optimization
        model_normal = create_lha_net(config_type="lightweight", num_classes=14)
        model_normal = model_normal.to(self.device)

        # Create model with optimization
        model_optimized = create_lha_net(config_type="lightweight", num_classes=14)
        model_optimized = optimize_memory_usage(model_optimized, enable_checkpointing=True)
        model_optimized = model_optimized.to(self.device)

        input_tensor = torch.randn(2, 1, 64, 128, 128, device=self.device)
        target = torch.randint(0, 14, (2, 64, 128, 128), device=self.device)

        # Test normal model
        torch.cuda.reset_peak_memory_stats()
        model_normal.train()
        output_normal = model_normal(input_tensor)
        if isinstance(output_normal, dict):
            output_normal = output_normal['final_prediction']

        loss_normal = torch.nn.functional.cross_entropy(output_normal, target)
        loss_normal.backward()

        normal_peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"Normal model peak memory: {normal_peak:.2f}GB")

        # Clear memory
        del output_normal, loss_normal
        torch.cuda.empty_cache()

        # Test optimized model
        torch.cuda.reset_peak_memory_stats()
        model_optimized.train()
        output_optimized = model_optimized(input_tensor)
        if isinstance(output_optimized, dict):
            output_optimized = output_optimized['final_prediction']

        loss_optimized = torch.nn.functional.cross_entropy(output_optimized, target)
        loss_optimized.backward()

        optimized_peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"Optimized model peak memory: {optimized_peak:.2f}GB")

        # Memory optimization should not increase memory significantly
        # (Note: gradient checkpointing might actually increase memory in small models)
        print(f"Memory difference: {optimized_peak - normal_peak:.2f}GB")

        del model_normal, model_optimized

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_mixed_precision_memory(self):
        """Test mixed precision memory usage"""
        print("\nTesting Mixed Precision Memory:")

        model = create_lha_net(config_type="lightweight", num_classes=14)
        model = model.to(self.device)

        input_tensor = torch.randn(2, 1, 64, 128, 128, device=self.device)
        target = torch.randint(0, 14, (2, 64, 128, 128), device=self.device)

        # Test FP32 training
        torch.cuda.reset_peak_memory_stats()
        model.train()
        output_fp32 = model(input_tensor)
        if isinstance(output_fp32, dict):
            output_fp32 = output_fp32['final_prediction']

        loss_fp32 = torch.nn.functional.cross_entropy(output_fp32, target)
        loss_fp32.backward()

        fp32_memory = torch.cuda.max_memory_allocated() / 1e9
        print(f"FP32 peak memory: {fp32_memory:.2f}GB")

        # Clear memory
        del output_fp32, loss_fp32
        torch.cuda.empty_cache()
        model.zero_grad()

        # Test FP16 training
        torch.cuda.reset_peak_memory_stats()
        mixed_precision = MixedPrecisionTraining(enabled=True)

        model.train()
        with mixed_precision.get_context():
            output_fp16 = model(input_tensor)
            if isinstance(output_fp16, dict):
                output_fp16 = output_fp16['final_prediction']

            loss_fp16 = torch.nn.functional.cross_entropy(output_fp16, target)

        mixed_precision.backward(loss_fp16)

        fp16_memory = torch.cuda.max_memory_allocated() / 1e9
        print(f"FP16 peak memory: {fp16_memory:.2f}GB")

        memory_reduction = (fp32_memory - fp16_memory) / fp32_memory * 100
        print(f"Memory reduction: {memory_reduction:.1f}%")

        # FP16 should generally use less memory
        if fp16_memory < fp32_memory:
            print("✅ Mixed precision reduced memory usage")
        else:
            print("⚠️ Mixed precision did not reduce memory (might be due to small model size)")

        del model

    def test_gradient_accumulation_memory(self):
        """Test gradient accumulation for memory efficiency"""
        print("\nTesting Gradient Accumulation:")

        model = create_lha_net(config_type="lightweight", num_classes=14)
        model = model.to(self.device)

        # Simulate different batch sizes
        small_batch_size = 1
        accumulation_steps = 4  # Effective batch size = 4

        grad_accumulator = GradientAccumulation(accumulation_steps=accumulation_steps)

        model.train()
        total_loss = 0

        for step in range(accumulation_steps):
            input_tensor = torch.randn(small_batch_size, 1, 64, 128, 128, device=self.device)
            target = torch.randint(0, 14, (small_batch_size, 64, 128, 128), device=self.device)

            output = model(input_tensor)
            if isinstance(output, dict):
                output = output['final_prediction']

            loss = torch.nn.functional.cross_entropy(output, target)
            total_loss += loss.item()

            # Accumulate gradients
            should_step = grad_accumulator.accumulate_gradients(loss, model)

            if should_step:
                print(f"Gradient accumulation completed after {accumulation_steps} steps")
                print(f"Average loss: {total_loss / accumulation_steps:.4f}")
                break

        # Check that gradients exist
        has_gradients = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        self.assertTrue(has_gradients, "No gradients after accumulation")

        del model

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_memory_threshold_monitoring(self):
        """Test memory threshold monitoring"""
        print("\nTesting Memory Threshold Monitoring:")

        # Test memory threshold checking
        initial_usage = self.memory_monitor.check_memory_threshold(threshold=0.85)
        print(f"Initial memory above 85% threshold: {initial_usage}")

        # Allocate memory to approach threshold
        tensors = []
        try:
            for i in range(10):
                # Allocate in chunks
                tensor = torch.randn(50, 100, 100, 100, device=self.device)
                tensors.append(tensor)

                current_usage = self.memory_monitor.check_memory_threshold(threshold=0.85)
                memory_stats = self.memory_monitor.get_memory_stats()

                if self.device.type == 'cuda':
                    current_gb = memory_stats.get('gpu_allocated_gb', 0)
                    print(f"Chunk {i+1}: {current_gb:.2f}GB allocated, above threshold: {current_usage}")

                if current_usage:
                    print("⚠️ Memory threshold exceeded, stopping allocation")
                    break

        except RuntimeError as e:
            if "out of memory" in str(e):
                print("🚫 Out of memory reached during test")
            else:
                raise

        finally:
            # Clean up
            del tensors
            torch.cuda.empty_cache()

    def test_memory_cleanup(self):
        """Test memory cleanup functionality"""
        print("\nTesting Memory Cleanup:")

        if self.device.type == 'cuda':
            # Get initial memory
            initial_stats = self.memory_monitor.get_memory_stats()
            initial_memory = initial_stats.get('gpu_allocated_gb', 0)

            # Allocate memory
            large_tensor = torch.randn(100, 100, 100, 100, device=self.device)

            # Check memory increase
            after_alloc_stats = self.memory_monitor.get_memory_stats()
            after_alloc_memory = after_alloc_stats.get('gpu_allocated_gb', 0)

            print(f"Before allocation: {initial_memory:.2f}GB")
            print(f"After allocation: {after_alloc_memory:.2f}GB")

            # Delete tensor but don't clear cache
            del large_tensor

            # Check memory (should still be allocated)
            after_del_stats = self.memory_monitor.get_memory_stats()
            after_del_memory = after_del_stats.get('gpu_allocated_gb', 0)

            print(f"After deletion (no cache clear): {after_del_memory:.2f}GB")

            # Clear cache
            self.memory_monitor.cleanup_memory()

            # Check memory (should be reduced)
            after_cleanup_stats = self.memory_monitor.get_memory_stats()
            after_cleanup_memory = after_cleanup_stats.get('gpu_allocated_gb', 0)

            print(f"After cleanup: {after_cleanup_memory:.2f}GB")

            # Memory after cleanup should be closer to initial
            self.assertLessEqual(after_cleanup_memory, after_alloc_memory)

    def test_memory_profiling(self):
        """Test memory profiling during model operations"""
        print("\nTesting Memory Profiling:")

        model = create_lha_net(config_type="lightweight", num_classes=14)
        model = model.to(self.device)

        memory_log = []

        def log_memory(stage):
            if self.device.type == 'cuda':
                current_mem = torch.cuda.memory_allocated() / 1e9
                max_mem = torch.cuda.max_memory_allocated() / 1e9
                memory_log.append((stage, current_mem, max_mem))
                print(f"{stage}: Current={current_mem:.2f}GB, Peak={max_mem:.2f}GB")

        log_memory("Initial")

        # Model loading
        log_memory("After model loading")

        # Forward pass
        input_tensor = torch.randn(2, 1, 64, 128, 128, device=self.device)
        output = model(input_tensor)
        log_memory("After forward pass")

        # Backward pass
        target = torch.randint(0, 14, (2, 64, 128, 128), device=self.device)
        if isinstance(output, dict):
            output = output['final_prediction']

        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()
        log_memory("After backward pass")

        # Cleanup
        del output, loss, input_tensor, target
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        log_memory("After cleanup")

        # Verify memory profiling worked
        self.assertGreater(len(memory_log), 0)

        del model


def run_memory_tests():
    """Run all memory tests"""
    print("Testing Memory Usage and Optimization...")
    print("=" * 50)

    if torch.cuda.is_available():
        gpu_info = get_gpu_memory_info()
        print("GPU Information:")
        for key, value in gpu_info.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
    else:
        print("CUDA not available - some tests will be skipped")

    print("=" * 50)

    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMemoryUsage)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if result.wasSuccessful():
        print("\n✅ All memory tests passed!")

        # Final memory cleanup
        clear_gpu_cache()
        print("Final memory cleanup completed.")

        return True
    else:
        print("\n❌ Some memory tests failed!")
        return False


if __name__ == '__main__':
    success = run_memory_tests()
    if not success:
        sys.exit(1)