#!/usr/bin/env python3
"""
Demo script to test LHA-Net model with various configurations
Optimized for 16GB GPU and 32GB RAM
"""

import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models.lha_net import create_lha_net, create_lha_net_with_auxiliary


def format_memory(bytes_val):
    """Format bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} TB"


def test_model_config(config_name, input_shape, num_classes=14, use_cuda=True):
    """Test a specific model configuration"""
    print(f"\n{'='*60}")
    print(f"Testing {config_name} configuration")
    print(f"Input shape: {input_shape}")
    print(f"{'='*60}")

    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')

    # Create model
    model = create_lha_net(config_type=config_name, num_classes=num_classes)
    model = model.to(device)
    model.eval()

    # Model statistics
    model_info = model.get_model_size()
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {model_info['total_parameters']:,}")
    print(f"  Trainable parameters: {model_info['trainable_parameters']:,}")
    print(f"  Backbone parameters: {model_info['backbone_parameters']:,}")
    print(f"  PMSA parameters: {model_info['pmsa_parameters']:,}")
    print(f"  Decoder parameters: {model_info['decoder_parameters']:,}")

    # Estimate model size in memory
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    print(f"  Model size in memory: {model_size_mb:.2f} MB")

    # Test forward pass
    print(f"\nRunning forward pass...")
    x = torch.randn(input_shape, device=device)

    if use_cuda and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    with torch.no_grad():
        output = model(x)

    if use_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        current_memory = torch.cuda.memory_allocated(device) / (1024 ** 2)
        print(f"\nGPU Memory Usage:")
        print(f"  Peak memory: {peak_memory:.2f} MB")
        print(f"  Current memory: {current_memory:.2f} MB")

    if isinstance(output, dict):
        output = output['final_prediction']

    print(f"\nOutput shape: {output.shape}")
    print(f"✅ Test passed!")

    # Clean up
    del model, x, output
    if use_cuda and torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    print("LHA-Net Model Demo")
    print("="*60)

    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        use_cuda = True
    else:
        print("CUDA not available. Running on CPU.")
        use_cuda = False

    # Test configurations suitable for 16GB GPU
    test_cases = [
        # (config_name, batch_size, depth, height, width)
        ("lightweight", 1, 64, 128, 128),    # Small volume
        ("lightweight", 1, 96, 192, 192),    # Medium volume
        ("lightweight", 2, 64, 128, 128),    # Batch of 2
        ("standard", 1, 64, 128, 128),       # Standard config
    ]

    for config_name, batch_size, d, h, w in test_cases:
        input_shape = (batch_size, 1, d, h, w)
        try:
            test_model_config(config_name, input_shape, use_cuda=use_cuda)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n❌ Out of memory for {config_name} with input {input_shape}")
                if use_cuda:
                    torch.cuda.empty_cache()
            else:
                raise e

    print("\n" + "="*60)
    print("Demo completed!")
    print("="*60)


if __name__ == "__main__":
    main()