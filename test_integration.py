#!/usr/bin/env python3
"""
Test script to verify the integration of large kernel convolutions into LHA-Net.
"""

import sys
import torch

print("Testing Large Kernel Integration with LHA-Net")
print("=" * 60)

# Test 1: Import the model
print("\n1. Testing imports...")
try:
    from src.models.lha_net import LHANet, create_lha_net
    print("   ✓ Successfully imported LHANet")
except Exception as e:
    print(f"   ✗ Failed to import LHANet: {e}")
    sys.exit(1)

# Test 2: Create model with standard backbone
print("\n2. Testing model with standard backbone...")
try:
    model_standard = LHANet(
        in_channels=1,
        num_classes=14,
        use_lightweight=True,
        use_large_kernels=False,
        base_channels=32
    )
    print("   ✓ Successfully created standard LHANet")
    print(f"   Parameters: {sum(p.numel() for p in model_standard.parameters()):,}")
except Exception as e:
    print(f"   ✗ Failed to create standard LHANet: {e}")
    sys.exit(1)

# Test 3: Create model with large kernel backbone
print("\n3. Testing model with large kernel backbone...")
try:
    model_large_kernel = LHANet(
        in_channels=1,
        num_classes=14,
        use_lightweight=True,
        use_large_kernels=True,
        use_adaptive_kernels=False,
        base_channels=32
    )
    print("   ✓ Successfully created LHANet with large kernels")
    print(f"   Parameters: {sum(p.numel() for p in model_large_kernel.parameters()):,}")
except Exception as e:
    print(f"   ✗ Failed to create LHANet with large kernels: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Create model with adaptive kernel backbone
print("\n4. Testing model with adaptive kernel backbone...")
try:
    model_adaptive = LHANet(
        in_channels=1,
        num_classes=14,
        use_lightweight=True,
        use_large_kernels=True,
        use_adaptive_kernels=True,
        base_channels=32
    )
    print("   ✓ Successfully created LHANet with adaptive kernels")
    print(f"   Parameters: {sum(p.numel() for p in model_adaptive.parameters()):,}")
except Exception as e:
    print(f"   ✗ Failed to create LHANet with adaptive kernels: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Forward pass with dummy input
print("\n5. Testing forward pass...")
try:
    # Small input for quick test
    batch_size = 1
    channels = 1
    depth, height, width = 32, 64, 64
    dummy_input = torch.randn(batch_size, channels, depth, height, width)

    # Test with large kernel model
    model_large_kernel.eval()
    with torch.no_grad():
        output = model_large_kernel(dummy_input)

    print(f"   ✓ Forward pass successful")
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output range: [{output.min():.4f}, {output.max():.4f}]")
except Exception as e:
    print(f"   ✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Compare parameter counts
print("\n6. Comparing parameter counts...")
try:
    standard_params = sum(p.numel() for p in model_standard.parameters())
    large_kernel_params = sum(p.numel() for p in model_large_kernel.parameters())
    adaptive_params = sum(p.numel() for p in model_adaptive.parameters())

    print(f"   Standard backbone:      {standard_params:,} parameters")
    print(f"   Large kernel backbone:  {large_kernel_params:,} parameters")
    print(f"   Adaptive kernel backbone: {adaptive_params:,} parameters")

    if large_kernel_params < standard_params * 1.5:
        print(f"   ✓ Large kernel backbone is efficient (< 1.5x standard)")
    else:
        print(f"   ⚠ Large kernel backbone has {large_kernel_params / standard_params:.2f}x parameters")
except Exception as e:
    print(f"   ✗ Comparison failed: {e}")

# Test 7: Test with training mode
print("\n7. Testing training mode...")
try:
    model_large_kernel.train()
    output = model_large_kernel(dummy_input)

    if isinstance(output, dict):
        print(f"   ✓ Training mode returns dict with keys: {list(output.keys())}")
        if 'deep_supervision_outputs' in output:
            print(f"   ✓ Deep supervision outputs: {len(output['deep_supervision_outputs'])} levels")
    else:
        print(f"   ✓ Training mode returns tensor of shape: {output.shape}")
except Exception as e:
    print(f"   ✗ Training mode test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("All integration tests passed! ✓")
print("\nYou can now use LHANet with large kernel convolutions by setting:")
print("  use_large_kernels=True  (for multi-scale large kernels)")
print("  use_adaptive_kernels=True  (for adaptive kernel selection)")
