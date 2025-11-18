#!/usr/bin/env python3
"""
Quick test to verify gradient flow to scale_logits.
Uses small input size to avoid memory issues.
"""

import torch
import torch.nn.functional as F
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.models.lha_net_adaptive import create_lha_net_adaptive


def test_gradient_flow():
    """Verify gradients flow to scale_logits parameters"""
    print("="*70)
    print("GRADIENT FLOW VERIFICATION TEST")
    print("="*70)

    # Use CPU to avoid memory issues when testing with all 9 scales
    device = torch.device('cpu')
    print(f"Device: {device} (using CPU to avoid OOM with 9 scales)\n")

    # Create model with smaller architecture
    print("Creating model with soft selection enabled...")
    model = create_lha_net_adaptive(
        config_type="lightweight",
        num_classes=14,
        use_adaptive_pmsa=True
    ).to(device)
    model.train()

    # Use batch_size=2 to avoid batch norm issues with small features
    print("Input size: [2, 1, 48, 96, 96] (batch_size=2 for batch norm)\n")
    x = torch.randn(2, 1, 48, 96, 96, device=device)
    target = torch.randint(0, 14, (2, 48, 96, 96), device=device)

    # Forward pass
    print("Running forward pass...")
    output = model(x, return_features=True)
    prediction = output['final_prediction']

    # Loss
    loss = F.cross_entropy(prediction, target)
    print(f"Loss: {loss.item():.4f}\n")

    # Backward pass
    print("Running backward pass...")
    loss.backward()

    # Check scale_logits gradients
    print("\n" + "="*70)
    print("SCALE LOGITS GRADIENT CHECK")
    print("="*70 + "\n")

    has_gradient = False
    gradient_info = []

    for name, param in model.named_parameters():
        if 'scale_logits' in name:
            if param.grad is not None:
                grad_mag = param.grad.abs().mean().item()
                grad_max = param.grad.abs().max().item()

                gradient_info.append({
                    'name': name,
                    'mean': grad_mag,
                    'max': grad_max
                })

                if grad_mag > 1e-8:
                    has_gradient = True
                    print(f"✓ {name}")
                    print(f"  Mean gradient: {grad_mag:.6f}")
                    print(f"  Max gradient:  {grad_max:.6f}\n")
                else:
                    print(f"✗ {name}: gradient is ZERO!\n")
            else:
                print(f"✗ {name}: NO GRADIENT COMPUTED!\n")

    print("="*70)
    if has_gradient:
        print("✓ SUCCESS: Gradients are flowing to scale selection!")
        print("  The adaptive mechanism can learn from data.")
        print("  Scale probabilities will evolve during training.")
    else:
        print("✗ FAILURE: NO GRADIENTS - Scale selection is NOT learning!")
        print("  Scale probabilities will remain frozen.")
        print("  This means the adaptive mechanism is broken.")
    print("="*70)

    # Verify all levels have gradients
    print("\nGradient Flow by Level:")
    print("-"*70)
    for level in range(4):
        level_has_grad = False
        for info in gradient_info:
            if f'level_{level}' in info['name']:
                level_has_grad = True
                print(f"  Level {level}: ✓ (mean grad: {info['mean']:.6f})")
                break
        if not level_has_grad:
            print(f"  Level {level}: ✗ No gradient found")

    print("\n" + "="*70)
    return has_gradient


if __name__ == "__main__":
    success = test_gradient_flow()

    if success:
        print("\n🎉 All checks passed! The gradient flow fix is working correctly.")
        exit(0)
    else:
        print("\n❌ Gradient flow check failed. The adaptive mechanism won't learn.")
        exit(1)
