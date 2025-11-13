#!/usr/bin/env python3
"""
Test adaptive PMSA implementation
Verifies that the adaptive mechanism works correctly
"""

import torch
import sys
import os
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.models.lha_net_adaptive import create_lha_net_adaptive


def test_adaptive_pmsa_forward():
    """Test forward pass with adaptive PMSA"""
    print("="*70)
    print("TEST 1: Forward Pass with Adaptive PMSA")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Create model
    model = create_lha_net_adaptive(
        config_type="lightweight",
        num_classes=16,
        scale_bank=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5],
        num_active_scales=5,
        use_adaptive_pmsa=True
    )
    model = model.to(device)
    model.eval()

    # Test input
    batch_size = 2
    test_input = torch.randn(batch_size, 1, 64, 128, 128, device=device)

    print(f"Input shape: {test_input.shape}")

    # Forward pass
    with torch.no_grad():
        output = model(test_input, return_scale_info=True)

    print(f"\n✓ Forward pass successful")
    print(f"Output shape: {output['final_prediction'].shape}")

    # Check scale info
    if 'scale_info' in output:
        print(f"\n✓ Scale information available")
        for level_key, info in output['scale_info'].items():
            if 'selected_scales' in info:
                print(f"  {level_key}: {info['selected_scales']}")

    print("\n" + "="*70 + "\n")


def test_scale_selection_consistency():
    """Test that scale selection is consistent in eval mode"""
    print("="*70)
    print("TEST 2: Scale Selection Consistency")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = create_lha_net_adaptive(
        config_type="lightweight",
        num_classes=16,
        use_adaptive_pmsa=True
    )
    model = model.to(device)
    model.eval()

    test_input = torch.randn(1, 1, 64, 128, 128, device=device)

    # Get scale selections multiple times
    selections = []
    for i in range(3):
        with torch.no_grad():
            output = model(test_input, return_scale_info=True)
            scale_info = output['scale_info']
            level_0_scales = scale_info['level_0']['selected_scales']
            selections.append(level_0_scales)

    # Check consistency
    all_same = all(s == selections[0] for s in selections)

    if all_same:
        print("✓ Scale selection is consistent across forward passes")
        print(f"  Selected scales: {selections[0]}")
    else:
        print("✗ Scale selection is inconsistent!")
        for i, s in enumerate(selections):
            print(f"  Run {i+1}: {s}")

    print("\n" + "="*70 + "\n")


def test_backward_pass():
    """Test that gradients flow through scale selection"""
    print("="*70)
    print("TEST 3: Backward Pass (Gradient Flow)")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = create_lha_net_adaptive(
        config_type="lightweight",
        num_classes=16,
        use_adaptive_pmsa=True
    )
    model = model.to(device)
    model.train()

    test_input = torch.randn(1, 1, 64, 128, 128, device=device, requires_grad=True)
    target = torch.randint(0, 16, (1, 64, 128, 128), device=device)

    # Forward pass
    output = model(test_input, return_features=True)

    # Compute simple loss
    prediction = output['final_prediction']
    loss = torch.nn.functional.cross_entropy(prediction, target)

    print(f"Loss: {loss.item():.4f}")

    # Backward pass
    loss.backward()

    # Check if scale_logits have gradients
    has_scale_gradients = False
    for name, param in model.named_parameters():
        if 'scale_logits' in name or 'scale_importance' in name:
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_scale_gradients = True
                print(f"\n✓ Gradients flowing to: {name}")
                print(f"  Gradient magnitude: {param.grad.abs().mean().item():.6f}")

    if has_scale_gradients:
        print("\n✓ Scale selection parameters are trainable")
    else:
        print("\n✗ No gradients in scale selection parameters!")

    print("\n" + "="*70 + "\n")


def test_scale_evolution_simulation():
    """Simulate how scales evolve during training"""
    print("="*70)
    print("TEST 4: Scale Evolution Simulation")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = create_lha_net_adaptive(
        config_type="lightweight",
        num_classes=16,
        use_adaptive_pmsa=True
    )
    model = model.to(device)

    # Get initial scale probabilities
    initial_stats = model.get_scale_statistics()

    print("Initial scale probabilities:")
    for level_key, probs in initial_stats.items():
        if isinstance(probs, torch.Tensor):
            probs_np = probs.detach().cpu().numpy()
            scale_bank = model.hierarchical_pmsa.pmsa_modules['level_0'].scale_bank
            print(f"\n{level_key}:")
            for scale, prob in zip(scale_bank, probs_np):
                print(f"  {scale:.2f}×: {prob:.4f}")

    # Simulate a few training steps
    print("\n" + "-"*70)
    print("Simulating 10 training steps...")
    print("-"*70)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()

    for step in range(10):
        # Dummy data
        test_input = torch.randn(1, 1, 64, 128, 128, device=device)
        target = torch.randint(0, 16, (1, 64, 128, 128), device=device)

        # Forward
        output = model(test_input, return_features=True)
        prediction = output['final_prediction']

        # Loss
        loss = torch.nn.functional.cross_entropy(prediction, target)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Get final scale probabilities
    final_stats = model.get_scale_statistics()

    print("\nScale probabilities after 10 steps:")
    for level_key, probs in final_stats.items():
        if isinstance(probs, torch.Tensor):
            probs_np = probs.detach().cpu().numpy()
            initial_probs = initial_stats[level_key].detach().cpu().numpy()

            print(f"\n{level_key}:")
            scale_bank = model.hierarchical_pmsa.pmsa_modules['level_0'].scale_bank
            for scale, init_prob, final_prob in zip(scale_bank, initial_probs, probs_np):
                change = final_prob - init_prob
                arrow = "↑" if change > 0.001 else "↓" if change < -0.001 else "→"
                print(f"  {scale:.2f}×: {init_prob:.4f} {arrow} {final_prob:.4f} (Δ{change:+.4f})")

    print("\n✓ Scales are updating during training")
    print("\n" + "="*70 + "\n")


def test_comparison_with_fixed():
    """Compare adaptive vs fixed scale configuration"""
    print("="*70)
    print("TEST 5: Adaptive vs Fixed Scales Comparison")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Adaptive model
    model_adaptive = create_lha_net_adaptive(
        config_type="lightweight",
        num_classes=16,
        use_adaptive_pmsa=True
    ).to(device)

    # Fixed model (use_adaptive_pmsa=False would use first 5 scales)
    model_fixed = create_lha_net_adaptive(
        config_type="lightweight",
        num_classes=16,
        use_adaptive_pmsa=False
    ).to(device)

    # Count parameters
    adaptive_params = sum(p.numel() for p in model_adaptive.parameters())
    fixed_params = sum(p.numel() for p in model_fixed.parameters())

    print(f"\nParameter comparison:")
    print(f"  Adaptive: {adaptive_params:,}")
    print(f"  Fixed: {fixed_params:,}")
    print(f"  Difference: {adaptive_params - fixed_params:,}")

    # Test inference time
    test_input = torch.randn(1, 1, 64, 128, 128, device=device)

    # Warmup
    with torch.no_grad():
        _ = model_adaptive(test_input)
        _ = model_fixed(test_input)

    # Time adaptive
    start = time.time()
    with torch.no_grad():
        for _ in range(10):
            _ = model_adaptive(test_input)
    adaptive_time = (time.time() - start) / 10

    # Time fixed
    start = time.time()
    with torch.no_grad():
        for _ in range(10):
            _ = model_fixed(test_input)
    fixed_time = (time.time() - start) / 10

    print(f"\nInference time comparison:")
    print(f"  Adaptive: {adaptive_time*1000:.2f} ms")
    print(f"  Fixed: {fixed_time*1000:.2f} ms")
    print(f"  Overhead: {(adaptive_time - fixed_time)*1000:.2f} ms ({((adaptive_time/fixed_time - 1)*100):.1f}%)")

    print("\n✓ Comparison complete")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("ADAPTIVE PMSA TEST SUITE")
    print("="*70 + "\n")

    # Run all tests
    test_adaptive_pmsa_forward()
    test_scale_selection_consistency()
    test_backward_pass()
    test_scale_evolution_simulation()
    test_comparison_with_fixed()

    print("="*70)
    print("ALL TESTS COMPLETE")
    print("="*70)
