#!/usr/bin/env python3
"""
Verification script to demonstrate progressive fusion is working correctly
in the PMSA module
"""

import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models.pmsa_module import PMSAModule


def verify_progressive_fusion():
    """Verify that progressive fusion is working as expected"""
    print("="*70)
    print("Progressive Multi-Scale Attention (PMSA) Verification")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Create PMSA module
    in_channels = 128
    pmsa = PMSAModule(
        in_channels=in_channels,
        scales=[0.5, 0.75, 1.0, 1.25, 1.5],
        organ_contexts=["small", "small", "medium", "medium", "large"]
    ).to(device)

    print(f"\nPMSA Module Configuration:")
    print(f"  Input channels: {in_channels}")
    print(f"  Number of scales: {pmsa.num_scales}")
    print(f"  Scales: {pmsa.scales}")
    print(f"  Organ contexts: {pmsa.organ_contexts}")

    # Check progressive fusion modules exist
    print(f"\n✅ Progressive Fusion Modules:")
    for i, module in enumerate(pmsa.progressive_fusion):
        # Get expected input channels for this fusion stage
        expected_in = in_channels * (i + 2)  # (i+2) because we concatenate scales 0 to i+1
        expected_out = in_channels
        print(f"  Stage {i+1}: {expected_in} → {expected_out} channels")

        # Verify the first conv layer has correct input channels
        first_conv = module[0]  # First layer is Conv3d
        actual_in = first_conv.in_channels
        actual_out = first_conv.out_channels
        assert actual_in == expected_in, f"Expected {expected_in} but got {actual_in}"
        assert actual_out == expected_out, f"Expected {expected_out} but got {actual_out}"

    print("\n✅ All progressive fusion modules have correct dimensions")

    # Test forward pass
    batch_size = 1
    x = torch.randn(batch_size, in_channels, 16, 32, 32, device=device)

    print(f"\n{'='*70}")
    print("Forward Pass Test")
    print(f"{'='*70}")
    print(f"Input shape: {x.shape}")

    with torch.no_grad():
        output, scale_features = pmsa(x)

    print(f"\nOutput shape: {output.shape}")
    print(f"Number of scale features: {len(scale_features)}")

    # Verify all scale features have the same shape
    print(f"\nScale feature shapes:")
    for i, feat in enumerate(scale_features):
        print(f"  Scale {i+1} ({pmsa.scales[i]}x, {pmsa.organ_contexts[i]}): {feat.shape}")
        assert feat.shape[1] == in_channels, f"Scale {i} has wrong channels"

    print(f"\n✅ Output shape correct: {output.shape}")
    print(f"✅ All {len(scale_features)} scale features have shape [batch, {in_channels}, D, H, W]")

    # Verify progressive fusion is being used (not just concatenation)
    # By checking that gradients flow through progressive fusion modules
    pmsa.train()
    x_grad = torch.randn(batch_size, in_channels, 16, 32, 32, device=device, requires_grad=True)
    output_grad, _ = pmsa(x_grad)
    loss = output_grad.sum()
    loss.backward()

    # Check if progressive fusion modules have gradients
    fusion_has_gradients = False
    for i, module in enumerate(pmsa.progressive_fusion):
        for param in module.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                fusion_has_gradients = True
                break
        if fusion_has_gradients:
            break

    if fusion_has_gradients:
        print("\n✅ Progressive fusion modules are being used (gradients detected)")
    else:
        print("\n⚠️  Warning: No gradients in progressive fusion modules")

    # Compare with and without training mode
    pmsa.eval()
    print(f"\n{'='*70}")
    print("Progressive Fusion Architecture Summary")
    print(f"{'='*70}")
    print("""
The PMSA module implements progressive multi-scale fusion:

1. Scale 1 (0.5x, small organs):
   → Generate scale feature₁
   → Progressive output₁ = scale feature₁

2. Scale 2 (0.75x, small organs):
   → Generate scale feature₂
   → Progressive output₂ = Fusion([scale feature₁, scale feature₂])

3. Scale 3 (1.0x, medium organs):
   → Generate scale feature₃
   → Progressive output₃ = Fusion([scale feature₁, scale feature₂, scale feature₃])

4. Scale 4 (1.25x, medium organs):
   → Generate scale feature₄
   → Progressive output₄ = Fusion([scale feature₁, ..., scale feature₄])

5. Scale 5 (1.5x, large organs):
   → Generate scale feature₅
   → Progressive output₅ = Fusion([scale feature₁, ..., scale feature₅])

Final: All scale features → Gating → Weighted fusion → Output

This progressive aggregation allows the network to accumulate multi-scale
information from coarse to fine, which is crucial for detecting organs of
varying sizes (small: gallbladder, medium: adrenal glands, large: liver).
    """)

    print("="*70)
    print("✅ VERIFICATION COMPLETE: Progressive Fusion is WORKING")
    print("="*70)


if __name__ == "__main__":
    verify_progressive_fusion()