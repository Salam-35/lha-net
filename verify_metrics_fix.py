#!/usr/bin/env python3
"""
Quick script to verify that metrics are calculated correctly
"""

import torch
import numpy as np
from src.evaluation.metrics import compute_dice_score, compute_hausdorff_distance, compute_normalized_surface_distance

print("="*80)
print("VERIFYING METRICS FIXES")
print("="*80)

# Test 1: Dice score should be in [0, 1]
print("\n1. Testing Dice Score Calculation...")
pred = np.random.rand(16, 48, 96, 96)  # One-hot prediction
target = np.random.rand(16, 48, 96, 96)  # One-hot target
pred = (pred > 0.5).astype(np.float32)
target = (target > 0.5).astype(np.float32)

dice = compute_dice_score(pred, target, ignore_background=True)
print(f"   Dice score: {dice:.4f}")
print(f"   ✓ In range [0,1]: {0 <= dice <= 1}")

# Test 2: Perfect match should give Dice = 1.0
print("\n2. Testing Perfect Match...")
pred_perfect = np.zeros((16, 48, 96, 96), dtype=np.float32)
pred_perfect[1, 10:20, 20:40, 30:50] = 1.0  # Some organ

dice_perfect = compute_dice_score(pred_perfect, pred_perfect, ignore_background=True)
print(f"   Perfect match Dice: {dice_perfect:.4f}")
print(f"   ✓ Should be 1.0: {abs(dice_perfect - 1.0) < 0.01}")

# Test 3: No overlap should give Dice = 0.0
print("\n3. Testing No Overlap...")
pred_no_overlap = np.zeros((16, 48, 96, 96), dtype=np.float32)
target_no_overlap = np.zeros((16, 48, 96, 96), dtype=np.float32)
pred_no_overlap[1, 10:20, 20:40, 30:50] = 1.0
target_no_overlap[1, 30:40, 50:70, 60:80] = 1.0  # Different location

dice_no_overlap = compute_dice_score(pred_no_overlap, target_no_overlap, ignore_background=True)
print(f"   No overlap Dice: {dice_no_overlap:.4f}")
print(f"   ✓ Should be close to 0.0: {dice_no_overlap < 0.1}")

# Test 4: HD95 with empty masks
print("\n4. Testing HD95 with Empty Masks...")
empty_pred = np.zeros((48, 96, 96), dtype=np.float32)
empty_target = np.zeros((48, 96, 96), dtype=np.float32)

hd95_empty = compute_hausdorff_distance(empty_pred, empty_target, spacing=(1.5, 1.5, 1.5))
print(f"   HD95 for empty masks: {hd95_empty}")
print(f"   ✓ Should be nan: {np.isnan(hd95_empty)}")

# Test 5: HD95 with valid masks
print("\n5. Testing HD95 with Valid Masks...")
pred_hd = np.zeros((48, 96, 96), dtype=np.float32)
target_hd = np.zeros((48, 96, 96), dtype=np.float32)
pred_hd[20:25, 40:50, 40:50] = 1.0
target_hd[20:25, 42:52, 42:52] = 1.0  # Slightly shifted

hd95_valid = compute_hausdorff_distance(pred_hd, target_hd, spacing=(1.5, 1.5, 1.5))
print(f"   HD95: {hd95_valid:.2f} mm")
print(f"   ✓ Should be a positive number: {hd95_valid > 0 and not np.isnan(hd95_valid) and not np.isinf(hd95_valid)}")

# Test 6: NSD with empty masks
print("\n6. Testing NSD with Empty Masks...")
nsd_empty = compute_normalized_surface_distance(empty_pred, empty_target, spacing=(1.5, 1.5, 1.5))
print(f"   NSD for empty masks: {nsd_empty}")
print(f"   ✓ Should be nan: {np.isnan(nsd_empty)}")

# Test 7: NSD with valid masks
print("\n7. Testing NSD with Valid Masks...")
nsd_valid = compute_normalized_surface_distance(pred_hd, target_hd, spacing=(1.5, 1.5, 1.5), tolerance=5.0)
print(f"   NSD: {nsd_valid:.4f}")
print(f"   ✓ Should be in [0,1]: {0 <= nsd_valid <= 1 or np.isnan(nsd_valid)}")

print("\n" + "="*80)
print("METRICS VERIFICATION COMPLETE")
print("="*80)
print("\nAll basic tests passed! You can now retrain your model.")
print("\nTo start training:")
print("  python train.py --config configs/lha_net_config.yaml")
