#!/usr/bin/env python3
"""
Test if data loading works correctly
Checks if AMOSDataset returns valid patches
"""

import sys
import os
import yaml
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.preprocessing import AMOSDataset

def test_data_loading(config_path='configs/lha_net_config.yaml'):
    """Test data loading pipeline"""

    print("="*80)
    print("TESTING DATA LOADING PIPELINE")
    print("="*80)

    # Load config
    print("\n1. Loading configuration...")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_root = config['paths']['data_root']
    print(f"   Data root: {data_root}")
    print(f"   Patch size: {config['data']['patch_size']}")
    print(f"   Num classes: {config['model']['num_classes']}")

    # Create dataset
    print("\n2. Creating dataset...")
    try:
        dataset = AMOSDataset(
            data_root=data_root,
            split='train',
            patch_size=config['data']['patch_size'],
            num_classes=config['model']['num_classes'],
            augmentation=False
        )
        print(f"   ✓ Dataset created successfully")
        print(f"   Dataset size: {len(dataset)} samples")
    except Exception as e:
        print(f"   ❌ Error creating dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

    if len(dataset) == 0:
        print("   ❌ Dataset is empty!")
        return False

    # Test loading first sample
    print("\n3. Loading first sample...")
    try:
        sample = dataset[0]
        print(f"   ✓ Sample loaded successfully")
        print(f"   Keys in sample: {list(sample.keys())}")
    except Exception as e:
        print(f"   ❌ Error loading sample: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Check image
    print("\n4. Checking image data...")
    image = sample['image']
    print(f"   Shape: {image.shape}")
    print(f"   Type: {image.dtype}")
    print(f"   Range: [{image.min():.3f}, {image.max():.3f}]")
    print(f"   Mean: {image.mean():.3f}")
    print(f"   Std: {image.std():.3f}")

    expected_shape = [1] + config['data']['patch_size']
    if list(image.shape) != expected_shape:
        print(f"   ❌ Wrong shape! Expected {expected_shape}")
        return False

    if image.min() == image.max():
        print(f"   ❌ Image is constant (all same value)!")
        return False

    # For normalized data, should be roughly zero mean, unit std
    if abs(image.mean()) > 3.0:
        print(f"   ⚠️  Mean is far from 0 - normalization might be wrong")

    if image.std() < 0.1 or image.std() > 10.0:
        print(f"   ⚠️  Std is unusual - normalization might be wrong")

    print(f"   ✓ Image data looks OK")

    # Check label
    print("\n5. Checking label data...")
    label = sample['label']
    print(f"   Shape: {label.shape}")
    print(f"   Type: {label.dtype}")

    unique_labels = torch.unique(label)
    print(f"   Unique values: {unique_labels.tolist()}")
    print(f"   Number of classes present: {len(unique_labels)}")

    if list(label.shape) != config['data']['patch_size']:
        print(f"   ❌ Wrong shape! Expected {config['data']['patch_size']}")
        return False

    if len(unique_labels) == 1 and unique_labels[0] == 0:
        print(f"   ❌ Only background (class 0) present!")
        return False

    # Check background ratio
    bg_ratio = (label == 0).float().mean().item()
    print(f"   Background ratio: {bg_ratio:.1%}")

    if bg_ratio > 0.98:
        print(f"   ⚠️  Very high background ratio - patch might be mostly empty")

    # List organs present
    organs_present = [int(x) for x in unique_labels if x > 0]
    print(f"   Organs present: {organs_present}")
    print(f"   ✓ Label data looks OK")

    # Test multiple samples
    print("\n6. Testing multiple samples...")
    num_test = min(5, len(dataset))

    issues = []
    for i in range(num_test):
        try:
            sample = dataset[i]
            img = sample['image']
            lbl = sample['label']

            # Check for issues
            if img.min() == img.max():
                issues.append(f"Sample {i}: Image is constant")

            if len(torch.unique(lbl)) == 1:
                issues.append(f"Sample {i}: Only background in label")

            if not torch.isfinite(img).all():
                issues.append(f"Sample {i}: Image has NaN or Inf")

        except Exception as e:
            issues.append(f"Sample {i}: Error loading - {e}")

    if issues:
        print(f"   ⚠️  Found issues in {len(issues)}/{num_test} samples:")
        for issue in issues:
            print(f"      - {issue}")
    else:
        print(f"   ✓ All {num_test} samples loaded successfully")

    # Final verdict
    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)

    if len(issues) > num_test // 2:
        print("\n❌ DATA LOADING HAS CRITICAL ISSUES!")
        print("   More than half of samples have problems.")
        print("   Check your data preprocessing and paths.")
        return False
    elif len(issues) > 0:
        print("\n⚠️  DATA LOADING HAS SOME ISSUES")
        print("   Some samples have problems but most are OK.")
        print("   Review the issues above.")
        return True
    else:
        print("\n✅ DATA LOADING WORKS CORRECTLY!")
        print("   All checks passed. Data pipeline is working.")
        return True

if __name__ == '__main__':
    try:
        success = test_data_loading()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
