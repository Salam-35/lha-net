#!/usr/bin/env python3
"""
Verify all fixes are in place before training
"""

import yaml
import sys
from pathlib import Path


def check_config_file(config_path):
    """Check if all required keys exist in config"""
    print(f"\n{'='*70}")
    print(f"Checking: {config_path}")
    print('='*70)

    if not Path(config_path).exists():
        print(f"✗ Config file not found!")
        return False

    with open(config_path) as f:
        config = yaml.safe_load(f)

    required_keys = {
        'training.save_freq': config.get('training', {}).get('save_freq'),
        'training.validation_freq': config.get('training', {}).get('validation_freq'),
        'training.detailed_metrics_freq': config.get('training', {}).get('detailed_metrics_freq'),
        'training.memory_optimization.empty_cache_freq':
            config.get('training', {}).get('memory_optimization', {}).get('empty_cache_freq'),
        'system.memory_log_interval': config.get('system', {}).get('memory_log_interval'),
        'training.optimizer.eps': config.get('training', {}).get('optimizer', {}).get('eps'),
        'training.scheduler.min_lr': config.get('training', {}).get('scheduler', {}).get('min_lr'),
        'training.learning_rate': config.get('training', {}).get('learning_rate'),
        'training.max_grad_norm': config.get('training', {}).get('max_grad_norm'),
    }

    all_present = True
    for key, value in required_keys.items():
        if value is None:
            print(f"  ✗ MISSING: {key}")
            all_present = False
        else:
            print(f"  ✓ {key}: {value}")

    if all_present:
        print(f"\n✓✓✓ All required keys present!")
        return True
    else:
        print(f"\n✗✗✗ Missing keys found!")
        return False


def check_train_py():
    """Check if train.py has the fixes"""
    print(f"\n{'='*70}")
    print(f"Checking: train.py")
    print('='*70)

    train_path = Path(__file__).parent / 'train.py'

    if not train_path.exists():
        print(f"✗ train.py not found!")
        return False

    with open(train_path) as f:
        content = f.read()

    checks = {
        'float() conversion for eps': "eps=float(opt_config.get('eps'",
        'float() conversion for min_lr': "min_lr=float(sched_config.get('min_lr'",
        'float() conversion for learning_rate': "learning_rate=float(self.config['training']['learning_rate']",
        'mixed_precision .get()': "mp_config = self.config['training'].get('mixed_precision'",
        'save_freq .get()': "save_freq = self.config['training'].get('save_freq'",
        'validation_freq .get()': "validation_freq = self.config['training'].get('validation_freq'",
        'memory_log_interval .get()': "memory_log_interval = self.config['system'].get('memory_log_interval'",
        'empty_cache_freq .get()': "empty_cache_freq = self.config['training'].get('memory_optimization'",
    }

    all_present = True
    for check_name, check_string in checks.items():
        if check_string in content:
            print(f"  ✓ {check_name}")
        else:
            print(f"  ✗ MISSING: {check_name}")
            all_present = False

    if all_present:
        print(f"\n✓✓✓ All fixes present in train.py!")
        return True
    else:
        print(f"\n✗✗✗ Some fixes missing in train.py!")
        return False


def check_import_fix():
    """Check if enhanced_backbone.py has the import fix"""
    print(f"\n{'='*70}")
    print(f"Checking: src/models/enhanced_backbone.py")
    print('='*70)

    backbone_path = Path(__file__).parent / 'src' / 'models' / 'enhanced_backbone.py'

    if not backbone_path.exists():
        print(f"✗ enhanced_backbone.py not found!")
        return False

    with open(backbone_path) as f:
        content = f.read()

    if 'from .large_kernel_conv import' in content:
        print(f"  ✓ Import statement is correct (relative import)")
        return True
    elif 'from large_kernel_conv import' in content:
        print(f"  ✗ Import statement is WRONG (should be relative import)")
        print(f"  Change: 'from large_kernel_conv import'")
        print(f"  To:     'from .large_kernel_conv import'")
        return False
    else:
        print(f"  ? Could not find import statement")
        return False


def main():
    print("\n" + "="*70)
    print("LHA-NET TRAINING FIXES VERIFICATION")
    print("="*70)

    results = []

    # Check config files
    results.append(check_config_file('configs/lha_net_adaptive_24gb.yaml'))
    results.append(check_config_file('configs/lha_net_adaptive_config.yaml'))

    # Check Python files
    results.append(check_train_py())
    results.append(check_import_fix())

    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)

    if all(results):
        print("\n✓✓✓ ALL CHECKS PASSED!")
        print("\nYou can now run training:")
        print("  python train.py --config configs/lha_net_adaptive_24gb.yaml")
        return 0
    else:
        print("\n✗✗✗ SOME CHECKS FAILED!")
        print("\nPlease fix the issues above before training.")
        print("\nIf you're on Windows and files are on Linux:")
        print("  1. Copy the updated files from Linux to Windows")
        print("  2. Run this script again on Windows to verify")
        return 1


if __name__ == "__main__":
    sys.exit(main())
