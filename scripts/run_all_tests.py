#!/usr/bin/env python3
"""
Comprehensive Test Runner for LHA-Net
Runs all tests to validate the complete implementation before real data training
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import traceback

# Add tests and src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "tests"))
sys.path.insert(0, str(project_root / "src"))

# Import test modules
try:
    from test_models import run_model_tests
    from test_metrics import run_metrics_tests
    from test_losses import run_loss_tests
    from test_memory import run_memory_tests
    from test_data_pipeline import run_data_pipeline_tests
except ImportError as e:
    print(f"Warning: Could not import test modules: {e}")
    print("Make sure you're running from the project root directory")


class TestRunner:
    """Comprehensive test runner for LHA-Net"""

    def __init__(self):
        self.test_results = {}
        self.start_time = None
        self.total_duration = 0

    def run_test_suite(self, test_name: str, test_function, skip_on_error: bool = False) -> bool:
        """Run a test suite and record results"""
        print(f"\n{'='*60}")
        print(f"🧪 RUNNING TEST SUITE: {test_name.upper()}")
        print(f"{'='*60}")

        start_time = time.time()

        try:
            success = test_function()
            duration = time.time() - start_time

            self.test_results[test_name] = {
                'success': success,
                'duration': duration,
                'error': None
            }

            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"\n{status} - {test_name} ({duration:.2f}s)")

            return success

        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)

            self.test_results[test_name] = {
                'success': False,
                'duration': duration,
                'error': error_msg
            }

            print(f"\n❌ FAILED - {test_name} ({duration:.2f}s)")
            print(f"Error: {error_msg}")

            if not skip_on_error:
                print("\nFull traceback:")
                traceback.print_exc()

            return False

    def run_system_checks(self) -> Dict[str, bool]:
        """Run system requirement checks"""
        print(f"\n{'='*60}")
        print("🔍 SYSTEM REQUIREMENTS CHECK")
        print(f"{'='*60}")

        checks = {}

        # Python version
        python_version = sys.version_info
        checks['python_version'] = python_version >= (3, 8)
        print(f"Python Version: {python_version.major}.{python_version.minor}.{python_version.micro} "
              f"{'✅' if checks['python_version'] else '❌'}")

        # PyTorch
        try:
            import torch
            torch_version = torch.__version__
            checks['pytorch'] = True
            print(f"PyTorch: {torch_version} ✅")

            # CUDA availability
            cuda_available = torch.cuda.is_available()
            checks['cuda'] = cuda_available
            if cuda_available:
                gpu_name = torch.cuda.get_device_name()
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"CUDA: Available ✅")
                print(f"GPU: {gpu_name}")
                print(f"GPU Memory: {gpu_memory:.1f}GB")
            else:
                print(f"CUDA: Not Available ⚠️")

        except ImportError:
            checks['pytorch'] = False
            checks['cuda'] = False
            print("PyTorch: Not installed ❌")

        # Required packages
        required_packages = [
            'numpy', 'scipy', 'scikit-learn', 'matplotlib',
            'nibabel', 'SimpleITK', 'pandas', 'tqdm'
        ]

        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
                print(f"{package}: ✅")
            except ImportError:
                missing_packages.append(package)
                print(f"{package}: ❌")

        checks['all_packages'] = len(missing_packages) == 0

        if missing_packages:
            print(f"\nMissing packages: {missing_packages}")
            print("Install with: pip install " + " ".join(missing_packages))

        # Project structure
        required_dirs = ['src', 'tests', 'configs']
        project_structure_ok = True

        for dir_name in required_dirs:
            dir_path = project_root / dir_name
            if dir_path.exists():
                print(f"Directory {dir_name}/: ✅")
            else:
                print(f"Directory {dir_name}/: ❌")
                project_structure_ok = False

        checks['project_structure'] = project_structure_ok

        return checks

    def run_quick_smoke_tests(self) -> bool:
        """Run quick smoke tests to check basic functionality"""
        print(f"\n{'='*60}")
        print("💨 QUICK SMOKE TESTS")
        print(f"{'='*60}")

        try:
            # Test basic imports
            print("Testing basic imports...")
            from models.lha_net import create_lha_net
            from losses.combo_loss import LHANetLoss
            from evaluation.metrics import SegmentationMetrics
            print("✅ All imports successful")

            # Test model creation
            print("Testing model creation...")
            model = create_lha_net(config_type="lightweight", num_classes=14)
            print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

            # Test basic forward pass
            print("Testing basic forward pass...")
            import torch
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)

            x = torch.randn(1, 1, 32, 64, 64, device=device)
            with torch.no_grad():
                output = model(x)

            if isinstance(output, dict):
                output_shape = output['final_prediction'].shape
            else:
                output_shape = output.shape

            print(f"✅ Forward pass successful, output shape: {output_shape}")

            # Test loss computation
            print("Testing loss computation...")
            loss_fn = LHANetLoss()
            target = torch.randint(0, 14, (1, 32, 64, 64), device=device)

            model.train()
            output = model(x, return_features=True)
            loss_dict = loss_fn(output, target)

            print(f"✅ Loss computation successful: {loss_dict['total_loss'].item():.4f}")

            return True

        except Exception as e:
            print(f"❌ Smoke test failed: {e}")
            return False

    def run_all_tests(self, skip_slow_tests: bool = False, skip_memory_tests: bool = False) -> bool:
        """Run all test suites"""
        self.start_time = time.time()

        print("🚀 LHA-Net Comprehensive Test Suite")
        print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # System checks
        system_checks = self.run_system_checks()
        if not all(system_checks.values()):
            print("\n⚠️ Some system requirements are not met!")
            print("Continuing with tests, but some may fail...")

        # Quick smoke tests
        if not self.run_quick_smoke_tests():
            print("\n🚫 Smoke tests failed! Stopping test execution.")
            return False

        # Define test suites
        test_suites = [
            ("model_architecture", run_model_tests, False),
            ("evaluation_metrics", run_metrics_tests, False),
            ("loss_functions", run_loss_tests, False),
            ("data_pipeline", run_data_pipeline_tests, skip_slow_tests),
        ]

        # Add memory tests if not skipped and CUDA is available
        if not skip_memory_tests and system_checks.get('cuda', False):
            test_suites.append(("memory_optimization", run_memory_tests, False))
        elif skip_memory_tests:
            print("\n⏭️ Skipping memory tests (--skip-memory-tests)")
        else:
            print("\n⏭️ Skipping memory tests (CUDA not available)")

        # Run all test suites
        all_passed = True
        for test_name, test_func, should_skip in test_suites:
            if should_skip:
                print(f"\n⏭️ Skipping {test_name} (--skip-slow-tests)")
                continue

            success = self.run_test_suite(test_name, test_func, skip_on_error=True)
            if not success:
                all_passed = False

        self.total_duration = time.time() - self.start_time

        # Print final summary
        self._print_summary()

        return all_passed

    def _print_summary(self):
        """Print test execution summary"""
        print(f"\n{'='*60}")
        print("📊 TEST EXECUTION SUMMARY")
        print(f"{'='*60}")

        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result['success'])
        failed_tests = total_tests - passed_tests

        print(f"Total Test Suites: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Total Duration: {self.total_duration:.2f}s")

        print(f"\nDetailed Results:")
        for test_name, result in self.test_results.items():
            status = "✅ PASSED" if result['success'] else "❌ FAILED"
            duration = result['duration']
            print(f"  {test_name:20} {status:10} ({duration:.2f}s)")

            if not result['success'] and result['error']:
                print(f"    Error: {result['error']}")

        if failed_tests == 0:
            print(f"\n🎉 ALL TESTS PASSED!")
            print("Your LHA-Net implementation is ready for real data training!")
        else:
            print(f"\n⚠️ {failed_tests} TEST SUITE(S) FAILED")
            print("Please fix the issues before proceeding to real data training.")

        print(f"\nNext Steps:")
        if failed_tests == 0:
            print("1. Download and preprocess AMOS22 dataset")
            print("2. Run dummy training: python scripts/dummy_training.py")
            print("3. Start real training with your data")
        else:
            print("1. Fix failing tests")
            print("2. Re-run tests: python scripts/run_all_tests.py")
            print("3. Proceed once all tests pass")


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive LHA-Net tests")
    parser.add_argument('--skip-slow-tests', action='store_true',
                        help='Skip slow tests (data pipeline)')
    parser.add_argument('--skip-memory-tests', action='store_true',
                        help='Skip memory tests (useful for CPU-only systems)')
    parser.add_argument('--smoke-only', action='store_true',
                        help='Run only quick smoke tests')
    parser.add_argument('--test-suite', type=str, choices=['models', 'metrics', 'losses', 'memory', 'data'],
                        help='Run specific test suite only')

    args = parser.parse_args()

    runner = TestRunner()

    if args.smoke_only:
        print("🚀 Running Quick Smoke Tests Only")
        system_checks = runner.run_system_checks()
        smoke_success = runner.run_quick_smoke_tests()

        if smoke_success:
            print("\n✅ Smoke tests passed! Basic functionality is working.")
            return 0
        else:
            print("\n❌ Smoke tests failed! Check your installation.")
            return 1

    elif args.test_suite:
        # Run specific test suite
        test_mapping = {
            'models': run_model_tests,
            'metrics': run_metrics_tests,
            'losses': run_loss_tests,
            'memory': run_memory_tests,
            'data': run_data_pipeline_tests
        }

        if args.test_suite in test_mapping:
            success = runner.run_test_suite(args.test_suite, test_mapping[args.test_suite])
            return 0 if success else 1
        else:
            print(f"Unknown test suite: {args.test_suite}")
            return 1

    else:
        # Run all tests
        success = runner.run_all_tests(
            skip_slow_tests=args.skip_slow_tests,
            skip_memory_tests=args.skip_memory_tests
        )

        return 0 if success else 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)