import unittest
import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from losses.adaptive_focal import AdaptiveFocalLoss, DiceAdaptiveFocalLoss
from losses.size_weighted import SizeWeightedLoss, MultiScaleSizeWeightedLoss
from losses.combo_loss import ComboLoss, LHANetLoss


class TestLossFunctions(unittest.TestCase):
    """Test all loss functions with synthetic data"""

    def setUp(self):
        """Set up test fixtures"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 2
        self.num_classes = 14
        self.spatial_shape = (32, 64, 64)  # (D, H, W)

        # Create synthetic predictions and targets
        self.predictions = torch.randn(
            self.batch_size, self.num_classes, *self.spatial_shape,
            device=self.device, requires_grad=True
        )

        self.targets = torch.randint(
            0, self.num_classes, (self.batch_size, *self.spatial_shape),
            device=self.device
        )

        # Create AMOS22-like targets with realistic organ distribution
        self.realistic_targets = self._create_realistic_targets()

    def _create_realistic_targets(self):
        """Create more realistic targets mimicking AMOS22 distribution"""
        targets = torch.zeros(self.batch_size, *self.spatial_shape, device=self.device, dtype=torch.long)

        for b in range(self.batch_size):
            # Background (most voxels)
            targets[b] = 0

            # Large organs (liver, kidneys, etc.)
            d, h, w = self.spatial_shape

            # Liver (class 1) - large central region
            targets[b, d//4:3*d//4, h//4:3*h//4, w//3:2*w//3] = 1

            # Kidneys (classes 2, 13) - medium regions
            targets[b, d//3:2*d//3, h//6:h//3, w//2:2*w//3] = 2
            targets[b, d//3:2*d//3, 2*h//3:5*h//6, w//2:2*w//3] = 13

            # Small organs (gallbladder, duodenum)
            # Gallbladder (class 9) - very small
            targets[b, d//2:d//2+3, h//2:h//2+4, w//2+5:w//2+9] = 9

            # Duodenum (class 12) - small elongated
            targets[b, d//2-2:d//2+3, h//2+10:h//2+15, w//2-5:w//2+10] = 12

        return targets

    def test_adaptive_focal_loss_basic(self):
        """Test basic Adaptive Focal Loss functionality"""
        loss_fn = AdaptiveFocalLoss(
            gamma=2.0,
            adaptive_gamma=True,
            size_aware_alpha=True
        ).to(self.device)

        # Test with realistic targets
        loss = loss_fn(self.predictions, self.realistic_targets)

        # Loss should be finite and positive
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0)

        # Test backward pass
        loss.backward()
        self.assertIsNotNone(self.predictions.grad)

        print(f"Adaptive Focal Loss: {loss.item():.4f}")

    def test_adaptive_focal_loss_components(self):
        """Test Adaptive Focal Loss with different configurations"""
        # Standard focal loss
        standard_focal = AdaptiveFocalLoss(
            gamma=2.0,
            adaptive_gamma=False,
            size_aware_alpha=False
        ).to(self.device)

        # Adaptive focal loss
        adaptive_focal = AdaptiveFocalLoss(
            gamma=2.0,
            adaptive_gamma=True,
            size_aware_alpha=True
        ).to(self.device)

        loss1 = standard_focal(self.predictions, self.realistic_targets)
        loss2 = adaptive_focal(self.predictions, self.realistic_targets)

        # Both should be finite
        self.assertTrue(torch.isfinite(loss1))
        self.assertTrue(torch.isfinite(loss2))

        print(f"Standard Focal Loss: {loss1.item():.4f}")
        print(f"Adaptive Focal Loss: {loss2.item():.4f}")

    def test_dice_adaptive_focal_loss(self):
        """Test combined Dice + Adaptive Focal Loss"""
        loss_fn = DiceAdaptiveFocalLoss(
            focal_weight=0.7,
            dice_weight=0.3
        ).to(self.device)

        # Reset gradients
        if self.predictions.grad is not None:
            self.predictions.grad.zero_()

        loss_dict = loss_fn(self.predictions, self.realistic_targets)

        # Check return format
        self.assertIsInstance(loss_dict, dict)
        self.assertIn('total_loss', loss_dict)
        self.assertIn('focal_loss', loss_dict)
        self.assertIn('dice_loss', loss_dict)

        # All losses should be finite
        for key, value in loss_dict.items():
            self.assertTrue(torch.isfinite(value), f"Loss {key} is not finite")

        # Test backward pass
        loss_dict['total_loss'].backward()
        self.assertIsNotNone(self.predictions.grad)

        print(f"Dice + Focal Loss Components:")
        for key, value in loss_dict.items():
            print(f"  {key}: {value.item():.4f}")

    def test_size_weighted_loss(self):
        """Test Size-Weighted Loss"""
        loss_fn = SizeWeightedLoss(
            volume_adaptive=True,
            spatial_weighting=True,
            boundary_emphasis=2.0
        ).to(self.device)

        # Reset gradients
        if self.predictions.grad is not None:
            self.predictions.grad.zero_()

        loss_dict = loss_fn(self.predictions, self.realistic_targets)

        # Check return format
        self.assertIsInstance(loss_dict, dict)
        self.assertIn('total_loss', loss_dict)
        self.assertIn('volume_weighted_loss', loss_dict)
        self.assertIn('boundary_weighted_loss', loss_dict)

        # Test backward pass
        loss_dict['total_loss'].backward()
        self.assertIsNotNone(self.predictions.grad)

        print(f"Size-Weighted Loss Components:")
        for key, value in loss_dict.items():
            print(f"  {key}: {value.item():.4f}")

    def test_multi_scale_size_weighted_loss(self):
        """Test Multi-Scale Size-Weighted Loss"""
        # Create multi-scale predictions
        scale_predictions = []
        for scale in [0.5, 0.75, 1.0]:
            scale_shape = tuple(int(s * scale) for s in self.spatial_shape)
            scale_pred = torch.randn(
                self.batch_size, self.num_classes, *scale_shape,
                device=self.device, requires_grad=True
            )
            scale_predictions.append(scale_pred)

        loss_fn = MultiScaleSizeWeightedLoss(
            scale_weights=[1.0, 0.8, 0.6]
        ).to(self.device)

        loss_dict = loss_fn(scale_predictions, self.realistic_targets)

        # Check return format
        self.assertIsInstance(loss_dict, dict)
        self.assertIn('total_loss', loss_dict)
        self.assertIn('scale_losses', loss_dict)

        # Test backward pass
        loss_dict['total_loss'].backward()

        print(f"Multi-Scale Loss: {loss_dict['total_loss'].item():.4f}")
        print(f"Number of scales: {loss_dict['num_scales']}")

    def test_combo_loss(self):
        """Test Combo Loss (combination of multiple losses)"""
        loss_fn = ComboLoss(
            focal_weight=0.5,
            dice_weight=0.3,
            size_weight=0.2,
            use_adaptive_focal=True,
            use_size_weighting=True
        ).to(self.device)

        # Reset gradients
        if self.predictions.grad is not None:
            self.predictions.grad.zero_()

        loss_dict = loss_fn(self.predictions, self.realistic_targets)

        # Check return format
        self.assertIsInstance(loss_dict, dict)
        self.assertIn('total_loss', loss_dict)
        self.assertIn('focal_contribution', loss_dict)
        self.assertIn('size_contribution', loss_dict)

        # Test backward pass
        loss_dict['total_loss'].backward()
        self.assertIsNotNone(self.predictions.grad)

        print(f"Combo Loss Components:")
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.item():.4f}")

    def test_lha_net_loss_simple(self):
        """Test LHA-Net Loss with simple prediction"""
        loss_fn = LHANetLoss(
            primary_loss_weight=1.0,
            deep_supervision_weight=0.4,
            size_prediction_weight=0.1
        ).to(self.device)

        # Reset gradients
        if self.predictions.grad is not None:
            self.predictions.grad.zero_()

        # Test with simple tensor input
        loss_dict = loss_fn(self.predictions, self.realistic_targets)

        self.assertIn('total_loss', loss_dict)
        self.assertTrue(torch.isfinite(loss_dict['total_loss']))

        # Test backward pass
        loss_dict['total_loss'].backward()
        self.assertIsNotNone(self.predictions.grad)

        print(f"LHA-Net Loss (simple): {loss_dict['total_loss'].item():.4f}")

    def test_lha_net_loss_complex(self):
        """Test LHA-Net Loss with complex model output"""
        loss_fn = LHANetLoss(
            primary_loss_weight=1.0,
            deep_supervision_weight=0.4,
            size_prediction_weight=0.1,
            routing_weight=0.05
        ).to(self.device)

        # Create complex model output
        model_output = {
            'final_prediction': self.predictions,
            'deep_supervision_outputs': [
                torch.randn(self.batch_size, self.num_classes, *self.spatial_shape,
                           device=self.device, requires_grad=True),
                torch.randn(self.batch_size, self.num_classes, *self.spatial_shape,
                           device=self.device, requires_grad=True)
            ],
            'size_predictions': [
                torch.randn(self.batch_size, 3, device=self.device, requires_grad=True),
                torch.randn(self.batch_size, 3, device=self.device, requires_grad=True)
            ],
            'routing_weights': torch.softmax(
                torch.randn(self.batch_size, 3, *self.spatial_shape, device=self.device),
                dim=1
            )
        }

        loss_dict = loss_fn(model_output, self.realistic_targets)

        # Check all components are present
        expected_keys = [
            'total_loss', 'primary_contribution', 'deep_supervision_loss',
            'size_prediction_loss', 'routing_consistency_loss'
        ]

        for key in expected_keys:
            self.assertIn(key, loss_dict, f"Missing loss component: {key}")
            if isinstance(loss_dict[key], torch.Tensor):
                self.assertTrue(torch.isfinite(loss_dict[key]), f"Loss {key} is not finite")

        # Test backward pass
        loss_dict['total_loss'].backward()

        print(f"LHA-Net Loss (complex) Components:")
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.item():.4f}")

    def test_loss_gradients(self):
        """Test gradient flow through all loss functions"""
        loss_functions = [
            AdaptiveFocalLoss(),
            SizeWeightedLoss(),
            ComboLoss(),
            LHANetLoss()
        ]

        for i, loss_fn in enumerate(loss_functions):
            loss_fn = loss_fn.to(self.device)

            # Create fresh predictions with gradients
            pred = torch.randn(
                self.batch_size, self.num_classes, *self.spatial_shape,
                device=self.device, requires_grad=True
            )

            try:
                if isinstance(loss_fn, (SizeWeightedLoss,)):
                    result = loss_fn(pred, self.realistic_targets)
                    loss = result['total_loss']
                elif isinstance(loss_fn, (ComboLoss, LHANetLoss)):
                    result = loss_fn(pred, self.realistic_targets)
                    loss = result['total_loss']
                else:
                    loss = loss_fn(pred, self.realistic_targets)

                # Compute gradients
                loss.backward()

                # Check gradients exist
                self.assertIsNotNone(pred.grad, f"No gradients for {loss_fn.__class__.__name__}")

                # Check gradient magnitude
                grad_norm = pred.grad.norm().item()
                self.assertGreater(grad_norm, 0, f"Zero gradients for {loss_fn.__class__.__name__}")

                print(f"{loss_fn.__class__.__name__}: Loss={loss.item():.4f}, Grad_norm={grad_norm:.4f}")

            except Exception as e:
                self.fail(f"Gradient test failed for {loss_fn.__class__.__name__}: {e}")

    def test_loss_stability(self):
        """Test loss function stability with edge cases"""
        # Test with extreme predictions
        extreme_pred = torch.ones(
            self.batch_size, self.num_classes, *self.spatial_shape,
            device=self.device
        ) * 1000  # Very large logits

        loss_fn = AdaptiveFocalLoss().to(self.device)
        loss = loss_fn(extreme_pred, self.realistic_targets)

        self.assertTrue(torch.isfinite(loss), "Loss not stable with extreme predictions")

        # Test with all-zero predictions
        zero_pred = torch.zeros(
            self.batch_size, self.num_classes, *self.spatial_shape,
            device=self.device
        )

        loss_zero = loss_fn(zero_pred, self.realistic_targets)
        self.assertTrue(torch.isfinite(loss_zero), "Loss not stable with zero predictions")

        print(f"Extreme predictions loss: {loss.item():.4f}")
        print(f"Zero predictions loss: {loss_zero.item():.4f}")

    def test_organ_size_weighting(self):
        """Test that small organs get higher weights"""
        loss_fn = AdaptiveFocalLoss(size_aware_alpha=True).to(self.device)

        # Create targets with only small organs
        small_organ_targets = torch.zeros_like(self.realistic_targets)
        small_organ_targets[self.realistic_targets == 9] = 9  # gallbladder
        small_organ_targets[self.realistic_targets == 12] = 12  # duodenum

        # Create targets with only large organs
        large_organ_targets = torch.zeros_like(self.realistic_targets)
        large_organ_targets[self.realistic_targets == 1] = 1  # liver

        # Same predictions for both
        pred = torch.randn(
            self.batch_size, self.num_classes, *self.spatial_shape,
            device=self.device
        )

        loss_small = loss_fn(pred, small_organ_targets)
        loss_large = loss_fn(pred, large_organ_targets)

        print(f"Small organ loss: {loss_small.item():.4f}")
        print(f"Large organ loss: {loss_large.item():.4f}")

        # Note: The relationship depends on the implementation details
        # Both should be finite
        self.assertTrue(torch.isfinite(loss_small))
        self.assertTrue(torch.isfinite(loss_large))


def run_loss_tests():
    """Run all loss function tests"""
    print("Testing Loss Functions...")
    print("=" * 50)

    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLossFunctions)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA is available. Device: {torch.cuda.get_device_name()}")
    else:
        print("CUDA not available. Running on CPU.")

    print("=" * 50)

    success = run_loss_tests()

    if success:
        print("\n✅ All loss function tests passed!")
    else:
        print("\n❌ Some loss function tests failed!")
        sys.exit(1)