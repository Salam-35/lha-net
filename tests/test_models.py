import unittest
import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.lha_net import LHANet, create_lha_net, LHANetWithAuxiliaryLoss
from models.pmsa_module import PMSAModule, HierarchicalPMSA
from models.decoder import OrganSizeAwareDecoder
from models.backbone import ResNet3DBackbone, LightweightBackbone, resnet18_3d


class TestModelInitialization(unittest.TestCase):
    """Test model initialization and basic forward passes"""

    def setUp(self):
        """Set up test fixtures"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 2
        self.input_shape = (self.batch_size, 1, 64, 128, 128)  # (B, C, D, H, W)
        self.num_classes = 14

    def test_lightweight_backbone_initialization(self):
        """Test lightweight backbone initialization"""
        backbone = LightweightBackbone(
            in_channels=1,
            base_channels=32,
            channel_multipliers=[1, 2, 4, 8, 16]
        )

        # Test forward pass
        x = torch.randn(self.input_shape, device=self.device)
        backbone = backbone.to(self.device)

        features = backbone(x)

        # Should return 5 feature maps
        self.assertEqual(len(features), 5)

        # Check feature dimensions
        expected_channels = [32, 64, 128, 256, 512]
        for i, (feat, expected_ch) in enumerate(zip(features, expected_channels)):
            self.assertEqual(feat.size(1), expected_ch)
            self.assertEqual(feat.size(0), self.batch_size)
            print(f"Feature {i}: {feat.shape}")

    def test_resnet3d_backbone_initialization(self):
        """Test ResNet3D backbone initialization"""
        backbone = resnet18_3d(in_channels=1, base_channels=64)
        backbone = backbone.to(self.device)

        x = torch.randn(self.input_shape, device=self.device)
        features = backbone(x)

        # Should return 5 feature maps
        self.assertEqual(len(features), 5)
        print(f"ResNet3D features shapes: {[f.shape for f in features]}")

    def test_pmsa_module_initialization(self):
        """Test PMSA module initialization and forward pass"""
        in_channels = 128
        pmsa = PMSAModule(
            in_channels=in_channels,
            scales=[0.5, 0.75, 1.0, 1.25, 1.5],
            organ_contexts=["small", "small", "medium", "medium", "large"]
        ).to(self.device)

        # Create input feature map
        x = torch.randn(self.batch_size, in_channels, 32, 64, 64, device=self.device)

        # Forward pass
        output, scale_features = pmsa(x)

        # Check outputs
        self.assertEqual(output.shape, x.shape)
        self.assertEqual(len(scale_features), 5)  # 5 scales

        print(f"PMSA output shape: {output.shape}")
        print(f"Scale features shapes: {[f.shape for f in scale_features]}")

    def test_hierarchical_pmsa_initialization(self):
        """Test Hierarchical PMSA initialization"""
        channels_list = [64, 128, 256, 512]
        h_pmsa = HierarchicalPMSA(
            channels_list=channels_list,
            scales=[0.5, 0.75, 1.0, 1.25, 1.5]
        ).to(self.device)

        # Create input feature maps
        feature_maps = []
        base_size = [32, 64, 64]
        for i, channels in enumerate(channels_list):
            # Reduce spatial size for deeper features
            size = [s // (2 ** i) for s in base_size]
            feat = torch.randn(self.batch_size, channels, *size, device=self.device)
            feature_maps.append(feat)

        # Forward pass
        outputs = h_pmsa(feature_maps)

        # Check outputs
        self.assertEqual(len(outputs), len(feature_maps))
        for i, (out, inp) in enumerate(zip(outputs, feature_maps)):
            self.assertEqual(out.size(1), inp.size(1))  # Same channels
            print(f"Hierarchical PMSA level {i}: {inp.shape} -> {out.shape}")

    def test_organ_size_aware_decoder_initialization(self):
        """Test Organ-Size-Aware Decoder initialization"""
        feature_channels = [64, 128, 256, 512]
        decoder_channels = [32, 64, 128, 256]

        decoder = OrganSizeAwareDecoder(
            feature_channels=feature_channels,
            decoder_channels=decoder_channels,
            num_classes=self.num_classes
        ).to(self.device)

        # Create input features
        features = []
        base_size = [32, 64, 64]
        for i, channels in enumerate(feature_channels):
            size = [s // (2 ** i) for s in base_size]
            feat = torch.randn(self.batch_size, channels, *size, device=self.device)
            features.append(feat)

        # Forward pass
        result = decoder(features, target_size=(64, 128, 128))

        # Check outputs
        self.assertIn('final_output', result)
        self.assertIn('level_outputs', result)
        self.assertIn('size_predictions', result)

        final_output = result['final_output']
        self.assertEqual(final_output.shape, (self.batch_size, self.num_classes, 64, 128, 128))

        print(f"Decoder final output shape: {final_output.shape}")
        print(f"Number of level outputs: {len(result['level_outputs'])}")

    def test_lha_net_lightweight_initialization(self):
        """Test LHA-Net lightweight configuration"""
        model = create_lha_net(
            config_type="lightweight",
            num_classes=self.num_classes
        ).to(self.device)

        x = torch.randn(self.input_shape, device=self.device)

        # Test inference mode
        model.eval()
        with torch.no_grad():
            output = model(x)

        if isinstance(output, dict):
            final_pred = output['final_prediction']
        else:
            final_pred = output

        self.assertEqual(final_pred.shape, (self.batch_size, self.num_classes, 64, 128, 128))
        print(f"LHA-Net lightweight output shape: {final_pred.shape}")

        # Test training mode with deep supervision
        model.train()
        output = model(x, return_features=True)

        self.assertIsInstance(output, dict)
        self.assertIn('final_prediction', output)

        if 'deep_supervision_outputs' in output:
            print(f"Deep supervision outputs: {len(output['deep_supervision_outputs'])}")

    def test_lha_net_standard_initialization(self):
        """Test LHA-Net standard configuration"""
        model = create_lha_net(
            config_type="standard",
            num_classes=self.num_classes
        ).to(self.device)

        x = torch.randn(self.input_shape, device=self.device)

        model.eval()
        with torch.no_grad():
            output = model(x)

        if isinstance(output, dict):
            final_pred = output['final_prediction']
        else:
            final_pred = output

        self.assertEqual(final_pred.shape, (self.batch_size, self.num_classes, 64, 128, 128))
        print(f"LHA-Net standard output shape: {final_pred.shape}")

    def test_lha_net_with_auxiliary_loss(self):
        """Test LHA-Net with auxiliary loss"""
        model = LHANetWithAuxiliaryLoss(
            num_classes=self.num_classes,
            backbone_type="resnet18",
            use_lightweight=True,
            base_channels=32
        ).to(self.device)

        x = torch.randn(self.input_shape, device=self.device)

        # Training mode
        model.train()
        output = model(x)

        self.assertIsInstance(output, dict)
        self.assertIn('final_prediction', output)

        if 'deep_supervision_outputs' in output:
            print(f"Auxiliary loss outputs: {output.keys()}")

    def test_model_memory_usage(self):
        """Test model memory usage estimation"""
        model = create_lha_net(
            config_type="lightweight",
            num_classes=self.num_classes
        )

        # Get model size information
        model_info = model.get_model_size()

        self.assertIn('total_parameters', model_info)
        self.assertIn('backbone_parameters', model_info)
        self.assertIn('pmsa_parameters', model_info)
        self.assertIn('decoder_parameters', model_info)

        print("Model Size Information:")
        for key, value in model_info.items():
            if 'parameters' in key:
                print(f"  {key}: {value:,}")

        # Test memory estimation (only on GPU)
        if self.device.type == 'cuda':
            model = model.to(self.device)
            try:
                memory_stats = model.get_memory_usage(self.input_shape)
                print("Memory Usage Estimation:")
                for key, value in memory_stats.items():
                    print(f"  {key}: {value:.2f}")
            except Exception as e:
                print(f"Memory estimation failed: {e}")

    def test_gradient_flow(self):
        """Test gradient flow through the model"""
        model = create_lha_net(
            config_type="lightweight",
            num_classes=self.num_classes
        ).to(self.device)

        x = torch.randn(self.input_shape, device=self.device, requires_grad=True)
        target = torch.randint(0, self.num_classes, (self.batch_size, 64, 128, 128), device=self.device)

        model.train()
        output = model(x)

        if isinstance(output, dict):
            predictions = output['final_prediction']
        else:
            predictions = output

        # Simple cross-entropy loss
        loss = torch.nn.functional.cross_entropy(predictions, target.long())
        loss.backward()

        # Check that gradients exist
        has_gradients = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        self.assertTrue(has_gradients, "No gradients found in model parameters")

        # Check gradient magnitudes
        total_grad_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.data.norm(2).item() ** 2
        total_grad_norm = total_grad_norm ** 0.5

        print(f"Total gradient norm: {total_grad_norm:.6f}")
        self.assertGreater(total_grad_norm, 0, "Gradient norm is zero")


def run_model_tests():
    """Run all model tests"""
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestModelInitialization)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    print("Testing LHA-Net Model Components...")
    print("=" * 50)

    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA is available. Device: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("CUDA not available. Running on CPU.")

    print("=" * 50)

    success = run_model_tests()

    if success:
        print("\n✅ All model tests passed!")
    else:
        print("\n❌ Some model tests failed!")
        sys.exit(1)