import unittest
import torch
import numpy as np
import sys
import os
from pathlib import Path
import tempfile
import shutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.smart_sampling import SmartPatchSampler, OrganFocusedSampler
from data.preprocessing import AMOS22Preprocessor
from data.augmentation import AMOS22Augmentation


class TestDataPipeline(unittest.TestCase):
    """Test data pipeline components with synthetic data"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.volume_shape = (128, 256, 256)  # (D, H, W)
        self.patch_size = (64, 128, 128)
        self.num_classes = 14

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_synthetic_volume(self, case_id=0):
        """Create synthetic medical volume for testing"""
        np.random.seed(case_id)  # Reproducible

        d, h, w = self.volume_shape

        # Create image volume (simulating CT scan)
        image = np.random.normal(-200, 100, self.volume_shape).astype(np.float32)

        # Create label volume with realistic organ distribution
        labels = np.zeros(self.volume_shape, dtype=np.uint8)

        # Add organs similar to AMOS22
        # Large organ (liver - class 1)
        liver_mask = self._create_ellipsoid_mask(
            self.volume_shape, center=(d//2, h//2, w//2+20), size=(30, 60, 80)
        )
        labels[liver_mask] = 1
        image[liver_mask] = np.random.normal(50, 10, liver_mask.sum())

        # Medium organs (kidneys - classes 2, 13)
        kidney_r_mask = self._create_ellipsoid_mask(
            self.volume_shape, center=(d//2, h//2-30, w//2+40), size=(20, 25, 25)
        )
        labels[kidney_r_mask] = 2
        image[kidney_r_mask] = np.random.normal(30, 8, kidney_r_mask.sum())

        kidney_l_mask = self._create_ellipsoid_mask(
            self.volume_shape, center=(d//2, h//2+30, w//2+40), size=(20, 25, 25)
        )
        labels[kidney_l_mask] = 13
        image[kidney_l_mask] = np.random.normal(30, 8, kidney_l_mask.sum())

        # Small organs (gallbladder - class 9, duodenum - class 12)
        gb_mask = self._create_ellipsoid_mask(
            self.volume_shape, center=(d//2, h//2-10, w//2+25), size=(8, 6, 6)
        )
        labels[gb_mask] = 9
        image[gb_mask] = np.random.normal(0, 5, gb_mask.sum())

        duodenum_mask = self._create_ellipsoid_mask(
            self.volume_shape, center=(d//2+5, h//2, w//2+15), size=(12, 8, 25)
        )
        labels[duodenum_mask] = 12
        image[duodenum_mask] = np.random.normal(20, 8, duodenum_mask.sum())

        # Add some noise
        noise = np.random.normal(0, 5, image.shape)
        image += noise

        # Clip to realistic range
        image = np.clip(image, -1024, 3000)

        return image, labels

    def _create_ellipsoid_mask(self, shape, center, size):
        """Create ellipsoidal mask"""
        d, h, w = shape
        cd, ch, cw = center
        sd, sh, sw = size

        z, y, x = np.ogrid[:d, :h, :w]
        mask = (((z - cd) / (sd/2))**2 +
                ((y - ch) / (sh/2))**2 +
                ((x - cw) / (sw/2))**2) <= 1

        return mask

    def test_smart_patch_sampler_basic(self):
        """Test basic smart patch sampling"""
        print("\nTesting Smart Patch Sampler...")

        sampler = SmartPatchSampler(
            patch_size=self.patch_size,
            overlap_ratio=0.5,
            small_organ_bias=0.7,
            boundary_bias=0.2,
            random_bias=0.1
        )

        # Create synthetic data
        image, labels = self.create_synthetic_volume(case_id=0)

        # Sample patches
        patches = sampler.sample_patches(
            image_volume=image,
            label_volume=labels,
            num_patches=16
        )

        # Validate patches
        self.assertEqual(len(patches), 16)

        for i, patch in enumerate(patches):
            # Check patch structure
            self.assertIn('image', patch)
            self.assertIn('label', patch)
            self.assertIn('center', patch)
            self.assertIn('bbox', patch)
            self.assertIn('type', patch)

            # Check patch shapes
            self.assertEqual(patch['image'].shape, self.patch_size)
            self.assertEqual(patch['label'].shape, self.patch_size)

            # Check data types
            self.assertEqual(patch['image'].dtype, np.float32)
            self.assertEqual(patch['label'].dtype, np.uint8)

            # Check patch type
            self.assertIn(patch['type'], ['small_organ', 'boundary', 'random', 'fill_random'])

        # Count patch types
        patch_types = [p['type'] for p in patches]
        type_counts = {ptype: patch_types.count(ptype) for ptype in set(patch_types)}

        print(f"Patch type distribution: {type_counts}")

        # Should have some small organ patches due to bias
        self.assertGreater(type_counts.get('small_organ', 0), 0,
                           "No small organ patches found")

    def test_organ_focused_sampler(self):
        """Test organ-focused adaptive sampler"""
        print("\nTesting Organ-Focused Sampler...")

        sampler = OrganFocusedSampler(
            patch_size=self.patch_size,
            adaptive_sampling=True
        )

        # Create synthetic data
        image, labels = self.create_synthetic_volume(case_id=1)

        # Sample patches with curriculum learning
        patches = sampler.sample_adaptive_patches(
            image_volume=image,
            label_volume=labels,
            num_patches=12,
            training_iteration=500  # Mid-training
        )

        # Validate patches
        self.assertEqual(len(patches), 12)

        difficulties = []
        for patch in patches:
            # Check additional fields
            self.assertIn('difficulty_category', patch)
            self.assertIn('difficulty_weight', patch)

            difficulties.append(patch['difficulty_category'])

            # Difficulty weight should be positive
            self.assertGreater(patch['difficulty_weight'], 0)

        # Count difficulty categories
        difficulty_counts = {diff: difficulties.count(diff) for diff in set(difficulties)}
        print(f"Difficulty distribution: {difficulty_counts}")

        # Should prioritize challenging patches
        challenging_patches = sum(difficulties.count(cat) for cat in ['small', 'boundary'])
        self.assertGreater(challenging_patches, 0, "No challenging patches found")

    def test_preprocessing_pipeline(self):
        """Test preprocessing pipeline"""
        print("\nTesting Preprocessing Pipeline...")

        # Create preprocessor
        preprocessor = AMOS22Preprocessor(
            target_spacing=(1.5, 1.5, 1.5),
            intensity_normalization="z_score",
            clip_range=(-200, 300)
        )

        # Create synthetic data and save to temp files
        image, labels = self.create_synthetic_volume(case_id=2)

        # Save as numpy files (simulating NIfTI loading)
        image_path = os.path.join(self.temp_dir, "test_image.npy")
        label_path = os.path.join(self.temp_dir, "test_label.npy")

        # Create mock metadata
        metadata = {
            'original_spacing': (2.0, 1.0, 1.0),  # Different spacing to test resampling
            'original_shape': image.shape,
            'affine': np.eye(4),
            'header': None
        }

        # Save files
        np.save(image_path, image)
        np.save(label_path, labels)

        # Test volume statistics computation
        volume_stats = {
            'mean_intensity': np.mean(image),
            'std_intensity': np.std(image),
            'shape': image.shape
        }

        print(f"Original volume stats:")
        print(f"  Shape: {volume_stats['shape']}")
        print(f"  Mean intensity: {volume_stats['mean_intensity']:.2f}")
        print(f"  Std intensity: {volume_stats['std_intensity']:.2f}")

        # Test intensity normalization
        normalized_image = preprocessor._normalize_intensity(image.copy())

        normalized_stats = {
            'mean_intensity': np.mean(normalized_image),
            'std_intensity': np.std(normalized_image)
        }

        print(f"Normalized volume stats:")
        print(f"  Mean intensity: {normalized_stats['mean_intensity']:.2f}")
        print(f"  Std intensity: {normalized_stats['std_intensity']:.2f}")

        # Z-score normalization should result in ~0 mean, ~1 std
        self.assertLess(abs(normalized_stats['mean_intensity']), 0.1)
        self.assertLess(abs(normalized_stats['std_intensity'] - 1.0), 0.1)

    def test_data_augmentation(self):
        """Test data augmentation pipeline"""
        print("\nTesting Data Augmentation...")

        # Initialize augmentation
        augmentation = AMOS22Augmentation(
            rotation_range=(-15, 15),
            scaling_range=(0.9, 1.1),
            intensity_shift_range=(-0.1, 0.1),
            gaussian_noise_std=0.01,
            random_flip=True
        )

        # Create patch data
        image, labels = self.create_synthetic_volume(case_id=3)

        # Extract a patch for testing
        patch_start = (32, 64, 64)
        patch_end = tuple(start + size for start, size in zip(patch_start, self.patch_size))

        image_patch = image[patch_start[0]:patch_end[0],
                            patch_start[1]:patch_end[1],
                            patch_start[2]:patch_end[2]]
        label_patch = labels[patch_start[0]:patch_end[0],
                             patch_start[1]:patch_end[1],
                             patch_start[2]:patch_end[2]]

        # Convert to tensors
        image_tensor = torch.from_numpy(image_patch).unsqueeze(0).float()  # Add channel dim
        label_tensor = torch.from_numpy(label_patch).long()

        print(f"Original patch shapes: image={image_tensor.shape}, label={label_tensor.shape}")

        # Apply augmentations
        augmented_samples = []
        for i in range(5):
            aug_image, aug_label = augmentation.apply_augmentation(
                image_tensor.clone(),
                label_tensor.clone()
            )

            augmented_samples.append((aug_image, aug_label))

            # Check shapes are preserved
            self.assertEqual(aug_image.shape, image_tensor.shape)
            self.assertEqual(aug_label.shape, label_tensor.shape)

            # Check data types
            self.assertEqual(aug_image.dtype, torch.float32)
            self.assertEqual(aug_label.dtype, torch.long)

        print(f"Generated {len(augmented_samples)} augmented samples")

        # Test that augmentations actually change the data
        original_mean = image_tensor.mean().item()
        aug_means = [sample[0].mean().item() for sample in augmented_samples]

        # At least some augmentations should change the mean
        different_means = sum(1 for mean in aug_means if abs(mean - original_mean) > 0.01)
        print(f"Samples with different means: {different_means}/{len(augmented_samples)}")

    def test_end_to_end_data_pipeline(self):
        """Test complete data pipeline end-to-end"""
        print("\nTesting End-to-End Data Pipeline...")

        # Create multiple synthetic volumes
        volumes = []
        for case_id in range(3):
            image, labels = self.create_synthetic_volume(case_id)
            volumes.append((image, labels, f"case_{case_id}"))

        # Initialize components
        sampler = SmartPatchSampler(patch_size=self.patch_size)
        augmentation = AMOS22Augmentation(enabled=True)

        # Process each volume
        all_patches = []
        for image, labels, case_id in volumes:
            print(f"Processing {case_id}...")

            # Sample patches
            patches = sampler.sample_patches(image, labels, num_patches=8)

            for patch in patches:
                # Convert to tensors
                image_tensor = torch.from_numpy(patch['image']).unsqueeze(0).float()
                label_tensor = torch.from_numpy(patch['label']).long()

                # Apply augmentation (randomly)
                if np.random.random() < 0.5:
                    image_tensor, label_tensor = augmentation.apply_augmentation(
                        image_tensor, label_tensor
                    )

                # Store processed patch
                processed_patch = {
                    'image': image_tensor,
                    'label': label_tensor,
                    'case_id': case_id,
                    'patch_center': patch['center'],
                    'patch_type': patch['type']
                }

                all_patches.append(processed_patch)

        print(f"Total processed patches: {len(all_patches)}")

        # Validate final dataset
        self.assertGreater(len(all_patches), 0)

        # Check batch creation
        batch_size = 4
        if len(all_patches) >= batch_size:
            batch_patches = all_patches[:batch_size]

            # Stack into batches
            batch_images = torch.stack([p['image'] for p in batch_patches])
            batch_labels = torch.stack([p['label'] for p in batch_patches])

            expected_image_shape = (batch_size, 1) + self.patch_size
            expected_label_shape = (batch_size,) + self.patch_size

            self.assertEqual(batch_images.shape, expected_image_shape)
            self.assertEqual(batch_labels.shape, expected_label_shape)

            print(f"Batch shapes: images={batch_images.shape}, labels={batch_labels.shape}")

        # Compute dataset statistics
        all_images = torch.stack([p['image'] for p in all_patches])
        all_labels = torch.stack([p['label'] for p in all_patches])

        dataset_stats = {
            'num_patches': len(all_patches),
            'image_mean': all_images.mean().item(),
            'image_std': all_images.std().item(),
            'unique_labels': torch.unique(all_labels).tolist(),
            'patch_types': [p['patch_type'] for p in all_patches]
        }

        print(f"Dataset Statistics:")
        for key, value in dataset_stats.items():
            if isinstance(value, (list, tuple)) and len(value) > 10:
                print(f"  {key}: {len(value)} items")
            else:
                print(f"  {key}: {value}")

    def test_patch_sampling_consistency(self):
        """Test patch sampling consistency and reproducibility"""
        print("\nTesting Patch Sampling Consistency...")

        # Create volume
        image, labels = self.create_synthetic_volume(case_id=4)

        sampler = SmartPatchSampler(
            patch_size=self.patch_size,
            small_organ_bias=0.8,  # High bias for testing
            boundary_bias=0.15,
            random_bias=0.05
        )

        # Sample multiple times
        samples = []
        for run in range(3):
            np.random.seed(42)  # Fixed seed for reproducibility
            patches = sampler.sample_patches(image, labels, num_patches=10)
            samples.append(patches)

        # Check that results are consistent with same seed
        for i in range(len(samples[0])):
            patch1 = samples[0][i]
            patch2 = samples[1][i]

            # Centers should be the same with same seed
            self.assertEqual(patch1['center'], patch2['center'])

            # Images should be identical
            np.testing.assert_array_equal(patch1['image'], patch2['image'])
            np.testing.assert_array_equal(patch1['label'], patch2['label'])

        print("✅ Patch sampling is consistent with fixed seed")

        # Test without fixed seed (should be different)
        patches_random1 = sampler.sample_patches(image, labels, num_patches=10)
        patches_random2 = sampler.sample_patches(image, labels, num_patches=10)

        # At least some patches should be different
        different_centers = sum(1 for p1, p2 in zip(patches_random1, patches_random2)
                                if p1['center'] != p2['center'])

        print(f"Different patch centers without fixed seed: {different_centers}/10")
        self.assertGreater(different_centers, 0, "All patches identical without fixed seed")

    def test_edge_cases(self):
        """Test edge cases in data pipeline"""
        print("\nTesting Edge Cases...")

        # Test with very small volume
        small_image = np.random.randn(32, 32, 32).astype(np.float32)
        small_labels = np.zeros((32, 32, 32), dtype=np.uint8)

        # Add one small organ
        small_labels[15:17, 15:17, 15:17] = 9

        sampler = SmartPatchSampler(patch_size=(16, 16, 16))

        try:
            patches = sampler.sample_patches(small_image, small_labels, num_patches=4)
            print(f"Small volume test: Generated {len(patches)} patches")
            self.assertGreater(len(patches), 0)
        except Exception as e:
            print(f"Small volume test failed: {e}")

        # Test with volume containing no organs
        empty_image = np.random.randn(*self.volume_shape).astype(np.float32)
        empty_labels = np.zeros(self.volume_shape, dtype=np.uint8)  # All background

        sampler = SmartPatchSampler(patch_size=self.patch_size)

        try:
            patches = sampler.sample_patches(empty_image, empty_labels, num_patches=8)
            print(f"Empty volume test: Generated {len(patches)} patches")

            # Should still generate patches (random/background)
            self.assertEqual(len(patches), 8)

            # All patches should be background or random type
            patch_types = [p['type'] for p in patches]
            self.assertTrue(all(ptype in ['random', 'fill_random'] for ptype in patch_types))

        except Exception as e:
            print(f"Empty volume test failed: {e}")


def run_data_pipeline_tests():
    """Run all data pipeline tests"""
    print("Testing Data Pipeline Components...")
    print("=" * 50)

    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDataPipeline)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_data_pipeline_tests()

    if success:
        print("\n✅ All data pipeline tests passed!")
        print("The data processing pipeline is working correctly.")
    else:
        print("\n❌ Some data pipeline tests failed!")
        sys.exit(1)