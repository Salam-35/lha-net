import unittest
import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from evaluation.metrics import (
    SegmentationMetrics,
    compute_dice_score,
    compute_hausdorff_distance,
    compute_normalized_surface_distance
)


class TestSegmentationMetrics(unittest.TestCase):
    """Test segmentation metrics with synthetic data"""

    def setUp(self):
        """Set up test fixtures"""
        self.num_classes = 14
        self.spacing = (1.5, 1.5, 1.5)
        self.metrics_calculator = SegmentationMetrics(
            num_classes=self.num_classes,
            spacing=self.spacing
        )

    def create_synthetic_segmentation(self, shape=(64, 128, 128), num_objects=5):
        """Create synthetic segmentation with known properties"""
        segmentation = np.zeros(shape, dtype=np.uint8)

        # Create different sized objects
        d, h, w = shape

        # Large object (liver-like)
        segmentation[10:40, 20:80, 30:90] = 1

        # Medium objects (kidney-like)
        segmentation[15:35, 90:110, 40:70] = 2
        segmentation[15:35, 10:30, 40:70] = 13

        # Small objects (gallbladder, duodenum-like)
        segmentation[20:30, 85:95, 75:85] = 9  # gallbladder
        segmentation[25:35, 60:70, 35:45] = 12  # duodenum

        return segmentation

    def create_noisy_prediction(self, ground_truth, noise_level=0.1):
        """Create noisy prediction from ground truth"""
        prediction = ground_truth.copy()

        # Add some noise
        mask = np.random.random(ground_truth.shape) < noise_level

        # Randomly change some labels
        noise_labels = np.random.randint(0, self.num_classes, size=mask.shape)
        prediction[mask] = noise_labels[mask]

        return prediction

    def test_dice_score_perfect_match(self):
        """Test Dice score with perfect match"""
        # Create binary masks
        pred = np.zeros((32, 64, 64))
        target = np.zeros((32, 64, 64))

        # Add same object to both
        pred[10:20, 20:40, 25:45] = 1
        target[10:20, 20:40, 25:45] = 1

        dice = compute_dice_score(pred, target)

        # Should be perfect match
        self.assertAlmostEqual(dice, 1.0, places=5)
        print(f"Perfect match Dice score: {dice:.6f}")

    def test_dice_score_no_overlap(self):
        """Test Dice score with no overlap"""
        pred = np.zeros((32, 64, 64))
        target = np.zeros((32, 64, 64))

        # Add different objects
        pred[10:20, 20:40, 25:45] = 1
        target[20:30, 40:60, 45:65] = 1

        dice = compute_dice_score(pred, target)

        # Should be zero overlap
        self.assertAlmostEqual(dice, 0.0, places=5)
        print(f"No overlap Dice score: {dice:.6f}")

    def test_dice_score_partial_overlap(self):
        """Test Dice score with partial overlap"""
        pred = np.zeros((32, 64, 64))
        target = np.zeros((32, 64, 64))

        # Add overlapping objects
        pred[10:25, 20:40, 25:45] = 1
        target[15:30, 20:40, 25:45] = 1

        dice = compute_dice_score(pred, target)

        # Should be between 0 and 1
        self.assertGreater(dice, 0.0)
        self.assertLess(dice, 1.0)
        print(f"Partial overlap Dice score: {dice:.6f}")

    def test_multiclass_dice(self):
        """Test multi-class Dice computation"""
        shape = (32, 64, 64)
        num_classes = 5

        # Create one-hot encoded tensors
        pred_onehot = np.zeros((num_classes,) + shape)
        target_onehot = np.zeros((num_classes,) + shape)

        # Class 1
        pred_onehot[1, 10:20, 20:40, 25:45] = 1
        target_onehot[1, 10:20, 20:40, 25:45] = 1

        # Class 2 (partial overlap)
        pred_onehot[2, 15:25, 30:50, 35:55] = 1
        target_onehot[2, 20:30, 30:50, 35:55] = 1

        dice_scores = compute_dice_score(pred_onehot, target_onehot, ignore_background=True)

        # Should have one perfect match and one partial
        self.assertEqual(len(dice_scores), 4)  # Excluding background
        self.assertAlmostEqual(dice_scores[0], 1.0, places=5)  # Class 1
        self.assertGreater(dice_scores[1], 0.0)  # Class 2

        print(f"Multi-class Dice scores: {dice_scores}")

    def test_hausdorff_distance(self):
        """Test Hausdorff distance computation"""
        # Create binary masks
        pred = np.zeros((32, 64, 64))
        target = np.zeros((32, 64, 64))

        # Perfect match
        pred[10:20, 20:40, 25:45] = 1
        target[10:20, 20:40, 25:45] = 1

        hd = compute_hausdorff_distance(pred, target, self.spacing)

        # Should be very small for perfect match
        self.assertLess(hd, 5.0)  # Less than 5mm
        print(f"Perfect match Hausdorff distance: {hd:.2f} mm")

        # Test with some difference
        target[10:20, 20:40, 25:50] = 1  # Slightly larger

        hd2 = compute_hausdorff_distance(pred, target, self.spacing)
        self.assertGreater(hd2, hd)
        print(f"Different shapes Hausdorff distance: {hd2:.2f} mm")

    def test_normalized_surface_distance(self):
        """Test Normalized Surface Distance"""
        # Create binary masks
        pred = np.zeros((32, 64, 64))
        target = np.zeros((32, 64, 64))

        # Perfect match
        pred[10:20, 20:40, 25:45] = 1
        target[10:20, 20:40, 25:45] = 1

        nsd = compute_normalized_surface_distance(pred, target, self.spacing, tolerance=1.0)

        # Should be perfect match
        self.assertAlmostEqual(nsd, 1.0, places=2)
        print(f"Perfect match NSD: {nsd:.4f}")

    def test_segmentation_metrics_comprehensive(self):
        """Test comprehensive segmentation metrics"""
        # Create synthetic data
        shape = (64, 128, 128)
        ground_truth = self.create_synthetic_segmentation(shape)

        # Create noisy prediction
        prediction = self.create_noisy_prediction(ground_truth, noise_level=0.05)

        # Compute all metrics
        metrics = self.metrics_calculator.compute_all_metrics(
            prediction, ground_truth, case_id="synthetic_test"
        )

        # Check that all expected metrics are present
        expected_keys = [
            'dice_scores', 'mean_dice', 'small_organ_dice', 'medium_organ_dice',
            'large_organ_dice', 'hd95_scores', 'mean_hd95', 'nsd_scores',
            'mean_nsd', 'volume_metrics', 'case_id'
        ]

        for key in expected_keys:
            self.assertIn(key, metrics, f"Missing metric: {key}")

        # Check dice scores
        self.assertIsInstance(metrics['dice_scores'], dict)
        self.assertEqual(len(metrics['dice_scores']), self.num_classes)

        # Check organ category dice scores
        self.assertGreaterEqual(metrics['small_organ_dice'], 0.0)
        self.assertLessEqual(metrics['small_organ_dice'], 1.0)

        print("Comprehensive Metrics Test Results:")
        print(f"  Mean Dice: {metrics['mean_dice']:.4f}")
        print(f"  Small Organ Dice: {metrics['small_organ_dice']:.4f}")
        print(f"  Medium Organ Dice: {metrics['medium_organ_dice']:.4f}")
        print(f"  Large Organ Dice: {metrics['large_organ_dice']:.4f}")
        print(f"  Mean NSD: {metrics['mean_nsd']:.4f}")

    def test_metrics_with_tensor_input(self):
        """Test metrics with PyTorch tensor input"""
        shape = (2, 14, 32, 64, 64)  # Batch of one-hot predictions

        # Create random one-hot predictions
        predictions = torch.zeros(shape)
        targets = torch.zeros(shape)

        # Set some random values
        for b in range(shape[0]):
            for c in range(1, 4):  # A few classes
                # Random small regions
                d_start, h_start, w_start = np.random.randint(0, [16, 32, 32])
                d_end = min(d_start + 10, shape[2])
                h_end = min(h_start + 15, shape[3])
                w_end = min(w_start + 15, shape[4])

                predictions[b, c, d_start:d_end, h_start:h_end, w_start:w_end] = 1
                targets[b, c, d_start:d_end, h_start:h_end, w_start:w_end] = 1

        # Convert targets to class indices for one batch
        target_indices = torch.argmax(targets[0], dim=0)

        # Test with tensor input
        metrics = self.metrics_calculator.compute_all_metrics(
            predictions[0], target_indices, case_id="tensor_test"
        )

        self.assertIn('dice_scores', metrics)
        print(f"Tensor input test - Mean Dice: {metrics['mean_dice']:.4f}")

    def test_metrics_aggregation(self):
        """Test metrics aggregation across multiple cases"""
        metrics_list = []

        # Generate multiple synthetic cases
        for i in range(5):
            shape = (32, 64, 64)
            ground_truth = self.create_synthetic_segmentation(shape)
            prediction = self.create_noisy_prediction(ground_truth, noise_level=0.1)

            metrics = self.metrics_calculator.compute_all_metrics(
                prediction, ground_truth, case_id=f"case_{i}"
            )
            metrics_list.append(metrics)

        # Aggregate metrics
        aggregated = self.metrics_calculator.aggregate_metrics(metrics_list)

        # Check aggregated metrics
        self.assertIn('num_cases', aggregated)
        self.assertEqual(aggregated['num_cases'], 5)

        self.assertIn('dice_scores', aggregated)
        self.assertIn('overall_dice', aggregated)

        # Check statistics
        for organ in self.metrics_calculator.organ_names:
            if organ in aggregated['dice_scores']:
                stats = aggregated['dice_scores'][organ]
                self.assertIn('mean', stats)
                self.assertIn('std', stats)
                self.assertIn('median', stats)

        print("Aggregation Test Results:")
        print(f"  Number of cases: {aggregated['num_cases']}")
        print(f"  Overall Dice mean: {aggregated['overall_dice']['mean']:.4f}")
        print(f"  Overall Dice std: {aggregated['overall_dice']['std']:.4f}")

    def test_edge_cases(self):
        """Test edge cases for metrics"""
        # Empty predictions and targets
        pred_empty = np.zeros((32, 64, 64))
        target_empty = np.zeros((32, 64, 64))

        dice_empty = compute_dice_score(pred_empty, target_empty)
        print(f"Empty masks Dice score: {dice_empty:.6f}")

        # Only prediction has content
        pred_only = np.zeros((32, 64, 64))
        pred_only[10:20, 20:30, 25:35] = 1

        dice_pred_only = compute_dice_score(pred_only, target_empty)
        self.assertAlmostEqual(dice_pred_only, 0.0, places=5)
        print(f"Prediction only Dice score: {dice_pred_only:.6f}")

        # Only target has content
        target_only = np.zeros((32, 64, 64))
        target_only[10:20, 20:30, 25:35] = 1

        dice_target_only = compute_dice_score(pred_empty, target_only)
        self.assertAlmostEqual(dice_target_only, 0.0, places=5)
        print(f"Target only Dice score: {dice_target_only:.6f}")


def create_validation_dataset():
    """Create a small validation dataset for testing"""
    print("Creating validation dataset...")

    # Create multiple synthetic cases with known properties
    cases = []
    metrics_calc = SegmentationMetrics()

    for i in range(10):
        shape = (64, 128, 128)
        gt = np.zeros(shape, dtype=np.uint8)

        # Create different scenarios
        if i < 3:
            # Perfect predictions
            gt[20:40, 40:80, 50:90] = 1  # liver
            gt[25:35, 90:110, 60:80] = 2  # kidney
            pred = gt.copy()

        elif i < 6:
            # Good predictions with small errors
            gt[20:40, 40:80, 50:90] = 1
            gt[25:35, 90:110, 60:80] = 2
            gt[30:35, 85:95, 75:85] = 9  # small organ

            pred = gt.copy()
            # Add small amount of noise
            noise_mask = np.random.random(shape) < 0.02
            pred[noise_mask] = np.random.randint(0, 14, size=noise_mask.sum())

        else:
            # Challenging predictions
            gt[20:40, 40:80, 50:90] = 1
            gt[25:35, 90:110, 60:80] = 2
            gt[30:35, 85:95, 75:85] = 9
            gt[28:33, 70:75, 40:45] = 12  # very small organ

            pred = gt.copy()
            # More noise
            noise_mask = np.random.random(shape) < 0.05
            pred[noise_mask] = np.random.randint(0, 14, size=noise_mask.sum())

            # Miss some small organs
            if np.random.random() < 0.3:
                pred[pred == 12] = 0

        # Compute metrics
        metrics = metrics_calc.compute_all_metrics(pred, gt, case_id=f"validation_case_{i}")

        cases.append({
            'case_id': f"validation_case_{i}",
            'prediction': pred,
            'ground_truth': gt,
            'metrics': metrics
        })

        print(f"Case {i}: Dice = {metrics['mean_dice']:.3f}, Small Organ Dice = {metrics['small_organ_dice']:.3f}")

    return cases


def run_metrics_tests():
    """Run all metrics tests"""
    print("Testing Segmentation Metrics...")
    print("=" * 50)

    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSegmentationMetrics)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if result.wasSuccessful():
        print("\n✅ All metrics tests passed!")

        # Create validation dataset
        print("\n" + "=" * 50)
        validation_cases = create_validation_dataset()

        # Aggregate validation results
        metrics_calc = SegmentationMetrics()
        aggregated = metrics_calc.aggregate_metrics([case['metrics'] for case in validation_cases])

        print("\nValidation Dataset Summary:")
        print(f"Cases: {aggregated['num_cases']}")
        print(f"Overall Dice: {aggregated['overall_dice']['mean']:.3f} ± {aggregated['overall_dice']['std']:.3f}")
        print(f"Small Organ Dice: {aggregated['small_organ_dice']['mean']:.3f} ± {aggregated['small_organ_dice']['std']:.3f}")
        print(f"Large Organ Dice: {aggregated['large_organ_dice']['mean']:.3f} ± {aggregated['large_organ_dice']['std']:.3f}")

        return True
    else:
        print("\n❌ Some metrics tests failed!")
        return False


if __name__ == '__main__':
    success = run_metrics_tests()
    if not success:
        sys.exit(1)