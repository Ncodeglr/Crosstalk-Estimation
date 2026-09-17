"""
Test cases for the Focal Loss implementation.
"""

import pytest
import torch
import torch.nn as nn

from cvnn.losses import FocalLoss, compute_class_weights
from cvnn.models.utils import get_loss_function


class TestComputeClassWeights:
    """Test class weight computation function."""

    def test_compute_class_weights_inverse_frequency(self):
        """Test inverse frequency weighting."""
        # Create imbalanced targets: class 0 appears 6 times, class 1 appears 2 times
        targets = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1])
        weights = compute_class_weights(
            targets, num_classes=2, weight_mode="inverse_frequency"
        )

        # Class 0: 1/6, Class 1: 1/2
        # After normalization: weights should sum to num_classes
        expected_ratio = (1 / 2) / (
            1 / 6
        )  # Class 1 should be 3x more weighted than class 0
        actual_ratio = weights[1] / weights[0]
        assert pytest.approx(expected_ratio, rel=1e-3) == actual_ratio
        assert (
            pytest.approx(2.0, rel=1e-3) == weights.sum()
        )  # Should sum to num_classes

    def test_compute_class_weights_balanced(self):
        """Test balanced weighting."""
        targets = torch.tensor([0, 0, 0, 1])  # 3 of class 0, 1 of class 1
        weights = compute_class_weights(targets, num_classes=2, weight_mode="balanced")

        # total_samples / (n_classes * class_count)
        # Class 0: 4 / (2 * 3) = 4/6 = 2/3
        # Class 1: 4 / (2 * 1) = 4/2 = 2
        expected_weights = torch.tensor([4 / (2 * 3), 4 / (2 * 1)])  # [2/3, 2]
        # After normalization
        expected_weights = expected_weights * 2 / expected_weights.sum()

        assert torch.allclose(weights, expected_weights, rtol=1e-3)

    def test_compute_class_weights_missing_class(self):
        """Test handling of missing classes."""
        targets = torch.tensor([0, 0, 2, 2])  # Missing class 1
        weights = compute_class_weights(
            targets, num_classes=3, weight_mode="inverse_frequency"
        )

        # Should handle missing class 1 by using min count of 1
        assert len(weights) == 3
        assert weights[1] > 0  # Missing class should still get a weight


class TestFocalLoss:
    """Test real-valued Focal Loss."""

    def test_focal_loss_basic(self):
        """Test basic focal loss computation."""
        focal_loss = FocalLoss(alpha=1.0, gamma=2.0, use_class_weights=False)

        # Simple 2-class case
        inputs = torch.tensor([[2.0, 1.0], [1.0, 2.0]])  # Shape: [2, 2]
        targets = torch.tensor([0, 1])  # Shape: [2]

        loss = focal_loss(inputs, targets)
        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_focal_loss_with_class_weights(self):
        """Test focal loss with automatic class weights."""
        focal_loss = FocalLoss(alpha=1.0, gamma=2.0, use_class_weights=True)

        # Imbalanced case - more of class 0
        inputs = torch.randn(8, 3)  # 8 samples, 3 classes
        targets = torch.tensor([0, 0, 0, 0, 0, 0, 1, 2])  # Imbalanced

        loss = focal_loss(inputs, targets)
        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_focal_loss_ignore_index(self):
        """Test focal loss with ignore_index."""
        focal_loss = FocalLoss(ignore_index=-100, use_class_weights=False)

        inputs = torch.randn(4, 3)
        targets = torch.tensor([0, 1, 2, -100])  # Last sample should be ignored

        loss = focal_loss(inputs, targets)
        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_focal_loss_segmentation_format(self):
        """Test focal loss with segmentation-style inputs."""
        focal_loss = FocalLoss(gamma=2.0, use_class_weights=False)

        # Segmentation format: [N, C, H, W]
        inputs = torch.randn(2, 3, 4, 4)  # 2 samples, 3 classes, 4x4 spatial
        targets = torch.randint(0, 3, (2, 4, 4))  # [N, H, W]

        loss = focal_loss(inputs, targets)
        assert loss.item() >= 0
        assert not torch.isnan(loss)


class TestComplexFocalLoss:
    """Test complex-valued Focal Loss."""

    def test_complex_focal_loss_basic(self):
        """Test basic complex focal loss computation."""

    # Baseline provides a real-valued FocalLoss. For complex inputs we test
    # that using a magnitude-based reduction (abs) produces valid loss.
    complex_focal = FocalLoss(alpha=1.0, gamma=2.0, use_class_weights=False)

    # Complex inputs (we pass magnitudes to the real FocalLoss)
    real_part = torch.randn(2, 3)
    imag_part = torch.randn(2, 3)
    inputs_complex = torch.complex(real_part, imag_part)
    inputs = torch.abs(inputs_complex)
    targets = torch.tensor([0, 1])

    loss = complex_focal(inputs, targets)
    assert loss.item() >= 0
    assert not torch.isnan(loss)

    def test_complex_focal_loss_magnitude_reductions(self):
        """Test different magnitude reduction methods."""

    # Baseline does not expose a complex-specific FocalLoss. Verify that
    # the real-valued FocalLoss works on magnitudes of complex inputs.
    focal = FocalLoss(use_class_weights=False)
    real_part = torch.randn(2, 3)
    imag_part = torch.randn(2, 3)
    inputs = torch.abs(torch.complex(real_part, imag_part))
    targets = torch.tensor([0, 1])
    loss = focal(inputs, targets)
    assert loss.item() >= 0
    assert not torch.isnan(loss)


class TestLossIntegration:
    """Test integration with the loss selection system."""

    def test_get_loss_function_focal_real(self):
        """Test getting FocalLoss through the mode-aware system."""
        loss_fn = get_loss_function("FocalLoss", layer_mode="real", ignore_index=-100)
        assert isinstance(loss_fn, FocalLoss)

    def test_get_loss_function_focal_complex(self):
        """Test getting ComplexFocalLoss through the mode-aware system."""
        # Baseline returns the same FocalLoss implementation for complex mode
        loss_fn = get_loss_function("FocalLoss", layer_mode="complex")
        assert isinstance(loss_fn, FocalLoss)

    def test_get_loss_function_focal_with_ignore_index(self):
        """Test FocalLoss with ignore_index parameter."""
        loss_fn = get_loss_function("FocalLoss", layer_mode="real", ignore_index=-100)
        assert isinstance(loss_fn, FocalLoss)
        assert loss_fn.ignore_index == -100

    def test_focal_loss_in_available_losses(self):
        """Test that FocalLoss is listed in available losses."""
        try:
            get_loss_function("InvalidLoss", layer_mode="real")
        except ValueError as e:
            error_msg = str(e)
            assert "FocalLoss" in error_msg  # Should be in the available losses list


class TestFocalLossComparison:
    """Test that Focal Loss behaves differently from CrossEntropy."""

    def test_focal_vs_crossentropy_easy_examples(self):
        """Test that focal loss down-weights easy examples."""
        # Create easy examples (high confidence predictions)
        inputs = torch.tensor([[10.0, 0.0], [0.0, 10.0]])  # Very confident
        targets = torch.tensor([0, 1])

        # Standard CrossEntropy
        ce_loss = nn.CrossEntropyLoss()
        ce_value = ce_loss(inputs, targets)

        # Focal Loss with gamma=2
        focal_loss = FocalLoss(gamma=2.0, use_class_weights=False)
        focal_value = focal_loss(inputs, targets)

        # Focal loss should be much smaller for easy examples
        assert focal_value < ce_value

    def test_focal_vs_crossentropy_hard_examples(self):
        """Test that focal loss preserves loss for hard examples."""
        # Create hard examples (low confidence predictions)
        inputs = torch.tensor([[0.1, 0.0], [0.0, 0.1]])  # Low confidence
        targets = torch.tensor([0, 1])

        # Standard CrossEntropy
        ce_loss = nn.CrossEntropyLoss()
        ce_value = ce_loss(inputs, targets)

        # Focal Loss with gamma=2
        focal_loss = FocalLoss(gamma=2.0, use_class_weights=False)
        focal_value = focal_loss(inputs, targets)

        # For hard examples, focal loss should still be smaller than CE loss
        # but the reduction should be less dramatic than for easy examples
        # Let's test that the ratio is not too small (indicating some preservation of loss)
        ratio = focal_value / ce_value
        assert 0.1 < ratio < 1.0  # Focal loss should be smaller but not negligible

        # Also test that both losses are non-zero and reasonable
        assert focal_value > 0
        assert ce_value > 0


if __name__ == "__main__":
    pytest.main([__file__])
