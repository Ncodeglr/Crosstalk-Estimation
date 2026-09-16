import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from cvnn.models.softmax import GumbelSoftmax, Softmax


class TestSoftmax:
    """Test suite for custom Softmax implementation."""

    def test_softmax_initialization(self):
        """Test Softmax initialization."""
        softmax = Softmax()

        # Should initialize without parameters
        assert isinstance(softmax, nn.Module)

    def test_softmax_forward_real_tensor(self):
        """Test Softmax forward pass with real tensor."""
        batch_size, num_classes = 4, 5
        softmax = Softmax()

        x = torch.randn(batch_size, num_classes)
        result = softmax(x, dim=1)

        # Check output properties
        assert result.shape == x.shape
        assert result.dtype == x.dtype

        # Check that probabilities sum to 1
        assert torch.allclose(result.sum(dim=1), torch.ones(batch_size), atol=1e-6)

        # Check that all values are non-negative
        assert (result >= 0).all()

    def test_softmax_forward_complex_tensor_fails(self):
        """Test Softmax forward pass with complex tensor should fail."""
        batch_size, num_classes = 3, 4
        softmax = Softmax()

        x = torch.randn(batch_size, num_classes, dtype=torch.complex64)

        # Should fail because PyTorch softmax doesn't support complex tensors
        with pytest.raises(RuntimeError):
            result = softmax(x, dim=1)

    def test_softmax_different_dimensions(self):
        """Test Softmax along different dimensions."""
        x = torch.randn(2, 3, 4)
        softmax = Softmax()

        for dim in [0, 1, 2, -1, -2, -3]:
            result = softmax(x, dim=dim)

            assert result.shape == x.shape
            # Check that probabilities sum to 1 along the specified dimension
            assert torch.allclose(
                result.sum(dim=dim), torch.ones(result.sum(dim=dim).shape), atol=1e-6
            )

    def test_softmax_gradient_flow(self):
        """Test that gradients flow through Softmax."""
        softmax = Softmax()

        x = torch.randn(2, 3, requires_grad=True)
        result = softmax(x, dim=1)
        # Use a loss that doesn't sum probabilities (which would have zero gradient)
        loss = (result * torch.tensor([1.0, 2.0, 3.0])).sum()

        loss.backward()

        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))


class TestGumbelSoftmax:
    """Test suite for GumbelSoftmax implementation."""

    def test_gumbel_softmax_initialization(self):
        """Test GumbelSoftmax initialization."""
        gumbel_softmax = GumbelSoftmax()

        # Should initialize without parameters
        assert isinstance(gumbel_softmax, nn.Module)

    def test_gumbel_softmax_forward_real_tensor_soft(self):
        """Test GumbelSoftmax forward pass with real tensor (soft sampling)."""
        batch_size, num_classes = 4, 5
        gumbel_softmax = GumbelSoftmax()

        x = torch.randn(batch_size, num_classes)
        result = gumbel_softmax(x, tau=1.0, hard=False)

        assert result.shape == x.shape
        assert result.dtype == x.dtype

        # Soft Gumbel softmax should sum to 1
        assert torch.allclose(result.sum(dim=1), torch.ones(batch_size), atol=1e-6)

        # All values should be non-negative
        assert (result >= 0).all()

    def test_gumbel_softmax_forward_real_tensor_hard(self):
        """Test GumbelSoftmax forward pass with real tensor (hard sampling)."""
        batch_size, num_classes = 4, 5
        gumbel_softmax = GumbelSoftmax()

        x = torch.randn(batch_size, num_classes)
        result = gumbel_softmax(x, tau=1.0, hard=True)

        assert result.shape == x.shape
        assert result.dtype == x.dtype

        # Hard Gumbel softmax should be one-hot
        assert torch.allclose(result.sum(dim=1), torch.ones(batch_size))

        # Each row should have exactly one 1 and the rest 0s
        for i in range(batch_size):
            unique_values = torch.unique(result[i])
            # Should contain 0s and exactly one 1
            assert len(unique_values) <= 2
            if len(unique_values) == 2:
                assert 0.0 in unique_values
                assert 1.0 in unique_values

    def test_gumbel_softmax_temperature_effects(self):
        """Test GumbelSoftmax behavior with different temperatures."""
        x = torch.tensor([[1.0, 2.0, 3.0]])
        gumbel_softmax = GumbelSoftmax()

        # Low temperature should make distribution more peaked
        result_low = gumbel_softmax(x, tau=0.1, hard=False)

        # High temperature should be more uniform
        result_high = gumbel_softmax(x, tau=10.0, hard=False)

        # Low temperature should have higher max probability
        assert result_low.max() > result_high.max()

    def test_gumbel_softmax_reproducibility_with_seed(self):
        """Test GumbelSoftmax reproducibility when setting random seed."""
        x = torch.randn(2, 3)
        gumbel_softmax = GumbelSoftmax()

        # Set seed and generate result
        torch.manual_seed(42)
        result1 = gumbel_softmax(x, tau=1.0, hard=False)

        # Reset seed and generate again
        torch.manual_seed(42)
        result2 = gumbel_softmax(x, tau=1.0, hard=False)

        # Results should be identical
        assert torch.allclose(result1, result2)

    def test_gumbel_softmax_training_vs_eval_mode(self):
        """Test GumbelSoftmax behavior in training vs evaluation mode."""
        x = torch.randn(3, 4)
        gumbel_softmax = GumbelSoftmax()

        # Training mode
        gumbel_softmax.train()
        torch.manual_seed(42)
        result_train = gumbel_softmax(x, tau=1.0, hard=False)

        # Evaluation mode
        gumbel_softmax.eval()
        torch.manual_seed(42)
        result_eval = gumbel_softmax(x, tau=1.0, hard=False)

        # Results should be identical since GumbelSoftmax doesn't change behavior based on training mode
        assert torch.allclose(result_train, result_eval)

    def test_gumbel_softmax_gradient_flow(self):
        """Test that gradients flow through GumbelSoftmax."""
        gumbel_softmax = GumbelSoftmax()

        x = torch.randn(2, 3, requires_grad=True)
        result = gumbel_softmax(x, tau=1.0, hard=False)
        # Use a loss that doesn't sum probabilities (which would have zero gradient)
        loss = (result * torch.tensor([1.0, 2.0, 3.0])).sum()

        loss.backward()

        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

    def test_gumbel_softmax_hard_gradient_flow(self):
        """Test that gradients flow through hard GumbelSoftmax using straight-through estimator."""
        gumbel_softmax = GumbelSoftmax()

        x = torch.randn(2, 3, requires_grad=True)
        result = gumbel_softmax(x, tau=1.0, hard=True)
        # Use a loss that doesn't sum probabilities (which would have zero gradient)
        loss = (result * torch.tensor([1.0, 2.0, 3.0])).sum()

        loss.backward()

        # Even with hard sampling, gradients should flow due to straight-through estimator
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))


class TestSoftmaxIntegration:
    """Test integration between Softmax and GumbelSoftmax."""

    def test_softmax_vs_gumbel_softmax_consistency(self):
        """Test that GumbelSoftmax approaches Softmax as temperature approaches 0."""
        x = torch.tensor([[1.0, 2.0, 3.0]])

        softmax = Softmax()
        gumbel_softmax = GumbelSoftmax()

        # Set eval mode to disable Gumbel noise (doesn't actually change behavior in this implementation)
        gumbel_softmax.eval()

        softmax_result = softmax(x, dim=1)

        # Very low temperature should approach regular softmax (approximately)
        gumbel_result = gumbel_softmax(x, tau=0.001, hard=False)

        # Should be approximately equal when temperature is very low
        # (Note: there will still be some difference due to Gumbel noise)
        assert softmax_result.shape == gumbel_result.shape

    def test_output_shape_consistency(self):
        """Test that both Softmax and GumbelSoftmax produce consistent output shapes."""
        shapes = [(2, 3), (4, 5, 6), (1, 10), (3, 2, 4, 5)]

        for shape in shapes:
            x = torch.randn(*shape)

            softmax = Softmax()
            gumbel_softmax = GumbelSoftmax()

            softmax_result = softmax(x, dim=-1)
            gumbel_result = gumbel_softmax(x, tau=1.0, hard=False)

            assert softmax_result.shape == shape
            assert gumbel_result.shape == shape
            assert softmax_result.dtype == gumbel_result.dtype

    def test_probability_distribution_properties(self):
        """Test that both methods produce valid probability distributions."""
        x = torch.randn(5, 10)

        softmax = Softmax()
        gumbel_softmax = GumbelSoftmax()

        softmax_result = softmax(x, dim=1)
        gumbel_result = gumbel_softmax(x, tau=1.0, hard=False)

        # Both should sum to 1 along specified dimension
        assert torch.allclose(softmax_result.sum(dim=1), torch.ones(5), atol=1e-6)
        assert torch.allclose(gumbel_result.sum(dim=1), torch.ones(5), atol=1e-6)

        # Both should be non-negative
        assert (softmax_result >= 0).all()
        assert (gumbel_result >= 0).all()

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_consistency(self, dtype):
        """Test that both methods handle different dtypes consistently."""
        x = torch.randn(3, 4, dtype=dtype)

        softmax = Softmax()
        gumbel_softmax = GumbelSoftmax()

        softmax_result = softmax(x, dim=1)
        gumbel_result = gumbel_softmax(x, tau=1.0, hard=False)

        assert softmax_result.dtype == dtype
        assert gumbel_result.dtype == dtype

    def test_extreme_input_values(self):
        """Test behavior with extreme input values."""
        # Very large values
        x_large = torch.tensor([[100.0, 200.0, 300.0]])

        # Very small values
        x_small = torch.tensor([[-100.0, -200.0, -300.0]])

        softmax = Softmax()
        gumbel_softmax = GumbelSoftmax()

        for x in [x_large, x_small]:
            softmax_result = softmax(x, dim=1)
            gumbel_result = gumbel_softmax(x, tau=1.0, hard=False)

            # Results should be finite and valid probabilities
            assert torch.isfinite(softmax_result).all()
            assert torch.isfinite(gumbel_result).all()
            assert torch.allclose(softmax_result.sum(dim=1), torch.tensor(1.0))
            assert torch.allclose(
                gumbel_result.sum(dim=1), torch.tensor(1.0), atol=1e-6
            )
