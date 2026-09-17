import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from cvnn.models.projection import PolyCtoR, MLPCtoR





class TestPolyCtoR:
    """Test suite for PolyCtoR projection module."""

    def test_poly_cto_r_initialization(self):
        """Test PolyCtoR initialization."""
        order = 3
        poly_cto_r = PolyCtoR(order=order)

        assert poly_cto_r.order == order

        # Check the linear layer size
        expected_size = ((order + 1) * (order + 2)) // 2 - 1
        assert poly_cto_r.poly.in_features == expected_size
        assert poly_cto_r.poly.out_features == 1

    def test_poly_cto_r_forward_2d_input(self):
        """Test PolyCtoR forward pass with 2D input."""
        batch_size, channels = 4, 8
        order = 2

        poly_cto_r = PolyCtoR(order=order)
        x = torch.randn(batch_size, channels, dtype=torch.complex64)

        result = poly_cto_r(x)

        # Output should be real-valued with same shape except last dimension
        assert result.shape == (batch_size, channels)
        assert result.dtype == torch.float32

    def test_poly_cto_r_forward_4d_input(self):
        """Test PolyCtoR forward pass with 4D input (images)."""
        batch_size, channels, height, width = 2, 3, 8, 8
        order = 2

        poly_cto_r = PolyCtoR(order=order)
        x = torch.randn(batch_size, channels, height, width, dtype=torch.complex64)

        result = poly_cto_r(x)

        # Output should preserve spatial dimensions but be real-valued
        assert result.shape == (batch_size, channels, height, width)
        assert result.dtype == torch.float32

    def test_poly_cto_r_different_orders(self):
        """Test PolyCtoR with different polynomial orders."""
        batch_size, channels = 2, 4
        x = torch.randn(batch_size, channels, dtype=torch.complex64)

        for order in [1, 2, 3, 4]:
            poly_cto_r = PolyCtoR(order=order)
            result = poly_cto_r(x)

            assert result.shape == (batch_size, channels)
            assert result.dtype == torch.float32

    def test_poly_cto_r_gradient_flow(self):
        """Test that gradients flow through PolyCtoR."""
        order = 2
        poly_cto_r = PolyCtoR(order=order)

        x = torch.randn(2, 3, dtype=torch.complex64, requires_grad=True)
        result = poly_cto_r(x)
        loss = result.sum()

        loss.backward()

        # Check that gradients are computed
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

    def test_poly_cto_r_complex128_input(self):
        """Test PolyCtoR with complex128 input."""
        batch_size, channels = 2, 4
        order = 2

        poly_cto_r = PolyCtoR(order=order)
        # Convert model to double precision to match complex128 input
        poly_cto_r = poly_cto_r.double()
        x = torch.randn(batch_size, channels, dtype=torch.complex128)

        result = poly_cto_r(x)

        # Should handle complex128 and output float64
        assert result.shape == (batch_size, channels)
        assert result.dtype == torch.float64


class TestMLPCtoR:
    """Test suite for MLPCtoR projection module."""

    def test_mlp_cto_r_initialization_default(self):
        """Test MLPCtoR initialization with default parameters."""
        mlp_cto_r = MLPCtoR()

        # Check that it has the mlp sequential module
        assert hasattr(mlp_cto_r, "mlp")
        assert isinstance(mlp_cto_r.mlp, torch.nn.Sequential)

        # Check default architecture: 2 -> 8 -> 16 -> 1 with ReLU activations
        layers = list(mlp_cto_r.mlp.children())
        linear_layers = [
            layer for layer in layers if isinstance(layer, torch.nn.Linear)
        ]
        relu_layers = [layer for layer in layers if isinstance(layer, torch.nn.ReLU)]

        assert len(linear_layers) == 3  # Three linear layers
        assert len(relu_layers) == 2  # Two ReLU layers (between linear layers)

        # Check layer dimensions
        assert linear_layers[0].in_features == 2  # real and imaginary parts
        assert linear_layers[0].out_features == 8  # first hidden layer
        assert linear_layers[1].in_features == 8  # first hidden layer
        assert linear_layers[1].out_features == 16  # second hidden layer
        assert linear_layers[2].in_features == 16  # second hidden layer
        assert linear_layers[2].out_features == 1  # output dimension

    def test_mlp_cto_r_initialization_custom(self):
        """Test MLPCtoR initialization (no custom parameters in this implementation)."""
        # The current implementation doesn't support custom parameters
        # Just test that it can be initialized
        mlp_cto_r = MLPCtoR()

        # Verify it has the expected structure
        assert hasattr(mlp_cto_r, "mlp")
        assert isinstance(mlp_cto_r.mlp, nn.Sequential)

    def test_mlp_cto_r_forward_2d_input(self):
        """Test MLPCtoR forward pass with 2D input."""
        batch_size, channels = 4, 8

        mlp_cto_r = MLPCtoR()
        x = torch.randn(batch_size, channels, dtype=torch.complex64)

        result = mlp_cto_r(x)

        assert result.shape == (batch_size, channels)
        assert result.dtype == torch.float32

    def test_mlp_cto_r_forward_4d_input(self):
        """Test MLPCtoR forward pass with 4D input (images)."""
        batch_size, channels, height, width = 2, 3, 8, 8

        mlp_cto_r = MLPCtoR()
        x = torch.randn(batch_size, channels, height, width, dtype=torch.complex64)

        result = mlp_cto_r(x)

        assert result.shape == (batch_size, channels, height, width)
        assert result.dtype == torch.float32

    def test_mlp_cto_r_training_mode(self):
        """Test MLPCtoR behavior in training vs eval mode."""
        # This implementation doesn't have dropout, so behavior should be identical
        mlp_cto_r = MLPCtoR()
        x = torch.randn(10, 5, dtype=torch.complex64)

        # Training mode
        mlp_cto_r.train()
        result_train = mlp_cto_r(x)

        # Eval mode
        mlp_cto_r.eval()
        result_eval = mlp_cto_r(x)

        # Results should be identical since there's no dropout
        assert result_train.shape == result_eval.shape
        assert torch.allclose(result_train, result_eval)

    def test_mlp_cto_r_gradient_flow(self):
        """Test that gradients flow through MLPCtoR."""
        mlp_cto_r = MLPCtoR()

        x = torch.randn(2, 3, dtype=torch.complex64, requires_grad=True)
        result = mlp_cto_r(x)
        loss = result.sum()

        loss.backward()

        # Check that gradients are computed
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

    def test_mlp_cto_r_no_dropout(self):
        """Test MLPCtoR (no dropout in current implementation)."""
        mlp_cto_r = MLPCtoR()
        x = torch.randn(4, 6, dtype=torch.complex64)

        # Should work without dropout layers
        result = mlp_cto_r(x)
        assert result.shape == (4, 6)
        assert result.dtype == torch.float32

    def test_mlp_cto_r_different_activations(self):
        """Test MLPCtoR (fixed activation in current implementation)."""
        x = torch.randn(2, 3, dtype=torch.complex64)

        # Current implementation has fixed ReLU activation
        mlp_cto_r = MLPCtoR()
        result = mlp_cto_r(x)

        assert result.shape == (2, 3)
        assert result.dtype == torch.float32


class TestProjectionIntegration:
    """Test integration between different projection modules."""

    def test_poly_vs_mlp_output_shapes(self):
        """Test that PolyCtoR and MLPCtoR produce same output shapes."""
        batch_size, channels, height, width = 2, 4, 8, 8
        x = torch.randn(batch_size, channels, height, width, dtype=torch.complex64)

        poly_cto_r = PolyCtoR(order=2)
        mlp_cto_r = MLPCtoR()

        poly_result = poly_cto_r(x)
        mlp_result = mlp_cto_r(x)

        assert poly_result.shape == mlp_result.shape
        assert poly_result.dtype == mlp_result.dtype

    def test_projection_consistency_across_dtypes(self):
        """Test that projections work consistently across different complex dtypes."""
        batch_size, channels = 3, 5

        # Test both complex64 and complex128
        for dtype in [torch.complex64, torch.complex128]:
            x = torch.randn(batch_size, channels, dtype=dtype)

            poly_cto_r = PolyCtoR(order=2)
            mlp_cto_r = MLPCtoR()

            # Convert models to appropriate precision for complex128
            if dtype == torch.complex128:
                poly_cto_r = poly_cto_r.double()
                mlp_cto_r = mlp_cto_r.double()

            poly_result = poly_cto_r(x)
            mlp_result = mlp_cto_r(x)

            assert poly_result.shape == (batch_size, channels)
            assert mlp_result.shape == (batch_size, channels)

            # Output dtype should match input precision
            if dtype == torch.complex64:
                assert poly_result.dtype == torch.float32
                assert mlp_result.dtype == torch.float32
            else:
                assert poly_result.dtype == torch.float64
                assert mlp_result.dtype == torch.float64

    @pytest.mark.parametrize(
        "input_shape",
        [
            (3, 5),  # 2D (supported by MLPCtoR)
            (1, 2, 3, 5),  # 4D (supported by MLPCtoR)
        ],
    )
    def test_projection_different_input_shapes(self, input_shape):
        """Test projections with different input shapes."""
        x = torch.randn(*input_shape, dtype=torch.complex64)

        poly_cto_r = PolyCtoR(order=2)
        mlp_cto_r = MLPCtoR()

        poly_result = poly_cto_r(x)
        mlp_result = mlp_cto_r(x)

        # Output should have same shape as input
        assert poly_result.shape == input_shape
        assert mlp_result.shape == input_shape

    def test_projection_poly_cto_r_various_shapes(self):
        """Test PolyCtoR with various input shapes (more flexible than MLPCtoR)."""
        input_shapes = [
            (5,),  # 1D
            (3, 5),  # 2D
            (2, 3, 5),  # 3D
            (1, 2, 3, 5),  # 4D
            (1, 1, 2, 3, 5),  # 5D
        ]

        poly_cto_r = PolyCtoR(order=2)

        for input_shape in input_shapes:
            x = torch.randn(*input_shape, dtype=torch.complex64)
            result = poly_cto_r(x)

            # PolyCtoR should preserve input shape
            assert result.shape == input_shape

    def test_projection_numerical_stability(self):
        """Test projection modules with extreme values."""
        # Test with very large values
        x_large = torch.tensor([[1e6 + 1e6j]], dtype=torch.complex64)

        # Test with very small values
        x_small = torch.tensor([[1e-6 + 1e-6j]], dtype=torch.complex64)

        # Test with zero
        x_zero = torch.tensor([[0 + 0j]], dtype=torch.complex64)

        poly_cto_r = PolyCtoR(order=2)
        mlp_cto_r = MLPCtoR()

        for x in [x_large, x_small, x_zero]:
            poly_result = poly_cto_r(x)
            mlp_result = mlp_cto_r(x)

            # Results should be finite
            assert torch.isfinite(poly_result).all()
            assert torch.isfinite(mlp_result).all()

            # Shape should be preserved
            assert poly_result.shape == x.shape
            assert mlp_result.shape == x.shape
