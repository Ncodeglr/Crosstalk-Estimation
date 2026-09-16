"""
Unit tests for mode-aware functionality.
"""

import pytest
import torch
import torch.nn as nn
import torchcvnn.nn.modules as c_nn
import warnings

from cvnn.models.utils import (
    is_real_mode,
    get_activation,
    get_loss_function,
    get_normalization,
    init_weights_mode_aware,
    validate_layer_mode,
)
from cvnn.models.models import AutoEncoder, get_activation as model_get_activation
from cvnn.train import setup_loss_optimizer


class TestModeUtils:
    """Test mode utility functions."""

    def test_is_real_mode(self):
        """Test mode classification."""
        assert is_real_mode("real") is True
        assert is_real_mode("complex") is False
        assert is_real_mode("split") is False

        with pytest.raises(ValueError):
            is_real_mode("invalid")

    def test_validate_layer_mode(self):
        """Test layer mode validation."""
        # Valid modes should not raise
        for mode in ["complex", "real", "split"]:
            validate_layer_mode(mode)

        # Invalid mode should raise
        with pytest.raises(ValueError):
            validate_layer_mode("invalid")

    @pytest.mark.parametrize(
        "activation_name,layer_mode,expected_type",
        [
            ("modReLU", "real", nn.ReLU),
            ("modReLU", "complex", c_nn.modReLU),
            ("ReLU", "real", nn.ReLU),
            ("modTanh", "real", nn.Tanh),
        ],
    )
    def test_get_activation(self, activation_name, layer_mode, expected_type):
        """Test activation selection based on mode."""
        activation = get_activation(activation_name, layer_mode)
        assert isinstance(activation, expected_type)

    def test_get_activation_unknown_real_mode(self):
        """Test handling of unknown activations in real mode."""
        # Should raise ValueError, not fall back (as per updated requirements)
        with pytest.raises(ValueError):
            get_activation("UnknownActivation", "real")

    def test_get_activation_unknown_complex_mode(self):
        """Test handling of unknown activations in complex mode."""
        with pytest.raises(ValueError):
            get_activation("UnknownActivation", "complex")

    @pytest.mark.parametrize(
        "loss_name,layer_mode,expected_type",
        [
            ("MSE", "real", nn.MSELoss),
            ("MSE", "complex", c_nn.ComplexMSELoss),
            ("L1", "real", nn.L1Loss),
        ],
    )
    def test_get_loss_function(self, loss_name, layer_mode, expected_type):
        """Test loss function selection based on mode."""
        loss_fn = get_loss_function(loss_name, layer_mode)
        assert isinstance(loss_fn, expected_type)

    def test_get_loss_function_unknown(self):
        """Test handling of unknown loss functions."""
        with pytest.raises(ValueError):
            get_loss_function("UnknownLoss", "real")

    @pytest.mark.parametrize(
        "norm_type,layer_mode,expected_type",
        [
            ("batch", "real", nn.BatchNorm2d),
            ("batch", "complex", c_nn.BatchNorm2d),
            ("layer", "real", nn.LayerNorm),
            ("layer", "complex", c_nn.LayerNorm),
            (None, "real", nn.Identity),
            ("none", "real", nn.Identity),
        ],
    )
    def test_get_normalization(self, norm_type, layer_mode, expected_type):
        """Test normalization selection based on mode."""
        norm = get_normalization(norm_type, layer_mode, 64)
        assert isinstance(norm, expected_type)

    def test_init_weights_mode_aware(self):
        """Test mode-aware weight initialization."""
        # Test real mode
        conv_real = nn.Conv2d(3, 64, 3)
        init_weights_mode_aware(conv_real, "real")
        # Should complete without error

        # Test complex mode
        conv_complex = nn.Conv2d(3, 64, 3, dtype=torch.complex64)
        init_weights_mode_aware(conv_complex, "complex")
        # Should complete without error


class TestModeAwareModels:
    """Test mode-aware model creation and functionality."""

    @pytest.mark.parametrize("layer_mode", ["real", "complex", "split"])
    def test_autoencoder_creation(self, layer_mode):
        """Test AutoEncoder creation with different modes."""
        model = AutoEncoder(
            num_channels=3,
            num_layers=2,
            channels_width=8,
            input_size=32,
            activation="modReLU",
            upsampling_layer="transpose",
            layer_mode=layer_mode,
            normalization_layer="batch",
            residual=False,
            num_blocks=1,
        )

        assert hasattr(model, "layer_mode")
        assert model.layer_mode == layer_mode

    @pytest.mark.parametrize("layer_mode", ["real", "complex", "split"])
    def test_model_forward_pass(self, layer_mode):
        """Test forward pass with different modes."""
        model = AutoEncoder(
            num_channels=3,
            num_layers=2,
            channels_width=8,
            input_size=16,
            activation="modReLU",
            upsampling_layer="transpose",
            layer_mode=layer_mode,
            normalization_layer="batch",
            residual=False,
            num_blocks=1,
        )

        # Create appropriate input tensor
        batch_size = 2
        if layer_mode == "real":
            x = torch.randn(batch_size, 3, 16, 16)
        else:
            x = torch.complex(
                torch.randn(batch_size, 3, 16, 16), torch.randn(batch_size, 3, 16, 16)
            )

        # Forward pass should not raise errors
        with torch.no_grad():
            output = model(x)

        assert output.shape[0] == batch_size
        assert output.shape[1] == 3
        assert output.shape[2] == 16
        assert output.shape[3] == 16

    def test_model_get_activation_integration(self):
        """Test integration of get_activation in models."""
        # Test that the model's get_activation function works with layer_mode
        activation = model_get_activation("modReLU", "real")
        assert isinstance(activation, nn.ReLU)

        activation = model_get_activation("modReLU", "complex")
        assert isinstance(activation, c_nn.modReLU)


class TestTrainingIntegration:
    """Test training pipeline integration."""

    def test_setup_loss_optimizer_mode_aware(self):
        """Test mode-aware loss selection in training setup."""
        # Create models with different modes
        models = {
            "real": AutoEncoder(
                num_channels=3,
                num_layers=2,
                channels_width=8,
                input_size=16,
                activation="modReLU",
                upsampling_layer="transpose",
                layer_mode="real",
                normalization_layer="batch",
                residual=False,
                num_blocks=1,
            ),
            "complex": AutoEncoder(
                num_channels=3,
                num_layers=2,
                channels_width=8,
                input_size=16,
                activation="modReLU",
                upsampling_layer="transpose",
                layer_mode="complex",
                normalization_layer="batch",
                residual=False,
                num_blocks=1,
            ),
        }

        cfg = {
            "loss": {"name": "MSE"},
            "optim": {"algo": "SGD", "params": {"lr": 0.01}},
            "model": {"cov_mode": "diag"},
        }
        # Current train.setup_loss_optimizer expects a task key in cfg
        cfg["task"] = "reconstruction"

        # Provide dummy dataset and device as required by current signature
        dummy_dataset = None
        device = "cpu"
        # Add minimal data entry expected by setup_loss_optimizer
        cfg["data"] = {"num_channels": 3}
        # Test real mode
        loss_fn, optimizer = setup_loss_optimizer(
            models["real"], cfg, dummy_dataset, device
        )
        assert isinstance(loss_fn, nn.MSELoss)

        # Test complex mode
        loss_fn, optimizer = setup_loss_optimizer(
            models["complex"], cfg, dummy_dataset, device
        )
        assert isinstance(loss_fn, c_nn.ComplexMSELoss)

    def test_setup_loss_optimizer_fallback(self):
        """Test fallback behavior for old-style loss configuration."""
        # Create model without layer_mode or with old-style loss name
        model = AutoEncoder(
            num_channels=3,
            num_layers=2,
            channels_width=8,
            input_size=16,
            activation="modReLU",
            upsampling_layer="transpose",
            layer_mode="complex",
            normalization_layer="batch",
            residual=False,
            num_blocks=1,
        )

        # Test with old-style specific loss name
        cfg = {
            "loss": {"name": "MSE"},
            "optim": {"algo": "SGD", "params": {"lr": 0.01}},
            "model": {"cov_mode": "diag"},
        }
        cfg["task"] = "reconstruction"

        # provide dummy dataset/device for current signature
        dummy_dataset = None
        device = "cpu"
        cfg["data"] = {"num_channels": 3}
        loss_fn, optimizer = setup_loss_optimizer(model, cfg, dummy_dataset, device)
        # Should still work with fallback
        assert isinstance(loss_fn, c_nn.ComplexMSELoss)


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_layer_mode_in_model(self):
        """Test that invalid layer modes raise appropriate errors."""
        with pytest.raises(ValueError):
            AutoEncoder(
                num_channels=3,
                num_layers=2,
                channels_width=8,
                input_size=16,
                activation="modReLU",
                upsampling_layer="transpose",
                layer_mode="invalid",
                normalization_layer="batch",
                residual=False,
                num_blocks=1,
            )

    def test_missing_complex_activation(self):
        """Test handling when complex activation is not available."""
        # This test depends on which activations are available in torchcvnn
        # We test the error handling mechanism
        with pytest.raises(ValueError):
            get_activation("NonExistentComplexActivation", "complex")

    def test_missing_complex_loss(self):
        """Test handling when complex loss is not available."""
        # Test with a loss that might not be available
        with pytest.raises(ValueError):
            get_loss_function("NonExistentLoss", "complex")


if __name__ == "__main__":
    pytest.main([__file__])
