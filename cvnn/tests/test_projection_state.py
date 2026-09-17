"""
Test suite for projection state management functionality.

This module specifically tests the robust state management for projection functions 
that we implemented to ensure correct saving/restoring during retrain and checkpointing,
similar to the gumbel tau mechanism.

Note: Basic projection functionality is tested in test_projection.py.
This file focuses on the state management aspects.
"""

import pytest
import torch
import tempfile
import yaml
from pathlib import Path

# Add src to path for imports
import sys

sys.path.append("src")

from cvnn.models.projection import PolyCtoR, MLPCtoR
from cvnn.models.utils import get_projection
from cvnn.config_utils import get_model_params


class TestProjectionConfigIntegration:
    """Test projection config integration with get_projection function."""

    def test_polynomial_config_integration(self):
        """Test that polynomial projection config is properly used."""
        config = {"order": 5}
        proj = get_projection(
            "polynomial", layer_mode="complex", projection_config=config
        )

        assert isinstance(proj, PolyCtoR)
        expected_features = ((5 + 1) * (5 + 2)) // 2 - 1  # = 20
        assert proj.poly.in_features == expected_features
        assert proj.order == 5

    def test_mlp_config_integration(self):
        """Test that MLP projection config is properly used."""
        config = {"hidden_sizes": [4, 8, 4], "input_size": 2, "output_size": 1}
        proj = get_projection("MLP", layer_mode="complex", projection_config=config)

        assert isinstance(proj, MLPCtoR)

        # Check the architecture
        layers = list(proj.mlp.children())
        linear_layers = [
            layer for layer in layers if isinstance(layer, torch.nn.Linear)
        ]

        # Should be: 2->4, 4->8, 8->4, 4->1 = 4 linear layers
        assert len(linear_layers) == 4
        assert linear_layers[0].in_features == 2
        assert linear_layers[0].out_features == 4
        assert linear_layers[1].in_features == 4
        assert linear_layers[1].out_features == 8
        assert linear_layers[2].in_features == 8
        assert linear_layers[2].out_features == 4
        assert linear_layers[3].in_features == 4
        assert linear_layers[3].out_features == 1

    def test_default_configs_when_none_provided(self):
        """Test that default configs are used when none provided."""
        # Test polynomial default
        poly_proj = get_projection("polynomial", layer_mode="complex")
        assert isinstance(poly_proj, PolyCtoR)
        assert poly_proj.order == 3  # Default order

        # Test MLP default
        mlp_proj = get_projection("MLP", layer_mode="complex")
        assert isinstance(mlp_proj, MLPCtoR)

        # Check default architecture: 2 -> 8 -> 16 -> 1
        layers = list(mlp_proj.mlp.children())
        linear_layers = [
            layer for layer in layers if isinstance(layer, torch.nn.Linear)
        ]
        assert len(linear_layers) == 3
        assert linear_layers[0].in_features == 2
        assert linear_layers[0].out_features == 8

    def test_real_mode_behavior(self):
        """Test that real mode returns Identity regardless of config."""
        configs = [{"order": 5}, {"hidden_sizes": [16, 32]}, {}]

        for config in configs:
            for proj_type in ["polynomial", "MLP", "amplitude"]:
                proj = get_projection(
                    proj_type, layer_mode="real", projection_config=config
                )
                assert isinstance(proj, torch.nn.Identity)


class TestProjectionStateManagement:
    """Test projection state saving and restoring functionality."""

    def test_polynomial_state_save_restore(self):
        """Test saving and restoring polynomial projection state."""
        # Create two identical projections
        config = {"order": 4}
        proj1 = get_projection(
            "polynomial", layer_mode="complex", projection_config=config
        )
        proj2 = get_projection(
            "polynomial", layer_mode="complex", projection_config=config
        )

        # Verify they start different (random initialization)
        original_equal = torch.equal(proj1.poly.weight, proj2.poly.weight)
        # They should be different due to random initialization

        # Modify proj1 weights to specific values
        with torch.no_grad():
            proj1.poly.weight.fill_(0.5)
            proj1.poly.bias.fill_(0.1)

        # Save proj1 state
        state1 = proj1.state_dict()

        # Load state into proj2
        proj2.load_state_dict(state1)

        # Verify they now match
        assert torch.equal(proj1.poly.weight, proj2.poly.weight)
        assert torch.equal(proj1.poly.bias, proj2.poly.bias)
        assert torch.allclose(
            proj2.poly.weight, torch.full_like(proj2.poly.weight, 0.5)
        )
        assert torch.allclose(proj2.poly.bias, torch.full_like(proj2.poly.bias, 0.1))

    def test_mlp_state_save_restore(self):
        """Test saving and restoring MLP projection state."""
        # Create two identical projections
        config = {"hidden_sizes": [6, 12], "input_size": 2, "output_size": 1}
        proj1 = get_projection("MLP", layer_mode="complex", projection_config=config)
        proj2 = get_projection("MLP", layer_mode="complex", projection_config=config)

        # Modify proj1 weights to specific values
        for layer in proj1.mlp:
            if isinstance(layer, torch.nn.Linear):
                with torch.no_grad():
                    layer.weight.fill_(0.7)
                    layer.bias.fill_(0.3)

        # Save proj1 state
        state1 = proj1.state_dict()

        # Load state into proj2
        proj2.load_state_dict(state1)

        # Verify they now match
        for layer1, layer2 in zip(proj1.mlp, proj2.mlp):
            if isinstance(layer1, torch.nn.Linear) and isinstance(
                layer2, torch.nn.Linear
            ):
                assert torch.equal(layer1.weight, layer2.weight)
                assert torch.equal(layer1.bias, layer2.bias)
                assert torch.allclose(
                    layer2.weight, torch.full_like(layer2.weight, 0.7)
                )
                assert torch.allclose(layer2.bias, torch.full_like(layer2.bias, 0.3))

    def test_state_dict_keys(self):
        """Test that state dict has expected keys."""
        # Test polynomial
        poly_proj = get_projection(
            "polynomial", layer_mode="complex", projection_config={"order": 3}
        )
        poly_state = poly_proj.state_dict()
        expected_poly_keys = {"poly.weight", "poly.bias", "_exp_i", "_exp_j"}
        assert set(poly_state.keys()) == expected_poly_keys

        # Test MLP
        mlp_proj = get_projection(
            "MLP", layer_mode="complex", projection_config={"hidden_sizes": [4, 8]}
        )
        mlp_state = mlp_proj.state_dict()
        # Should have keys for all linear layers
        expected_mlp_keys = {
            "mlp.0.weight",
            "mlp.0.bias",  # 2->4
            "mlp.2.weight",
            "mlp.2.bias",  # 4->8
            "mlp.4.weight",
            "mlp.4.bias",  # 8->1
        }
        assert set(mlp_state.keys()) == expected_mlp_keys


class TestConfigUtilsProjection:
    """Test projection config utilities."""

    def test_get_model_params_includes_projection_config(self):
        """Test that get_model_params extracts projection config."""
        cfg = {
            "model": {"projection_layer": "polynomial", "projection": {"order": 7}},
            "data": {
                "patch_size": 64,
                "inferred_input_channels": 1,
            },
        }

        params = get_model_params(cfg)

        assert "projection_config" in params
        assert params["projection_config"] == {"order": 7}
        assert params["projection_layer"] == "polynomial"

    def test_get_model_params_empty_projection_config(self):
        """Test that get_model_params handles missing projection config."""
        cfg = {
            "model": {"projection_layer": "amplitude"},
            "data": {
                "patch_size": 64,
                "inferred_input_channels": 1,
            },
        }

        params = get_model_params(cfg)

        assert "projection_config" in params
        assert params["projection_config"] == {}
        assert params["projection_layer"] == "amplitude"

    def test_get_model_params_no_projection_layer(self):
        """Test that get_model_params handles missing projection_layer."""
        cfg = {
            "model": {},
            "data": {
                "patch_size": 64,
                "inferred_input_channels": 1,
            },
        }

        params = get_model_params(cfg)

        assert "projection_config" in params
        assert params["projection_config"] == {}
        assert params["projection_layer"] is None


class TestProjectionStateWorkflow:
    """Test the complete projection state management workflow."""

    def test_full_state_management_workflow(self):
        """Test the complete workflow: create, modify, save, restore."""
        # Step 1: Create projection with config
        config = {"order": 3}
        projection = get_projection(
            "polynomial", layer_mode="complex", projection_config=config
        )

        # Step 2: Set known initial values
        with torch.no_grad():
            projection.poly.weight.fill_(0.0)
            projection.poly.bias.fill_(0.0)

        # Step 3: Save this as our "initial" state (with cloning)
        initial_state = {k: v.clone() for k, v in projection.state_dict().items()}

        # Step 4: Modify the projection (simulate training)
        with torch.no_grad():
            projection.poly.weight.fill_(0.5)
            projection.poly.bias.fill_(0.1)

        # Step 5: Verify modification
        assert torch.allclose(
            projection.poly.weight, torch.full_like(projection.poly.weight, 0.5)
        )
        assert torch.allclose(
            projection.poly.bias, torch.full_like(projection.poly.bias, 0.1)
        )

        # Step 6: Save modified state (simulate checkpoint, with cloning)
        modified_state = {k: v.clone() for k, v in projection.state_dict().items()}

        # Step 7: Restore initial state (simulate loading checkpoint)
        projection.load_state_dict(initial_state)

        # Step 8: Verify restoration to initial state
        assert torch.allclose(
            projection.poly.weight, torch.full_like(projection.poly.weight, 0.0)
        )
        assert torch.allclose(
            projection.poly.bias, torch.full_like(projection.poly.bias, 0.0)
        )

        # Step 9: Load modified state again
        projection.load_state_dict(modified_state)

        # Step 10: Verify we're back to modified state
        assert torch.allclose(
            projection.poly.weight, torch.full_like(projection.poly.weight, 0.5)
        )
        assert torch.allclose(
            projection.poly.bias, torch.full_like(projection.poly.bias, 0.1)
        )

    def test_cross_instance_state_transfer(self):
        """Test transferring state between different projection instances."""
        config = {"hidden_sizes": [4, 8, 4]}

        # Create source projection and train it (modify weights)
        source_proj = get_projection(
            "MLP", layer_mode="complex", projection_config=config
        )
        for i, layer in enumerate(source_proj.mlp):
            if isinstance(layer, torch.nn.Linear):
                with torch.no_grad():
                    layer.weight.fill_(float(i + 1))
                    layer.bias.fill_(float(i + 0.5))

        # Create target projection (fresh initialization)
        target_proj = get_projection(
            "MLP", layer_mode="complex", projection_config=config
        )

        # Verify they're different initially
        source_weights = [
            layer.weight.clone()
            for layer in source_proj.mlp
            if isinstance(layer, torch.nn.Linear)
        ]
        target_weights = [
            layer.weight.clone()
            for layer in target_proj.mlp
            if isinstance(layer, torch.nn.Linear)
        ]

        # They should be different due to random initialization
        weights_differ = any(
            not torch.equal(s, t) for s, t in zip(source_weights, target_weights)
        )
        # Note: In rare cases they might be equal by chance, but that's extremely unlikely

        # Transfer state
        state = source_proj.state_dict()
        target_proj.load_state_dict(state)

        # Verify transfer worked
        for i, layer in enumerate(target_proj.mlp):
            if isinstance(layer, torch.nn.Linear):
                expected_weight_val = float(i + 1)
                expected_bias_val = float(i + 0.5)
                assert torch.allclose(
                    layer.weight, torch.full_like(layer.weight, expected_weight_val)
                )
                assert torch.allclose(
                    layer.bias, torch.full_like(layer.bias, expected_bias_val)
                )


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v"])
