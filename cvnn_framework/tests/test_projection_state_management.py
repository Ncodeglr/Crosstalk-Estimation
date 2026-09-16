"""
Test suite for projection state management functionality.

This module tests the robust state management for projection functions (polynomial and MLP)
to ensure correct saving/restoring during retrain and checkpointing, similar to the gumbel tau mechanism.
"""

import pytest
import torch
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch

# Add src to path for imports
import sys

sys.path.append("src")

from cvnn.models.projection import PolyCtoR, MLPCtoR
from cvnn.models.utils import get_projection
from cvnn.model_utils import build_unet
from cvnn.config_utils import get_model_params


class TestProjectionInstantiation:
    """Test projection layer instantiation with various configurations."""

    """
    Cleaned test suite for projection state management functionality.
    This file provides minimal configs required by the current baseline
    (explicit 'dropout' and 'inferred_input_channels').
    """

    import sys

    import pytest
    import torch

    # Add src to path for imports
    sys.path.append("src")

    from cvnn.models.projection import PolyCtoR, MLPCtoR
    from cvnn.models.utils import get_projection
    from cvnn.model_utils import build_unet
    from cvnn.config_utils import get_model_params

    class TestProjectionInstantiation:
        def test_polynomial_projection_default(self):
            proj = get_projection("polynomial", layer_mode="complex")
            assert isinstance(proj, PolyCtoR)
            assert proj.poly.in_features == 9
            assert proj.poly.out_features == 1

        def test_polynomial_projection_custom_order(self):
            config = {"order": 5}
            proj = get_projection(
                "polynomial", layer_mode="complex", projection_config=config
            )
            assert isinstance(proj, PolyCtoR)
            expected_features = ((5 + 1) * (5 + 2)) // 2 - 1
            assert proj.poly.in_features == expected_features

        def test_mlp_projection_default(self):
            proj = get_projection("MLP", layer_mode="complex")
            assert isinstance(proj, MLPCtoR)
            layers = list(proj.mlp.children())
            linear_layers = [l for l in layers if isinstance(l, torch.nn.Linear)]
            assert len(linear_layers) >= 3

        def test_mlp_projection_custom_architecture(self):
            config = {"hidden_sizes": [4, 8, 4], "input_size": 2, "output_size": 1}
            proj = get_projection("MLP", layer_mode="complex", projection_config=config)
            assert isinstance(proj, MLPCtoR)
            layers = list(proj.mlp.children())
            linear_layers = [l for l in layers if isinstance(l, torch.nn.Linear)]
            assert len(linear_layers) == 4

        def test_amplitude_projection(self):
            proj = get_projection("amplitude", layer_mode="complex")
            assert proj.__class__.__name__ == "Mod"

        def test_real_mode_returns_identity(self):
            for proj_type in ["polynomial", "MLP", "amplitude"]:
                proj = get_projection(proj_type, layer_mode="real")
                assert isinstance(proj, torch.nn.Identity)


    class TestConfigDrivenProjections:
        def test_config_driven_polynomial_projection(self):
            cfg = {
                "model": {
                    "num_layers": 2,
                    "channels_width": 8,
                    "activation": "modReLU",
                    "layer_mode": "complex",
                    "downsampling_layer": "LPD",
                    "projection_layer": "polynomial",
                    "projection": {"order": 4},
                    "num_classes": 2,
                    "dropout": 0.0,
                    "num_blocks": 1,
                },
                "data": {"patch_size": 64, "inferred_input_channels": 2},
            }
            model = build_unet(cfg)
            poly_projs = [
                m for _, m in model.named_modules() if isinstance(m, PolyCtoR)
            ]
            assert len(poly_projs) > 0
            expected_features = ((4 + 1) * (4 + 2)) // 2 - 1
            for p in poly_projs:
                assert p.poly.in_features == expected_features

        def test_config_driven_mlp_projection(self):
            cfg = {
                "model": {
                    "num_layers": 2,
                    "channels_width": 8,
                    "activation": "modReLU",
                    "layer_mode": "complex",
                    "downsampling_layer": "LPD",
                    "projection_layer": "MLP",
                    "projection": {
                        "hidden_sizes": [6, 12],
                        "input_size": 2,
                        "output_size": 1,
                    },
                    "num_classes": 2,
                    "dropout": 0.0,
                    "num_blocks": 1,
                },
                "data": {"patch_size": 64, "inferred_input_channels": 2},
            }
            model = build_unet(cfg)
            mlp_projs = [m for _, m in model.named_modules() if isinstance(m, MLPCtoR)]
            assert len(mlp_projs) > 0
            for p in mlp_projs:
                layers = list(p.mlp.children())
                linear_layers = [l for l in layers if isinstance(l, torch.nn.Linear)]
                assert len(linear_layers) >= 3
                assert linear_layers[0].in_features == 2
                assert linear_layers[0].out_features == 6
                assert linear_layers[1].in_features == 6
                assert linear_layers[1].out_features == 12

    class TestProjectionStateManagement:
        def test_polynomial_projection_state_roundtrip(self):
            cfg = {
                "model": {
                    "projection_layer": "polynomial",
                    "projection": {"order": 3},
                    "dropout": 0.0,
                    "num_classes": 1,
                    "layer_mode": "complex",
                    # Required sizing for UNet construction
                    "num_layers": 2,
                    "channels_width": 8,
                    "activation": "modReLU",
                    "num_blocks": 1,
                },
                "data": {
                    "inferred_input_channels": 2,
                    "inferred_input_size": 32,
                },
            }
            model1 = build_unet(cfg)
            model2 = build_unet(cfg)

            projection_states = {}
            for name, module in model1.named_modules():
                if isinstance(module, PolyCtoR):
                    with torch.no_grad():
                        module.poly.weight.fill_(0.5)
                        module.poly.bias.fill_(0.1)
                    projection_states[name] = module.state_dict()

            for name, module in model2.named_modules():
                if isinstance(module, PolyCtoR) and name in projection_states:
                    module.load_state_dict(projection_states[name])

            for name, module in model2.named_modules():
                if isinstance(module, PolyCtoR):
                    assert torch.allclose(
                        module.poly.weight, torch.full_like(module.poly.weight, 0.5)
                    )
                    assert torch.allclose(
                        module.poly.bias, torch.full_like(module.poly.bias, 0.1)
                    )

        def test_mlp_projection_state_management(self):
            cfg = {
                "model": {
                    "projection_layer": "MLP",
                    "projection": {"hidden_sizes": [4, 8]},
                    "dropout": 0.0,
                    "num_classes": 1,
                    "layer_mode": "complex",
                    # Required sizing for UNet construction
                    "num_layers": 2,
                    "channels_width": 8,
                    "activation": "modReLU",
                    "num_blocks": 1,
                },
                "data": {
                    "inferred_input_channels": 2,
                    "inferred_input_size": 32,
                },
            }
            model1 = build_unet(cfg)
            model2 = build_unet(cfg)

            for name, module in model1.named_modules():
                if isinstance(module, MLPCtoR):
                    for layer in module.mlp:
                        if isinstance(layer, torch.nn.Linear):
                            with torch.no_grad():
                                layer.weight.fill_(0.7)
                                layer.bias.fill_(0.3)

            projection_states = {
                name: m.state_dict()
                for name, m in model1.named_modules()
                if isinstance(m, MLPCtoR)
            }

            for name, module in model2.named_modules():
                if isinstance(module, MLPCtoR) and name in projection_states:
                    module.load_state_dict(projection_states[name])

            for name, module in model2.named_modules():
                if isinstance(module, MLPCtoR):
                    for layer in module.mlp:
                        if isinstance(layer, torch.nn.Linear):
                            assert torch.allclose(
                                layer.weight, torch.full_like(layer.weight, 0.7)
                            )
                            assert torch.allclose(
                                layer.bias, torch.full_like(layer.bias, 0.3)
                            )

    class TestProjectionConfigUtils:
        def test_get_model_params_includes_projection_config(self):
            cfg = {
                "model": {"projection_layer": "polynomial", "projection": {"order": 7}},
                "data": {"inferred_input_channels": 2},
            }
            params = get_model_params(cfg)
            assert "projection_config" in params
            assert params["projection_config"] == {"order": 7}
            assert params["projection_layer"] == "polynomial"

        def test_get_model_params_empty_projection_config(self):
            cfg = {
                "model": {"projection_layer": "amplitude"},
                "data": {"inferred_input_channels": 2},
            }
            params = get_model_params(cfg)
            assert "projection_config" in params
            assert params["projection_config"] == {}
            assert params["projection_layer"] == "amplitude"

    class TestProjectionIntegration:
        def test_full_workflow_polynomial(self):
            proj = get_projection(
                "polynomial", layer_mode="complex", projection_config={"order": 4}
            )
            assert isinstance(proj, PolyCtoR)
            expected_features = ((4 + 1) * (4 + 2)) // 2 - 1
            assert proj.poly.in_features == expected_features

            original_state = proj.state_dict()
            original_weight = original_state["poly.weight"].clone()
            original_bias = original_state["poly.bias"].clone()

            with torch.no_grad():
                proj.poly.weight.fill_(0.9)
                proj.poly.bias.fill_(0.2)

            modified_state = proj.state_dict()
            assert not torch.equal(original_weight, modified_state["poly.weight"])

            proj.load_state_dict(
                {"poly.weight": original_weight, "poly.bias": original_bias}, strict=False
            )
            restored_state = proj.state_dict()
            assert torch.equal(original_weight, restored_state["poly.weight"])
            assert torch.equal(original_bias, restored_state["poly.bias"])

        def test_full_workflow_mlp(self):
            proj = get_projection(
                "MLP",
                layer_mode="complex",
                projection_config={
                    "hidden_sizes": [6, 12, 6],
                    "input_size": 2,
                    "output_size": 1,
                },
            )
            assert isinstance(proj, MLPCtoR)
            layers = list(proj.mlp.children())
            linear_layers = [l for l in layers if isinstance(l, torch.nn.Linear)]
            assert len(linear_layers) == 4

            original_state = proj.state_dict()
            original_state_copy = {k: v.clone() for k, v in original_state.items()}

            for layer in proj.mlp:
                if isinstance(layer, torch.nn.Linear):
                    with torch.no_grad():
                        layer.weight.fill_(0.8)
                        layer.bias.fill_(0.4)

            modified = any(
                isinstance(layer, torch.nn.Linear)
                and torch.all(layer.weight == 0.8)
                and torch.all(layer.bias == 0.4)
                for layer in proj.mlp
            )
            assert modified

            proj.load_state_dict(original_state_copy)
            restored_correctly = any(
                isinstance(layer, torch.nn.Linear)
                and not torch.all(layer.weight == 0.8)
                for layer in proj.mlp
            )
            assert restored_correctly

    if __name__ == "__main__":
        pytest.main([__file__, "-v"])
    """Test projection config utilities."""

    def test_get_model_params_includes_projection_config(self):
        """Test that get_model_params extracts projection config."""
        cfg = {
            "model": {"projection_layer": "polynomial", "projection": {"order": 7}},
            "data": {
                "patch_size": 64,
                "inferred_input_channels": 2,
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
                "inferred_input_channels": 2,
            },
        }

        params = get_model_params(cfg)

        assert "projection_config" in params
        assert params["projection_config"] == {}
        assert params["projection_layer"] == "amplitude"


# Integration test for the complete workflow
class TestProjectionIntegration:
    """Integration tests for the complete projection state management workflow."""

    def test_full_workflow_polynomial(self):
        """Test the complete workflow with polynomial projections."""
        # This test simulates the full model creation, state extraction, and restoration workflow
        config = {
            "model": {"projection_layer": "polynomial", "projection": {"order": 4}}
        }

        # Test that config is correctly propagated through the system
        proj_config = config["model"]["projection"]
        proj = get_projection(
            "polynomial", layer_mode="complex", projection_config=proj_config
        )

        assert isinstance(proj, PolyCtoR)
        expected_features = ((4 + 1) * (4 + 2)) // 2 - 1
        assert proj.poly.in_features == expected_features

        # Test state dict operations
        original_state = proj.state_dict()
        # Clone the original state to avoid reference issues
        original_weight = original_state["poly.weight"].clone()
        original_bias = original_state["poly.bias"].clone()

        # Modify weights
        with torch.no_grad():
            proj.poly.weight.fill_(0.9)
            proj.poly.bias.fill_(0.2)

        modified_state = proj.state_dict()
        assert not torch.equal(original_weight, modified_state["poly.weight"])

        # Restore original state
        proj.load_state_dict(
            {"poly.weight": original_weight, "poly.bias": original_bias}, strict=False
        )
        restored_state = proj.state_dict()

        assert torch.equal(original_weight, restored_state["poly.weight"])
        assert torch.equal(original_bias, restored_state["poly.bias"])

    def test_full_workflow_mlp(self):
        """Test the complete workflow with MLP projections."""
        config = {"hidden_sizes": [6, 12, 6], "input_size": 2, "output_size": 1}

        proj = get_projection("MLP", layer_mode="complex", projection_config=config)

        assert isinstance(proj, MLPCtoR)

        # Check architecture
        layers = list(proj.mlp.children())
        linear_layers = [
            layer for layer in layers if isinstance(layer, torch.nn.Linear)
        ]
        assert len(linear_layers) == 4  # 2->6, 6->12, 12->6, 6->1

        # Test state dict operations
        original_state = proj.state_dict()
        # Deep copy the original state to avoid reference issues
        original_state_copy = {k: v.clone() for k, v in original_state.items()}

        # Modify weights
        for layer in proj.mlp:
            if isinstance(layer, torch.nn.Linear):
                with torch.no_grad():
                    layer.weight.fill_(0.8)
                    layer.bias.fill_(0.4)

        # Verify modification worked
        modified = False
        for layer in proj.mlp:
            if isinstance(layer, torch.nn.Linear):
                if torch.all(layer.weight == 0.8) and torch.all(layer.bias == 0.4):
                    modified = True
                    break
        assert modified, "Weight modification failed"

        # Restore original state
        proj.load_state_dict(original_state_copy)

        # Verify restoration (check that at least one weight is different from 0.8)
        restored_correctly = False
        for layer in proj.mlp:
            if isinstance(layer, torch.nn.Linear):
                if not torch.all(layer.weight == 0.8):
                    restored_correctly = True
                    break

        assert restored_correctly, "State restoration failed"


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v"])
