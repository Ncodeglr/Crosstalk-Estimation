import pytest
from cvnn.config_utils import (
    get_model_params,
    update_config_with_inferred_values,
    validate_required_config_sections,
)


def test_get_model_params_minimal():
    cfg = {
        "data": {"inferred_input_channels": 3, "inferred_input_size": 32},
        "model": {"num_layers": 2, "channels_width": 8, "activation": "ReLU"},
    }
    params = get_model_params(cfg)
    assert params["num_channels"] == 3
    assert params["input_size"] == 32
    assert params["num_layers"] == 2


def test_update_config_with_inferred_values_adds_keys():
    cfg = {}
    update_config_with_inferred_values(
        cfg, inferred_input_channels=4, inferred_num_classes=10
    )
    assert cfg["data"]["inferred_input_channels"] == 4
    assert cfg["model"]["inferred_num_classes"] == 10


def test_validate_required_config_sections_raises():
    with pytest.raises(ValueError):
        validate_required_config_sections({})
