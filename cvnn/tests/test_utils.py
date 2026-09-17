import os
import random
import numpy as np
import torch
import logging
import pytest

from cvnn.utils import set_seed, setup_logging
from cvnn.config import load_config


def test_set_seed_reproducibility():
    # Set a seed and generate a few random numbers
    set_seed(123)
    vals1 = [random.random(), np.random.rand(), torch.rand(1).item()]
    # Reset seed and generate again
    set_seed(123)
    vals2 = [random.random(), np.random.rand(), torch.rand(1).item()]
    assert vals1 == pytest.approx(vals2)


def test_setup_logging_default_level(caplog):
    caplog.set_level(logging.DEBUG)
    logger = setup_logging("test_logger", level=logging.DEBUG)
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger"
    # Logging a message should appear in records
    logger.debug("debug message")
    assert any("debug message" in rec.message for rec in caplog.records)


def test_load_config_existing_file(tmp_path):
    # Create a test config file
    import yaml

    test_config = {
        "data": {"batch_size": 16},
        "model": {"layers": 3},
        "logging": {"level": "INFO"},
    }
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(test_config, f)

    cfg = load_config(str(config_file))
    assert isinstance(cfg, dict)
    assert cfg["data"]["batch_size"] == 16
    assert cfg["model"]["layers"] == 3


def test_load_config_not_found():
    with pytest.raises(FileNotFoundError):
        # Non-existent path should raise FileNotFoundError specifically
        load_config("nonexistent_config.yaml")


def test_load_config_invalid_yaml(tmp_path):
    # Test with invalid YAML content
    invalid_config_file = tmp_path / "invalid.yaml"
    invalid_config_file.write_text("invalid: yaml: content: [")  # Invalid YAML

    with pytest.raises(Exception):  # Could be yaml.YAMLError or other parsing error
        load_config(str(invalid_config_file))
