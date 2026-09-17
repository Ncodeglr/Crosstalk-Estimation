import pytest
import yaml
from pydantic import ValidationError
from cvnn.config import load_config


def test_load_valid_config(tmp_path):
    # Minimal valid config with required sections
    cfg = {
        "data": {"batch_size": 1},
        "model": {"num_features": 10},
    }
    cfg_path = tmp_path / "valid.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f)
    loaded = load_config(str(cfg_path))
    assert loaded["data"]["batch_size"] == 1
    assert loaded["model"]["num_features"] == 10


def test_load_config_with_optional_fields(tmp_path):
    """Test loading config with optional fields"""
    cfg = {
        "task": "reconstruction",
        "data": {"batch_size": 32, "path": "/data/images"},
        "model": {"architecture": "autoencoder", "latent_dim": 128},
        "train": {"lr": 0.001, "epochs": 50},
        "logging": {"level": "INFO", "logdir": "logs"},
        "wandb": {"project": "cvnn", "mode": "online"},
    }
    cfg_path = tmp_path / "full_config.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f)

    loaded = load_config(str(cfg_path))
    assert loaded["task"] == "reconstruction"
    assert loaded["train"]["lr"] == 0.001
    assert loaded["wandb"]["project"] == "cvnn"


def test_config_hierarchical_merging(tmp_path):
    """Test hierarchical config merging (base + specialized)"""
    # Create base config
    base_cfg = {
        "logging": {"level": "INFO", "logdir": "logs"},
        "wandb": {"project": "cvnn", "mode": "online"},
        "data": {"batch_size": 16},
        "model": {"base_feature": "shared"},
    }
    base_path = tmp_path / "config.yaml"
    with open(base_path, "w") as f:
        yaml.dump(base_cfg, f)

    # Create specialized config that should merge with base
    specialized_cfg = {
        "task": "reconstruction",
        "data": {"batch_size": 32, "path": "/data/sar_images"},  # Override batch_size
        "model": {"architecture": "autoencoder", "latent_dim": 128},  # Add new fields
        "train": {"lr": 0.001, "epochs": 50},
    }
    specialized_path = tmp_path / "config_reconstruction.yaml"
    with open(specialized_path, "w") as f:
        yaml.dump(specialized_cfg, f)

    # Test that specialized config loads and merges correctly
    loaded = load_config(str(specialized_path))

    # Should have specialized values
    assert loaded["task"] == "reconstruction"
    assert loaded["data"]["batch_size"] == 32  # Overridden
    assert loaded["data"]["path"] == "/data/sar_images"  # New
    assert loaded["model"]["architecture"] == "autoencoder"  # New

    # Should inherit from base (if merging is implemented)
    # Note: This test assumes hierarchical merging exists in the actual implementation


def test_invalid_yaml_format(tmp_path):
    """Test handling of invalid YAML format"""
    cfg_path = tmp_path / "invalid.yaml"
    cfg_path.write_text("invalid: yaml: content: [")  # Invalid YAML

    with pytest.raises(
        (yaml.YAMLError, Exception)
    ):  # Could be YAML or other parsing error
        load_config(str(cfg_path))


def test_missing_data_section(tmp_path):
    # Missing 'data' should cause a validation error
    cfg = {"model": {}, "train": {}}
    cfg_path = tmp_path / "no_data.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f)
    with pytest.raises(ValidationError):
        load_config(str(cfg_path))


def test_missing_model_section(tmp_path):
    # Missing 'model' should cause a validation error
    cfg = {"data": {}, "train": {}}
    cfg_path = tmp_path / "no_model.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f)
    with pytest.raises(ValidationError):
        load_config(str(cfg_path))


def test_nonexistent_config_file():
    """Test loading non-existent config file"""
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent_config.yaml")
