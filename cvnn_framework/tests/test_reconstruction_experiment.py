"""
Test the ReconstructionExperiment class and related functionality.
"""

import pytest
import tempfile
import yaml
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch

# Import ReconstructionExperiment from the projects module
import sys

# Add repository root to sys.path so tests can import the 'projects' package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

from projects.reconstruction.experiment import ReconstructionExperiment


def test_reconstruction_experiment_registration():
    """Test that ReconstructionExperiment is properly registered as a plugin."""
    from cvnn.plugins import get_plugins

    plugins = get_plugins()
    assert "reconstruction" in plugins
    assert plugins["reconstruction"] == ReconstructionExperiment


def test_reconstruction_experiment_init(tmp_path, monkeypatch):
    """Test ReconstructionExperiment initialization with minimal config."""

    # Mock all the problematic imports and functions at the module level
    import cvnn.data as data_mod
    from torch.utils.data import Dataset

    class MockDataset(Dataset):
        def __init__(self):
            # Add required attributes that real datasets have
            self.nsamples_per_cols = 64  # Mock value
            self.nsamples_per_rows = 64  # Mock value
            # For PolSFDataset compatibility - add alos_dataset attribute
            self.alos_dataset = self

        def __len__(self):
            return 10

        def __getitem__(self, idx):
            return torch.randn(3, 32, 32, dtype=torch.complex64), torch.tensor(0)

    # Mock dataset classes and file system functions completely
    def mock_dataset_constructor(**kwargs):
        return MockDataset()

    monkeypatch.setattr(data_mod, "ALOSDataset", mock_dataset_constructor)
    monkeypatch.setattr(data_mod, "PolSFDataset", mock_dataset_constructor)
    monkeypatch.setattr(data_mod, "Bretigny", mock_dataset_constructor)

    # Mock the _find_volpath function to return a fake path
    def mock_find_volpath(base_path, vol_folder):
        return tmp_path / "fake_vol"

    monkeypatch.setattr(data_mod, "_find_volpath", mock_find_volpath)

    # Mock the entire get_dataloaders function to avoid any file system access
    def mock_get_dataloaders(cfg, use_cuda):
        mock_train = MockDataset()
        mock_valid = MockDataset()
        mock_test = MockDataset()
        from torch.utils.data import DataLoader

        return (
            DataLoader(mock_train, batch_size=4),
            DataLoader(mock_valid, batch_size=4),
            DataLoader(mock_test, batch_size=4),
        )

    def mock_get_full_image_dataloader(cfg, use_cuda):
        from torch.utils.data import DataLoader

        return DataLoader(MockDataset(), batch_size=1)

    monkeypatch.setattr(data_mod, "get_dataloaders", mock_get_dataloaders)
    monkeypatch.setattr(
        data_mod, "get_full_image_dataloader", mock_get_full_image_dataloader
    )

    # Create a minimal config for testing
    config_data = {
        "task": "reconstruction",
        "data": {
            "supports_full_image_reconstruction": True,
            "batch_size": 4,
            "num_channels": 3,
            "patch_size": 32,
            "patch_stride": 32,
            "dataset": {
                "name": "ALOSDataset",
                "trainpath": str(tmp_path),
            },  # Use tmp_path
            "crop": {"start_row": 0, "start_col": 0, "end_row": 256, "end_col": 256},
            "test_ratio": 0.1,
            "valid_ratio": 0.1,
            "num_workers": 0,
        },
        "model": {
            "activation": "modReLU",
            "channels_width": 8,
            "layer_mode": "complex",
            "dropout": 0.0,
            "class": "AutoEncoder",
            "conv_mode": "complex",
            "upsampling_layer": "bilinear",
            "num_layers": 2,
            "num_blocks": 1,
        },
        "logging": {"logdir": str(tmp_path / "logs")},
        "nepochs": 1,
        "optim": {"algo": "AdamW", "params": {"lr": 0.001}},
        "loss": {"name": "MSE"},
    }

    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    # Mock other dependencies that would require actual CUDA/training setup
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("wandb.init"),
    ):
        # Initialize the experiment
        exp = ReconstructionExperiment(str(config_path))

        # Verify basic properties
        assert exp.cfg["task"] == "reconstruction"
        assert exp.model is not None
        assert exp.train_loader is not None
        assert exp.valid_loader is not None
        assert exp.test_loader is not None
        assert exp.full_loader is not None


def test_reconstruction_experiment_build_model(tmp_path, monkeypatch):
    """Test that ReconstructionExperiment builds the correct model."""

    # Mock all the problematic imports and functions at the module level
    import cvnn.data as data_mod
    from torch.utils.data import Dataset

    class MockDataset(Dataset):
        def __init__(self):
            # Add required attributes that real datasets have
            self.nsamples_per_cols = 64  # Mock value
            self.nsamples_per_rows = 64  # Mock value
            # For PolSFDataset compatibility - add alos_dataset attribute
            self.alos_dataset = self

        def __len__(self):
            return 10

        def __getitem__(self, idx):
            return torch.randn(3, 64, 64, dtype=torch.complex64), torch.tensor(0)

    # Mock dataset classes and file system functions completely
    def mock_dataset_constructor(**kwargs):
        return MockDataset()

    monkeypatch.setattr(data_mod, "ALOSDataset", mock_dataset_constructor)
    monkeypatch.setattr(data_mod, "PolSFDataset", mock_dataset_constructor)
    monkeypatch.setattr(data_mod, "Bretigny", mock_dataset_constructor)

    # Mock the _find_volpath function to return a fake path
    def mock_find_volpath(base_path, vol_folder):
        return tmp_path / "fake_vol"

    monkeypatch.setattr(data_mod, "_find_volpath", mock_find_volpath)

    # Mock the entire get_dataloaders function to avoid any file system access
    def mock_get_dataloaders(cfg, use_cuda):
        mock_train = MockDataset()
        mock_valid = MockDataset()
        mock_test = MockDataset()
        from torch.utils.data import DataLoader

        return (
            DataLoader(mock_train, batch_size=4),
            DataLoader(mock_valid, batch_size=4),
            DataLoader(mock_test, batch_size=4),
        )

    def mock_get_full_image_dataloader(cfg, use_cuda):
        from torch.utils.data import DataLoader

        return DataLoader(MockDataset(), batch_size=1)

    monkeypatch.setattr(data_mod, "get_dataloaders", mock_get_dataloaders)
    monkeypatch.setattr(
        data_mod, "get_full_image_dataloader", mock_get_full_image_dataloader
    )

    config_data = {
        "task": "reconstruction",
        "data": {
            "supports_full_image_reconstruction": True,
            "num_channels": 3,
            "patch_size": 64,
            "patch_stride": 32,  # Add missing patch_stride field
            "batch_size": 4,
            "dataset": {"name": "ALOSDataset", "trainpath": str(tmp_path)},
            "crop": {"start_row": 0, "start_col": 0, "end_row": 256, "end_col": 256},
            "test_ratio": 0.1,
            "valid_ratio": 0.1,
            "num_workers": 0,
        },
        "model": {
            "activation": "modReLU",
            "channels_width": 16,
            "layer_mode": "complex",
            "dropout": 0.0,
            "conv_mode": "complex",
            "num_layers": 4,
            "class": "AutoEncoder",
            "upsampling_layer": "bilinear",
            "num_blocks": 1,
        },
        "logging": {"logdir": str(tmp_path / "logs")},
        "nepochs": 1,
        "optim": {"algo": "AdamW", "params": {"lr": 0.001}},
        "loss": {"name": "ComplexMSELoss"},
        "loss": {"name": "MSE"},
    }

    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("wandb.init"),
    ):
        exp = ReconstructionExperiment(str(config_path))
        model = exp.build_model()

        # Verify it's an AutoEncoder instance
        from cvnn.models.models import AutoEncoder

        assert isinstance(model, AutoEncoder)
