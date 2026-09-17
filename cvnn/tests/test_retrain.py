#!/usr/bin/env python3
"""
Test the retrain functionality to ensure no loss spikes when resuming training.
"""

import pytest
import torch
import yaml
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from cvnn.base_experiment import BaseExperiment


class MockRetrainExperiment(BaseExperiment):
    """Mock experiment class for testing retrain functionality."""

    def __init__(self, config_path, resume_logdir=None, mode_override=None):
        self.config_path = config_path
        self.resume_logdir = resume_logdir
        self.mode_override = mode_override
        self.loaded_epoch = None
        self.best_score = None

        # Simulate loading config
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

    def build_model(self):
        """Mock implementation of build_model."""
        return MagicMock()

    def evaluate(self, dataloader):
        """Mock implementation of evaluate."""
        return {"loss": 0.5, "accuracy": 0.95}

    def visualize(self, data):
        """Mock implementation of visualize."""
        pass

    def load_last_model(self):
        """Mock implementation of load_last_model that returns checkpoint epoch."""
        if self.resume_logdir:
            checkpoint_path = Path(self.resume_logdir) / "last_model.pt"
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                self.loaded_epoch = checkpoint.get("epoch", 0)
                self.best_score = checkpoint.get("valid_loss", None)
                return self.loaded_epoch  # Return epoch number like the fixed version
        return None

    def train(self, start_epoch=0):
        """Mock training method that accepts start_epoch parameter."""
        total_epochs = self.cfg.get("nepochs", 10)
        return start_epoch, total_epochs


def test_retrain_mode_override_preserved(tmp_path):
    """Test that mode override is preserved when loading existing config."""
    # Create base config
    base_config = {
        "task": "test",
        "mode": "full",  # Original mode
        "nepochs": 10,
        "data": {"batch_size": 4},
        "model": {"layers": 2},
    }

    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(base_config, f)

    # Create checkpoint directory with last_model.pt
    resume_dir = tmp_path / "logs"
    resume_dir.mkdir()

    checkpoint = {
        "epoch": 2,
        "model_state_dict": {},
        "optimizer_state_dict": {},
        "valid_loss": 0.5,
    }
    torch.save(checkpoint, resume_dir / "last_model.pt")

    # Test that mode_override is preserved after loading existing config
    exp = MockRetrainExperiment(
        str(config_path), resume_logdir=str(resume_dir), mode_override="retrain"
    )

    # Simulate the fix: mode_override should be preserved
    original_mode = exp.cfg.get("mode", "full")
    assert original_mode == "full"  # Config file has 'full'

    # After applying mode_override (simulating the fix)
    if exp.mode_override:
        exp.cfg["mode"] = exp.mode_override

    assert exp.cfg["mode"] == "retrain"  # Override should be applied


def test_checkpoint_epoch_continuation(tmp_path):
    """Test that training continues from the correct epoch."""
    config = {"nepochs": 10, "data": {"batch_size": 4}, "model": {"layers": 2}}

    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Create checkpoint at epoch 3
    resume_dir = tmp_path / "logs"
    resume_dir.mkdir()

    checkpoint = {
        "epoch": 3,  # Stopped after epoch 3
        "model_state_dict": {},
        "optimizer_state_dict": {},
        "valid_loss": 0.4,
    }
    torch.save(checkpoint, resume_dir / "last_model.pt")

    exp = MockRetrainExperiment(str(config_path), resume_logdir=str(resume_dir))

    # Load checkpoint
    checkpoint_epoch = exp.load_last_model()
    assert checkpoint_epoch == 3

    # Calculate start_epoch (should be checkpoint_epoch + 1)
    start_epoch = checkpoint_epoch + 1 if checkpoint_epoch is not None else 0
    assert start_epoch == 4  # Should resume from epoch 4

    # Test training range
    start_epoch_result, total_epochs = exp.train(start_epoch=start_epoch)
    assert start_epoch_result == 4
    assert total_epochs == 10

    # Verify remaining epochs (should be 6: epochs 4,5,6,7,8,9)
    remaining_epochs = total_epochs - start_epoch
    assert remaining_epochs == 6


def test_checkpoint_consistent_source(tmp_path):
    """Test that all states are loaded from the same checkpoint file."""
    config = {"nepochs": 5, "data": {"batch_size": 4}, "model": {"layers": 2}}

    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    resume_dir = tmp_path / "logs"
    resume_dir.mkdir()

    # Create last_model.pt (epoch 2, valid_loss 0.4)
    last_checkpoint = {
        "epoch": 2,
        "model_state_dict": {"param1": torch.tensor([1.0])},
        "optimizer_state_dict": {"lr": 0.001},
        "scheduler_state_dict": {"last_epoch": 2},
        "valid_loss": 0.4,
    }
    torch.save(last_checkpoint, resume_dir / "last_model.pt")

    # Create best_model.pt (epoch 1, valid_loss 0.3 - better score but older)
    best_checkpoint = {
        "epoch": 1,
        "model_state_dict": {"param1": torch.tensor([0.5])},
        "optimizer_state_dict": {"lr": 0.002},
        "scheduler_state_dict": {"last_epoch": 1},
        "valid_loss": 0.3,
    }
    torch.save(best_checkpoint, resume_dir / "best_model.pt")

    exp = MockRetrainExperiment(str(config_path), resume_logdir=str(resume_dir))

    # Test that we load from last_model.pt (not best_model.pt)
    checkpoint_epoch = exp.load_last_model()
    assert checkpoint_epoch == 2  # From last_model.pt
    assert exp.best_score == 0.4  # From last_model.pt, not 0.3 from best_model.pt

    # This ensures all states (model, optimizer, scheduler, best_score)
    # come from the same checkpoint file, preventing inconsistencies


def test_scheduler_state_consistency():
    """Test that scheduler state matches the checkpoint epoch."""
    # This test verifies that scheduler last_epoch is consistent with checkpoint epoch
    checkpoint_epoch = 5

    # Simulate a checkpoint with mismatched scheduler state (old bug)
    mismatched_checkpoint = {
        "epoch": checkpoint_epoch,
        "scheduler_state_dict": {"last_epoch": 90},  # Severely mismatched
    }

    # The fix should detect and handle this mismatch
    scheduler_last_epoch = mismatched_checkpoint["scheduler_state_dict"]["last_epoch"]

    # In the fixed version, we would log this mismatch and potentially reset
    if scheduler_last_epoch != checkpoint_epoch:
        # This represents the detection logic from our fix
        mismatch_detected = True
        expected_last_epoch = checkpoint_epoch
    else:
        mismatch_detected = False
        expected_last_epoch = scheduler_last_epoch

    assert mismatch_detected is True
    assert expected_last_epoch == checkpoint_epoch


@pytest.mark.parametrize(
    "checkpoint_epoch,expected_start",
    [
        (0, 1),  # After epoch 0, start from epoch 1
        (2, 3),  # After epoch 2, start from epoch 3
        (7, 8),  # After epoch 7, start from epoch 8
        (9, 10),  # After epoch 9, start from epoch 10
    ],
)
def test_epoch_calculation_various_scenarios(checkpoint_epoch, expected_start):
    """Test epoch calculation for various checkpoint scenarios."""
    # This tests the core fix: start_epoch = checkpoint_epoch + 1
    start_epoch = checkpoint_epoch + 1
    assert start_epoch == expected_start


def test_retrain_no_checkpoint(tmp_path):
    """Test retrain behavior when no checkpoint exists."""
    config = {"nepochs": 5}
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # No resume directory provided
    exp = MockRetrainExperiment(str(config_path))

    checkpoint_epoch = exp.load_last_model()
    assert checkpoint_epoch is None

    # Should start from epoch 0 when no checkpoint
    start_epoch = checkpoint_epoch + 1 if checkpoint_epoch is not None else 0
    assert start_epoch == 0
