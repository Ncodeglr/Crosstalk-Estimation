import torch
import torch.nn as nn
import torch.optim as optim
import pytest
from cvnn.callbacks import ModelCheckpoint


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, x):
        return self.linear(x)


def test_modelcheckpoint_default(tmp_path):
    model = DummyModel()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    # min_is_best=True (lower is better)
    ckpt = ModelCheckpoint(
        model,
        optimizer,
        tmp_path,
        num_input_dims=2,
        min_is_best=True,
        warmup_scheduler=None,
        scheduler=scheduler,
    )
    # first call should save
    assert ckpt.update(1.0, epoch=0)
    path = tmp_path / "best_model.pt"
    assert path.exists()
    state = torch.load(path)
    assert state["epoch"] == 0
    assert state["loss"] == 1.0
    assert "model_state_dict" in state
    assert "optimizer_state_dict" in state
    assert "scheduler_state_dict" in state

    # worse score should not save
    assert not ckpt.update(2.0, epoch=1)

    # better (lower) score should save
    assert ckpt.update(0.5, epoch=2)
    state2 = torch.load(path)
    assert state2["epoch"] == 2
    assert state2["loss"] == 0.5


def test_modelcheckpoint_higher_better(tmp_path):
    model = DummyModel()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    # min_is_best=False (higher is better)
    ckpt = ModelCheckpoint(
        model, optimizer, tmp_path, num_input_dims=2, min_is_best=False
    )

    # first call should save
    assert ckpt.update(0.1, epoch=0)
    path = tmp_path / "best_model.pt"
    assert path.exists()
    state = torch.load(path)
    assert state["loss"] == 0.1

    # lower score should not save
    assert not ckpt.update(0.05, epoch=1)

    # higher score should save
    assert ckpt.update(0.2, epoch=2)
    state2 = torch.load(path)
    assert state2["loss"] == 0.2
