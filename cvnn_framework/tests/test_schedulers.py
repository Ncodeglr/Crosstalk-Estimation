import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import SGD
from torch.optim.lr_scheduler import (
    StepLR,
    ExponentialLR,
    MultiStepLR,
    CosineAnnealingLR,
)
from pathlib import Path

from cvnn.train import train_model


def get_loader(num_samples=10, batch_size=5):
    data = torch.zeros((num_samples, 1, 1))
    ds = TensorDataset(data)
    return DataLoader(ds, batch_size=batch_size)


@pytest.mark.parametrize(
    "sched_name, sched_params, nepochs, expected_factor",
    [
        ("StepLR", {"step_size": 1, "gamma": 0.5}, 4, 0.5**4),
        ("ExponentialLR", {"gamma": 0.9}, 3, 0.9**3),
        ("MultiStepLR", {"milestones": [1, 3], "gamma": 0.1}, 4, 0.1**2),
        ("CosineAnnealingLR", {"T_max": 4}, 4, 0.0),
    ],
)
def test_scheduler_epoch_stepping(
    tmp_path, sched_name, sched_params, nepochs, expected_factor
):
    initial_lr = 0.1
    model = nn.Linear(1, 1)
    optimizer = SGD(model.parameters(), lr=initial_lr)
    cfg = {
        "nepochs": nepochs,
        "loss": {"name": "MSELoss"},
        "optim": {"algo": "SGD", "params": {"lr": initial_lr}},
        "scheduler": {"name": sched_name, "params": sched_params, "step_on": "epoch"},
    }
    train_loader = get_loader()
    valid_loader = get_loader()

    train_model(
        model,
        train_loader,
        valid_loader,
        cfg,
        logdir=tmp_path,
        device=torch.device("cpu"),
        loss_fn=nn.MSELoss(),
        optimizer=optimizer,
    )
    final_lr = optimizer.param_groups[0]["lr"]
    assert pytest.approx(initial_lr * expected_factor, rel=1e-3) == final_lr


def test_scheduler_batch_stepping(tmp_path):
    # StepLR with batch-level stepping over 4 batches
    initial_lr = 0.2
    step_size = 2
    gamma = 0.5
    model = nn.Linear(1, 1)
    optimizer = SGD(model.parameters(), lr=initial_lr)
    cfg = {
        "nepochs": 1,
        "loss": {"name": "MSELoss"},
        "optim": {"algo": "SGD", "params": {"lr": initial_lr}},
        "scheduler": {
            "name": "StepLR",
            "params": {"step_size": step_size, "gamma": gamma},
            "step_on": "batch",
        },
    }
    # 4 samples, batch_size=1 => 4 batches
    train_loader = get_loader(num_samples=4, batch_size=1)
    valid_loader = get_loader(num_samples=4, batch_size=1)

    train_model(
        model,
        train_loader,
        valid_loader,
        cfg,
        logdir=tmp_path,
        device=torch.device("cpu"),
        loss_fn=nn.MSELoss(),
        optimizer=optimizer,
    )
    # Expect lr to drop every 'step_size' batches: 4 batches, step_size=2 -> 2 drops
    expected_lr = initial_lr * (gamma**2)
    final_lr = optimizer.param_groups[0]["lr"]
    assert pytest.approx(expected_lr, rel=1e-3) == final_lr
