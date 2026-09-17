import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

from cvnn.train import train_one_epoch, train_model


def get_dummy_loader(batch_size=1, num_batches=1):
    # create dummy data: input and target same
    data = torch.zeros((batch_size * num_batches, 1, 1))
    ds = TensorDataset(data)
    return DataLoader(ds, batch_size=batch_size)


def test_warmup_lambda_scheduler_progression_direct():
    # Test LambdaLR warmup behavior directly
    optimizer = torch.optim.SGD(nn.Linear(1, 1).parameters(), lr=10.0)
    # initial lr
    assert optimizer.param_groups[0]["lr"] == 10.0
    warmup_epochs = 2
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda e: min(1.0, float(e + 1) / warmup_epochs)
    )
    print("Initial LR after scheduler creation:", optimizer.param_groups[0]["lr"])
    # first step: lr = 10 * (1/2)
    assert optimizer.param_groups[0]["lr"] == 5.0
    # second step: lr = 10 * (2/2)
    warmup_scheduler.step()
    assert pytest.approx(10.0, rel=1e-3) == optimizer.param_groups[0]["lr"]
    # third step: should remain at 10.0
    warmup_scheduler.step()
    assert pytest.approx(10.0, rel=1e-3) == optimizer.param_groups[0]["lr"]
    # fourth step: should remain at 10.0
    warmup_scheduler.step()
    assert pytest.approx(10.0, rel=1e-3) == optimizer.param_groups[0]["lr"]


def test_step_lr_scheduler_epoch_stepping(tmp_path):
    # Model with one parameter
    model = nn.Linear(1, 1)
    # prepare single-element loader
    train_loader = get_dummy_loader(batch_size=1, num_batches=1)
    valid_loader = get_dummy_loader(batch_size=1, num_batches=1)
    # config for StepLR
    cfg = {
        "nepochs": 3,
        "loss": {"name": "MSELoss"},
        "optim": {"algo": "SGD", "params": {"lr": 0.1}},
        "scheduler": {
            "name": "StepLR",
            "params": {"step_size": 1, "gamma": 0.1},
            "step_on": "epoch",
        },
    }
    # pass optimizer externally
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    # pass explicit loss_fn to match train_model's expectations
    history = train_model(
        model,
        train_loader,
        valid_loader,
        cfg,
        logdir=tmp_path,
        device=torch.device("cpu"),
        loss_fn=nn.MSELoss(),
        optimizer=optimizer,
    )
    # after 3 epochs, lr reduced 3 times: 0.1 * 0.1^3
    final_lr = optimizer.param_groups[0]["lr"]
    assert pytest.approx(0.1 * (0.1**3), rel=1e-3) == final_lr
    # history length matches nepochs
    assert len(history["train_loss"]) == 3
    assert len(history["valid_loss"]) == 3
