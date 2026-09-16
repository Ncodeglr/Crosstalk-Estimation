import torch
import pytest
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import LambdaLR, StepLR
from pathlib import Path

from cvnn.train import train_model


def get_dummy_loader():
    # create a single batch of zeros as both input and target
    data = torch.zeros((1, 1, 1))
    ds = TensorDataset(data)
    return DataLoader(ds, batch_size=1)


def test_resume_checkpoint_state(tmp_path):
    # Configuration for a single epoch training with warmup and scheduler
    cfg = {
        "nepochs": 1,
        "loss": {"name": "MSELoss"},
        "optim": {"algo": "SGD", "params": {"lr": 0.1}},
        "warmup": {"epochs": 1},
        "scheduler": {
            "name": "StepLR",
            "params": {"step_size": 1, "gamma": 0.5},
            "step_on": "epoch",
        },
    }

    # Build model, optimizer and schedulers
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    warmup = LambdaLR(
        optimizer, lr_lambda=lambda e: min(1.0, float(e + 1) / cfg["warmup"]["epochs"])
    )
    scheduler = StepLR(optimizer, step_size=1, gamma=0.5)

    # Prepare dummy loaders
    train_loader = get_dummy_loader()
    valid_loader = get_dummy_loader()

    # Use pytest's tmp_path fixture instead of hardcoded path
    logdir = tmp_path / "logs"
    logdir.mkdir(exist_ok=True)

    # Run training which will save a checkpoint in logdir
    train_model(
        model,
        train_loader,
        valid_loader,
        cfg,
        logdir=logdir,
        device=torch.device("cpu"),
        loss_fn=torch.nn.MSELoss(),
        optimizer=optimizer,
        warmup_scheduler=warmup,
        scheduler=scheduler,
    )

    # Load the last checkpoint
    ckpt_path = logdir / "last_model.pt"
    assert ckpt_path.exists(), "Checkpoint file was not created"
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Model state check
    model_sd = model.state_dict()
    for key, val in model_sd.items():
        assert (
            key in ckpt["model_state_dict"]
        ), f"Missing {key} in checkpoint model state"
        assert torch.equal(
            val, ckpt["model_state_dict"][key]
        ), f"Model state mismatch for {key}"

    # Restore and check model state
    loaded_model = nn.Linear(1, 1)
    loaded_model.load_state_dict(ckpt["model_state_dict"])
    for key, val in ckpt["model_state_dict"].items():
        assert torch.equal(
            loaded_model.state_dict()[key], val
        ), f"Loaded model state mismatch for {key}"

    # Restore and check optimizer state
    loaded_optimizer = torch.optim.SGD(loaded_model.parameters(), lr=0.1)
    loaded_optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    # Verify optimizer param_groups and lr restored correctly
    orig_pg = ckpt["optimizer_state_dict"]["param_groups"][0]
    loaded_pg = loaded_optimizer.state_dict()["param_groups"][0]
    # Check learning rate matches checkpoint
    assert (
        pytest.approx(loaded_pg["lr"], rel=1e-6) == orig_pg["lr"]
    ), "Optimizer lr mismatch in loaded state"
    # Ensure other hyperparams are restored
    for key in [k for k in orig_pg.keys() if k != "lr"]:
        assert (
            loaded_pg[key] == orig_pg[key]
        ), f"Optimizer param_groups key '{key}' mismatch"

    # Restore and check warmup state
    loaded_warmup = LambdaLR(
        loaded_optimizer,
        lr_lambda=lambda e: min(1.0, float(e + 1) / cfg["warmup"]["epochs"]),
    )
    loaded_warmup.load_state_dict(ckpt["warmup_state_dict"])
    assert set(loaded_warmup.state_dict().keys()) == set(
        ckpt["warmup_state_dict"].keys()
    ), "Loaded warmup state keys do not match expected keys"
    # ensure warmup scheduler state was restored (last_epoch should match checkpoint)
    assert (
        loaded_warmup.state_dict()["last_epoch"]
        == ckpt["warmup_state_dict"]["last_epoch"]
    )

    # Restore and check scheduler state
    loaded_scheduler = StepLR(loaded_optimizer, step_size=1, gamma=0.5)
    loaded_scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    assert set(loaded_scheduler.state_dict().keys()) == set(
        ckpt["scheduler_state_dict"].keys()
    ), "Loaded scheduler state keys do not match expected keys"
    # ensure scheduler last_epoch was restored
    assert (
        loaded_scheduler.state_dict()["last_epoch"]
        == ckpt["scheduler_state_dict"]["last_epoch"]
    )
