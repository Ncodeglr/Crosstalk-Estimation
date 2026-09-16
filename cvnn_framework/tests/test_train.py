import os
import torch
import torch.nn as nn
import torchcvnn.nn.modules as c_nn
import pytest
from torch.utils.data import DataLoader, TensorDataset

import cvnn.train as train_mod
from cvnn.train import (
    ModelCheckpoint,
    generate_unique_logpath,
    train_one_epoch,
    validate_one_epoch,
    setup_loss_optimizer,
    train_model,
)


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # add a dummy parameter so optimizer has parameters
        self.dummy = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        # add dummy parameter to input to enable gradient computation
        return x + self.dummy


@pytest.mark.parametrize(
    "initial,score,expected", [(None, 1.0, True), (1.0, 0.5, True), (0.5, 2.0, False)]
)
def test_lower_higher_is_better(initial, score, expected):
    model = DummyModel()
    optim = torch.optim.SGD(model.parameters(), lr=0.1)
    cp = ModelCheckpoint(model, optim, savepath=".", num_input_dims=4, min_is_best=True)
    cp.best_score = initial
    assert cp.lower_is_better(score) == expected
    cp = ModelCheckpoint(
        model, optim, savepath=".", num_input_dims=4, min_is_best=False
    )
    cp.best_score = initial
    assert cp.higher_is_better(score) == (initial is None or score > initial)


def test_modelcheckpoint_update(tmp_path):
    model = nn.Linear(2, 2)
    optim = torch.optim.SGD(model.parameters(), lr=0.1)
    save_dir = tmp_path
    cp = ModelCheckpoint(
        model, optim, str(save_dir), num_input_dims=2, min_is_best=True
    )
    # first update
    res1 = cp.update(score=0.5, epoch=0)
    assert res1 is True
    best_file = os.path.join(str(save_dir), "best_model.pt")
    assert os.path.isfile(best_file)
    # second update with worse score
    res2 = cp.update(score=1.0, epoch=1)
    assert res2 is False


def test_generate_unique_logpath(tmp_path):
    base = tmp_path / "runs"
    base.mkdir()
    # create some dirs
    (base / "exp_0").mkdir()
    (base / "exp_2").mkdir()
    # bad suffix
    (base / "exp_x").mkdir()
    new_path = generate_unique_logpath(str(base), "exp")
    assert new_path.endswith("exp_3")
    assert os.path.isdir(new_path)


def make_loader(n_samples=4, batch_size=2):
    # simple dataset: inputs shape (1,), outputs unused
    data = torch.arange(n_samples, dtype=torch.float32).unsqueeze(-1)
    ds = TensorDataset(data)
    return DataLoader(ds, batch_size=batch_size)


def test_train_validate_one_epoch():
    model = DummyModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loader = make_loader(n_samples=6, batch_size=3)
    # train: since model identity and criterion compares input-output, loss=0
    train_loss, _ = train_one_epoch(model, loader, criterion, optimizer, device="cpu")
    valid_loss, _ = validate_one_epoch(model, loader, criterion, device="cpu")
    assert train_loss == pytest.approx(0.0)
    assert valid_loss == pytest.approx(0.0)


def test_setup_loss_optimizer():
    model = DummyModel()
    cfg = {
        "loss": {"name": "MSE"},
        "optim": {"algo": "SGD", "params": {"lr": 0.2}},
    }
    # provide a minimal dataset and device required by the function signature
    dummy_data = make_loader(n_samples=2, batch_size=1).dataset
    # include minimal data/task keys expected by setup_loss_optimizer
    cfg["data"] = {}
    cfg["task"] = "reconstruction"
    cfg["model"] = {}  # Add missing model key
    loss_fn, optimizer = setup_loss_optimizer(model, cfg, dummy_data, device="cpu")
    assert isinstance(loss_fn, c_nn.ComplexMSELoss)
    assert isinstance(optimizer, torch.optim.SGD)
    # optimizer has correct lr
    assert optimizer.param_groups[0]["lr"] == 0.2


def test_train_model(tmp_path, monkeypatch):
    # prepare simple model, data, config
    model = DummyModel()
    train_loader = make_loader(n_samples=2, batch_size=1)
    valid_loader = make_loader(n_samples=2, batch_size=1)
    cfg = {
        "nepochs": 1,
        "loss": {"name": "MSE"},
        "optim": {"algo": "SGD", "params": {"lr": 0.1}},
    }
    # monkeypatch wandb.run to False
    import wandb

    monkeypatch.setattr(wandb, "run", None, raising=False)
    # pass an explicit loss_fn to avoid relying on setup inside train_model
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    history = train_model(
        model,
        train_loader,
        valid_loader,
        cfg,
        str(tmp_path),
        device="cpu",
        loss_fn=nn.MSELoss(),
        optimizer=optimizer,
    )
    assert "train_loss" in history and "valid_loss" in history
    assert len(history["train_loss"]) == 1 and len(history["valid_loss"]) == 1
    # check that last_model.pt exists
    assert os.path.isfile(tmp_path / "last_model.pt")
    # best_model.pt for zero loss exists
    assert os.path.isfile(tmp_path / "best_model.pt")
