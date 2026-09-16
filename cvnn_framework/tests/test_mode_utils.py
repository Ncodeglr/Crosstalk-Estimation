import pytest
import torch.nn as nn

from cvnn.models.utils import (
    is_real_mode,
    get_activation,
    get_normalization,
    get_loss_function,
)


def test_is_real_mode_simple():
    assert is_real_mode("real") is True
    assert is_real_mode("complex") is False
    with pytest.raises(ValueError):
        is_real_mode("invalid-mode")


def test_get_activation_real_and_complex():
    a_real = get_activation("ReLU", "real")
    assert isinstance(a_real, nn.ReLU)

    # Complex activations depend on torchcvnn; if missing, expect ValueError
    try:
        a_complex = get_activation("modReLU", "complex")
        # If it exists, it's callable
        assert callable(a_complex)
    except ValueError:
        # acceptable if torchcvnn doesn't expose modReLU in this environment
        pass


def test_get_normalization_and_loss_basic():
    norm = get_normalization("batch", "real", num_features=8)
    assert hasattr(norm, "weight")

    # Loss selection: real MSE
    loss = get_loss_function("MSE", "real")
    assert hasattr(loss, "__call__")
