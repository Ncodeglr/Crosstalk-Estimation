import pytest
import torch
import torch.nn as nn

from cvnn.models.utils import (
    get_dropout,
    get_projection,
    get_downsampling,
    get_upsampling,
    validate_layer_mode,
)
from cvnn.models import UNet


def test_validate_layer_mode_and_basic_constructors():
    # validate_layer_mode should accept known modes
    validate_layer_mode("real")
    validate_layer_mode("complex")
    validate_layer_mode("split")
    with pytest.raises(ValueError):
        validate_layer_mode("unknown-mode")


def test_dropout_and_projection_shapes():
    # Dropout returns nn.Module or None
    d = get_dropout(0.5, layer_mode="real", use_2d=True)
    assert isinstance(d, nn.Module)
    d_none = get_dropout(0.0, layer_mode="real")
    assert d_none is None

    # Projection returns Identity for real mode
    p = get_projection("amplitude", layer_mode="real")
    assert isinstance(p, nn.Module)


def test_down_up_sampling_identity_roundtrip():
    # Ensure that up/down sampling doesn't error and preserves module type
    down = get_downsampling("maxpool", layer_mode="real", factor=2)
    up = get_upsampling("nearest", layer_mode="real", factor=2)
    x = torch.randn(1, 4, 8, 8)
    y = down(x)
    z = up(y)
    assert z.shape[2] == x.shape[2]


def test_unet_encode_decode_roundtrip():
    model = UNet(
        num_channels=1,
        num_layers=1,
        channels_width=4,
        input_size=16,
        activation="ReLU",
        num_classes=2,
        layer_mode="real",
        downsampling_layer="maxpool",
        upsampling_layer="nearest",
        num_blocks=1,
    )
    x = torch.randn(1, 1, 16, 16)

    # encode should return encoded tensor and lists when requested
    encoded, probs, skips = model.encode(x, return_probs=True)
    assert isinstance(encoded, torch.Tensor)
    assert isinstance(probs, list)
    assert isinstance(skips, list)
    assert len(skips) >= 1

    # internal decoder should accept the lists and return (tensor, projected)
    decoded, projected = model.decode(encoded, probs=probs, skips=skips)
    assert isinstance(decoded, torch.Tensor)
    # projected may be None if projection is identity
    assert projected is None or isinstance(projected, torch.Tensor)

    # forward should still return (x, x_projected)
    out = model(x)
    assert isinstance(out, tuple) and len(out) == 2
    assert out[0].shape[0] == x.shape[0]
