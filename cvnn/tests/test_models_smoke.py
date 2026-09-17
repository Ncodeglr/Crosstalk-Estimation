import torch
import torch.nn as nn

from cvnn.models.models import LatentAutoEncoder, UNet, ResNet


def _make_input(channels=1, size=16):
    return torch.randn(1, channels, size, size)


def test_latent_autoencoder_forward_basic():
    model = LatentAutoEncoder(
        num_channels=1,
        num_layers=1,
        channels_width=4,
        input_size=16,
        activation="ReLU",
        latent_dim=8,
        upsampling_layer="nearest",
        layer_mode="real",
        num_blocks=1,
    )
    x = _make_input(channels=1, size=16)
    out = model(x)
    # out should be a tensor with same spatial dims
    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == x.shape[0]


def test_unet_forward_basic():
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
    x = _make_input(channels=1, size=16)
    out = model(x)
    # UNet.forward may return tuple (x, x_projected) per implementation
    assert isinstance(out, tuple) or isinstance(out, torch.Tensor)


def test_resnet_forward_basic():
    model = ResNet(
        num_channels=1,
        num_layers=1,
        channels_width=4,
        input_size=16,
        activation="ReLU",
        num_classes=2,
        layer_mode="real",
        downsampling_layer="maxpool",
        num_blocks=1,
    )
    x = _make_input(channels=1, size=16)
    out = model(x)
    # ResNet.forward may return tuple (x, x_projected) per implementation
    assert isinstance(out, tuple) or isinstance(out, torch.Tensor)
