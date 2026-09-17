import pytest
import torch
import torch.nn as nn
import numpy as np
import torchcvnn.nn.modules as c_nn
from cvnn.models.models import AutoEncoder
from cvnn.models.utils import get_activation
from cvnn.models.conv import DoubleConv
from cvnn.models.conv import BaseConv2d, ConvBlock, SingleConv

# Helper to initialize conv weight and bias to identity mapping or scaling


def init_conv_identity(conv, in_ch, out_ch):
    # conv.weight shape: (out_ch, in_ch, kH, kW)
    with torch.no_grad():
        conv.weight.zero_()
        # identity mapping for each matching channel
        for i in range(min(in_ch, out_ch)):
            conv.weight[i, i, ...] = 1.0
        if conv.bias is not None:
            conv.bias.zero_()


@pytest.mark.parametrize("mode", ["complex", "split"])
def test_baseconv2d_forward_identity(mode):
    # Test that BaseConv2d with kernel_size=1, padding=0 performs identity mapping for each mode
    in_ch = 1
    out_ch = 1
    # Create a simple input: batch=2, channels=1, H=4, W=4, complex
    real = torch.randn(2, in_ch, 4, 4)
    imag = torch.randn(2, in_ch, 4, 4)
    x = torch.complex(real, imag)
    # Build block with identity conv
    block = BaseConv2d(
        in_ch, out_ch, kernel_size=1, stride=1, padding=0, conv_mode=mode
    )
    # initialize weight to identity
    # Note: For split mode, the conv layer itself still has in_ch x out_ch dimensions
    # The splitting happens in the forward pass, not in the conv layer structure
    init_conv_identity(block.conv, in_ch, out_ch)

    # forward pass
    out = block(x)
    # verify output matches input mapping
    if mode == "complex":
        # complex conv identity -> out == x
        assert torch.allclose(out, x)
    elif mode == "real":
        # real mode uses only real part
        assert out.dtype == real.dtype
        assert torch.allclose(out, real)
    elif mode == "split":
        # split mode applies conv to both parts
        assert torch.allclose(out.real, real)
        assert torch.allclose(out.imag, imag)


@pytest.mark.parametrize("mode", ["complex", "split"])
def test_convblock_and_singleconv_output_shape(mode):
    in_ch, out_ch = 2, 3
    batch, H, W = 1, 8, 8
    activation = None
    norm = None
    # random complex input
    x = torch.complex(torch.randn(batch, in_ch, H, W), torch.randn(batch, in_ch, H, W))
    # ConvBlock
    conv_block = ConvBlock(
        in_ch,
        out_ch,
        conv_mode=mode,
        activation=activation,
        normalization=norm,
        kernel_size=1,
        padding=0,
    )
    out_cb = conv_block(x)
    assert out_cb.shape == (batch, out_ch, H, W)
    # SingleConv
    single = SingleConv(
        in_ch, out_ch, conv_mode=mode, activation=activation, kernel_size=1, padding=0
    )
    out_sc = single(x)
    assert out_sc.shape == (batch, out_ch, H, W)


@pytest.mark.parametrize("mode", ["complex", "split"])
def test_doubleconv_and_autoencoder_shapes(mode):
    # DoubleConv should stack two ConvBlocks
    in_ch, mid_ch = 1, 1
    out_ch = 2
    activation = None
    batch, H, W = 1, 8, 8
    x = torch.complex(torch.ones(batch, in_ch, H, W), torch.zeros(batch, in_ch, H, W))

    # Test regular DoubleConv
    double = DoubleConv(
        in_ch=in_ch,
        out_ch=out_ch,
        conv_mode=mode,
        activation=activation,
        kernel_size=1,
        padding=0,
    )
    out_dc = double(x)
    assert out_dc.shape == (batch, out_ch, H, W)

    # AutoEncoder: make tiny model with input_size=8, num_layers=2
    ae = AutoEncoder(
        num_channels=in_ch,
        num_layers=2,
        channels_width=2,
        input_size=H,
        activation="modReLU",
        upsampling_layer="bilinear",  # Required parameter
        conv_mode=mode,
        num_blocks=1,
    )
    # Test checkpointing if method exists, otherwise skip
    if hasattr(ae, "use_checkpointing"):
        ae.use_checkpointing()
    out_ae = ae(x)
    assert out_ae.shape == (batch, in_ch, H, W)


@pytest.mark.parametrize("mode", ["complex", "split"])
def test_residual_doubleconv_with_different_modes(mode):
    """Test DoubleConv with residual connections for different conv_modes."""
    in_ch, out_ch = 2, 2
    batch, H, W = 1, 8, 8

    # Create input tensor
    x = torch.complex(torch.randn(batch, in_ch, H, W), torch.randn(batch, in_ch, H, W))

    # Create DoubleConv with residual connection
    double_res = DoubleConv(
        in_ch=in_ch,
        out_ch=out_ch,
        conv_mode=mode,
        residual=True,
        kernel_size=3,
        padding=1,
    )

    # Create DoubleConv without residual connection
    double_no_res = DoubleConv(
        in_ch=in_ch,
        out_ch=out_ch,
        conv_mode=mode,
        residual=False,
        kernel_size=3,
        padding=1,
    )

    # Initialize weights to small values
    for module in double_res.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, mean=0, std=0.01)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    for module in double_no_res.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, mean=0, std=0.01)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # Forward pass
    with torch.no_grad():
        out_res = double_res(x)
        out_no_res = double_no_res(x)

        # Check shapes
        assert out_res.shape == (batch, out_ch, H, W)
        assert out_no_res.shape == (batch, out_ch, H, W)

        # For residual connections, verify skip connection is working
        if in_ch == out_ch:  # Only for matching channel dimensions
            # The residual output should be closer to the input in magnitude
            input_mag = torch.abs(x).mean().item()
            res_mag = torch.abs(out_res).mean().item()
            no_res_mag = torch.abs(out_no_res).mean().item()

            # Skip connection should make the output closer to the input
            assert abs(res_mag - input_mag) < abs(
                no_res_mag - input_mag
            ), f"Residual not working properly for conv_mode={mode}"


@pytest.mark.parametrize("mode", ["complex", "split"])
def test_autoencoder_with_residual(mode):
    """Test AutoEncoder with residual connections for different conv_modes."""
    in_ch, H, W = 1, 16, 16
    batch = 1

    # Create input tensor
    x = torch.complex(torch.randn(batch, in_ch, H, W), torch.randn(batch, in_ch, H, W))

    # Create model with residual connections
    model_res = AutoEncoder(
        num_channels=in_ch,
        num_layers=2,
        channels_width=2,
        input_size=H,
        activation="modReLU",
        upsampling_layer="bilinear",
        conv_mode=mode,
        residual=True,
        num_blocks=1,
    )

    # Create model without residual connections
    model_no_res = AutoEncoder(
        num_channels=in_ch,
        num_layers=2,
        channels_width=2,
        input_size=H,
        activation="modReLU",
        upsampling_layer="bilinear",
        conv_mode=mode,
        residual=False,
        num_blocks=1,
    )

    # Forward pass
    with torch.no_grad():
        out_res = model_res(x)
        out_no_res = model_no_res(x)

        # Check shapes
        assert out_res.shape == (batch, in_ch, H, W)
        assert out_no_res.shape == (batch, in_ch, H, W)

        # Verify that the model contains residual DoubleConv blocks
        found_residual = False
        for name, module in model_res.named_modules():
            if hasattr(module, "residual") and module.residual:
                found_residual = True
                break

        assert (
            found_residual
        ), f"No residual blocks found in model with conv_mode={mode}"


def test_get_activation_invalid():
    with pytest.raises(ValueError):
        get_activation("NonExistentActivation", layer_mode="complex")
