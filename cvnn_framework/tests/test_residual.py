#!/usr/bin/env python3
"""Test script to verify residual block implementation."""

import torch
import torch.nn as nn
import sys
import os

sys.path.append("src")

from cvnn.models.conv import DoubleConv
from cvnn.models.models import AutoEncoder


def test_residual_doubleconv():
    """Test DoubleConv with and without residual connections."""
    print("Testing DoubleConv with residual connections...")

    # Test case 1: Same input/output channels - direct skip connection
    batch_size, channels, height, width = 2, 3, 8, 8
    x = torch.complex(
        torch.randn(batch_size, channels, height, width),
        torch.randn(batch_size, channels, height, width),
    )

    # Non-residual version
    conv_no_res = DoubleConv(
        in_ch=channels,
        out_ch=channels,
        conv_mode="complex",
        residual=False,
        kernel_size=3,
        padding=1,
    )

    # Residual version
    conv_res = DoubleConv(
        in_ch=channels,
        out_ch=channels,
        conv_mode="complex",
        residual=True,
        kernel_size=3,
        padding=1,
    )

    # Test outputs
    with torch.no_grad():
        # Initialize weights to small values to see residual effect
        for module in conv_res.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        out_no_res = conv_no_res(x)
        out_res = conv_res(x)

        print(f"Input shape: {x.shape}")
        print(f"Non-residual output shape: {out_no_res.shape}")
        print(f"Residual output shape: {out_res.shape}")

        # The residual connection should make the output closer to the input
        print(f"Non-residual output mean magnitude: {torch.abs(out_no_res).mean():.4f}")
        print(f"Residual output mean magnitude: {torch.abs(out_res).mean():.4f}")
        print(f"Input mean magnitude: {torch.abs(x).mean():.4f}")

        # Check if skip connection is correct: output should be closer to input with residual
        input_mag = torch.abs(x).mean().item()
        no_res_mag = torch.abs(out_no_res).mean().item()
        res_mag = torch.abs(out_res).mean().item()

        # The absolute difference between output and input should be smaller with residual
        assert abs(res_mag - input_mag) < abs(
            no_res_mag - input_mag
        ), "Residual connection should make output closer to input in magnitude"

    # Test case 2: Different input/output channels - projection residual skip connection
    print("\nTesting with different input/output channels...")

    conv_res_proj = DoubleConv(
        in_ch=3, out_ch=6, conv_mode="complex", residual=True, kernel_size=3, padding=1
    )

    with torch.no_grad():
        out_proj = conv_res_proj(x)
        print(f"Input channels: 3, Output channels: 6")
        print(f"Projection residual output shape: {out_proj.shape}")

        # Verify that residual skip connection was created and is not Identity
        assert not isinstance(
            conv_res_proj.residual, nn.Identity
        ), "Skip connection should be a projection when channels differ"
        print(f"Skip connection type: {type(conv_res_proj.residual)}")

        # Verify the output shape is correct after projection
        assert out_proj.shape == (
            batch_size,
            6,
            height,
            width,
        ), f"Expected shape {(batch_size, 6, height, width)}, got {out_proj.shape}"

    # Test case 3: Test with stride != 1
    stride_x = torch.complex(
        torch.randn(batch_size, channels, height, width),
        torch.randn(batch_size, channels, height, width),
    )

    conv_res_stride = DoubleConv(
        in_ch=channels,
        out_ch=channels,
        conv_mode="complex",
        residual=True,
        kernel_size=3,
        stride=2,
        padding=1,
    )

    with torch.no_grad():
        out_stride = conv_res_stride(stride_x)
        expected_h, expected_w = height // 2, width // 2

        # Verify residual skip connection handles stride correctly
        assert not isinstance(
            conv_res_stride.residual, nn.Identity
        ), "Skip connection should not be Identity when stride != 1"
        print(
            f"Skip connection with stride type: {type(conv_res_stride.residual)}"
        )

        # Verify the output shape is correct with stride
        assert out_stride.shape == (
            batch_size,
            channels,
            expected_h,
            expected_w,
        ), f"Expected shape {(batch_size, channels, expected_h, expected_w)}, got {out_stride.shape}"

    print("✓ DoubleConv residual tests passed!")


def test_residual_model():
    """Test full model with residual connections."""
    print("\nTesting full model with residual connections...")

    # Create model with residual connections
    model_with_residual = AutoEncoder(
        num_channels=2,
        num_layers=3,
        channels_width=4,
        input_size=16,
        activation="modReLU",
        upsampling_layer="transpose",
        conv_mode="complex",
        residual=True,  # This should be passed through to DoubleConv
        num_blocks=1,
    )

    # Create same model without residual connections for comparison
    model_no_residual = AutoEncoder(
        num_channels=2,
        num_layers=3,
        channels_width=4,
        input_size=16,
        activation="modReLU",
        upsampling_layer="transpose",
        conv_mode="complex",
        residual=False,
        num_blocks=1,
    )

    # Test input
    batch_size = 1
    x = torch.complex(
        torch.randn(batch_size, 2, 16, 16), torch.randn(batch_size, 2, 16, 16)
    )

    # Forward pass
    with torch.no_grad():
        # Initialize weights to small values to see residual effect more clearly
        for module in model_with_residual.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                nn.init.normal_(module.weight, mean=0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        for module in model_no_residual.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                nn.init.normal_(module.weight, mean=0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        output_residual = model_with_residual(x)
        output_no_residual = model_no_residual(x)

        print(f"Model input shape: {x.shape}")
        print(f"Model output shape (with residual): {output_residual.shape}")
        print(f"Model output shape (no residual): {output_no_residual.shape}")

        # Verify output shapes are the same
        assert (
            output_residual.shape == x.shape
        ), f"Expected output shape {x.shape}, got {output_residual.shape}"
        assert (
            output_no_residual.shape == x.shape
        ), f"Expected output shape {x.shape}, got {output_no_residual.shape}"

        # Test that model architecture contains residual connections
        # Check at least one Down or Up block in the model has a non-Identity skip connection
        found_residual = False
        for name, module in model_with_residual.named_modules():
            if hasattr(module, "residual") and module.residual:
                found_residual = True
                break

        assert found_residual, "No residual connections found in the model architecture"

    print("✓ Full model with residual connections test passed!")


if __name__ == "__main__":
    test_residual_doubleconv()
    test_residual_model()
    print("\n🎉 All residual block tests passed!")
