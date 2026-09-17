#!/usr/bin/env python3
"""Tests for training models with residual connections."""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from cvnn.models.models import AutoEncoder
from cvnn.train import train_one_epoch, validate_one_epoch

import torchcvnn.nn.modules as c_nn


def create_complex_dataset(num_samples=10, input_size=8, channels=1):
    """Create a simple complex dataset for testing."""
    # Create complex input tensors
    x_real = torch.randn(num_samples, channels, input_size, input_size)
    x_imag = torch.randn(num_samples, channels, input_size, input_size)
    x = torch.complex(x_real, x_imag)

    # Target is same as input (for autoencoder testing)
    y = x.clone()

    return TensorDataset(x, y)


def test_train_autoencoder_with_residual(initialize_small_weights):
    """Test that AutoEncoder with residual connections trains properly."""
    # Parameters
    batch_size = 2
    input_size = 16
    channels = 1

    # Create models with and without residual connections
    model_with_residual = AutoEncoder(
        num_channels=channels,
        num_layers=2,
        channels_width=2,
        input_size=input_size,
        activation="modReLU",
        upsampling_layer="bilinear",
        conv_mode="complex",
        residual=True,
        num_blocks=1,
    )

    model_without_residual = AutoEncoder(
        num_channels=channels,
        num_layers=2,
        channels_width=2,
        input_size=input_size,
        activation="modReLU",
        upsampling_layer="bilinear",
        conv_mode="complex",
        residual=False,
        num_blocks=1,
    )

    # Initialize weights
    initialize_small_weights(model_with_residual)
    initialize_small_weights(model_without_residual)

    # Create datasets and dataloaders
    dataset = create_complex_dataset(
        num_samples=10, input_size=input_size, channels=channels
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Loss and optimizers
    criterion = c_nn.ComplexMSELoss()
    optimizer_residual = optim.Adam(model_with_residual.parameters(), lr=0.001)
    optimizer_no_residual = optim.Adam(model_without_residual.parameters(), lr=0.001)

    # Train for a few epochs and track loss
    num_epochs = 3
    residual_losses = []
    no_residual_losses = []

    for epoch in range(num_epochs):
        # Train with residual
        train_loss_res, _ = train_one_epoch(
            model_with_residual, dataloader, criterion, optimizer_residual, device="cpu"
        )
        residual_losses.append(train_loss_res)

        # Train without residual
        train_loss_no_res, _ = train_one_epoch(
            model_without_residual,
            dataloader,
            criterion,
            optimizer_no_residual,
            device="cpu",
        )
        no_residual_losses.append(train_loss_no_res)

    # Verify both models trained without errors
    assert len(residual_losses) == num_epochs
    assert len(no_residual_losses) == num_epochs

    # Verify loss decreased over time (training happened)
    assert (
        residual_losses[-1] < residual_losses[0]
    ), "Residual model loss did not decrease"
    assert (
        no_residual_losses[-1] < no_residual_losses[0]
    ), "Non-residual model loss did not decrease"


@pytest.mark.parametrize("conv_mode", ["complex", "split", "dual"])
def test_validate_with_residual_connections(conv_mode, initialize_small_weights):
    """Test validation with models using different conv_modes and residual connections."""
    # Parameters
    batch_size = 2
    input_size = 16
    channels = 1

    # Create model with residual connections
    model = AutoEncoder(
        num_channels=channels,
        num_layers=2,
        channels_width=2,
        input_size=input_size,
        activation="modReLU",
        upsampling_layer="bilinear",
        conv_mode=conv_mode,
        residual=True,
        num_blocks=1,
    )

    # Initialize weights
    initialize_small_weights(model)

    # Create datasets and dataloaders
    dataset = create_complex_dataset(
        num_samples=8, input_size=input_size, channels=channels
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Loss function
    criterion = c_nn.ComplexMSELoss()

    # Validate
    with torch.no_grad():
        val_loss, _ = validate_one_epoch(model, dataloader, criterion, device="cpu")

    # Ensure validation completes without errors and returns a valid loss
    assert isinstance(val_loss, float)
    assert not torch.isnan(torch.tensor(val_loss))


def test_residual_model_saving_loading(tmp_path, initialize_small_weights):
    """Test that residual models can be saved and loaded properly."""
    # Create model with residual connections
    model = AutoEncoder(
        num_channels=1,
        num_layers=2,
        channels_width=2,
        input_size=16,
        activation="modReLU",
        upsampling_layer="bilinear",
        conv_mode="complex",
        residual=True,
        num_blocks=1,
    )

    # Initialize weights
    initialize_small_weights(model)

    # Save model
    save_path = tmp_path / "residual_model.pt"
    torch.save(model.state_dict(), save_path)

    # Create a new model instance and load weights
    model_loaded = AutoEncoder(
        num_channels=1,
        num_layers=2,
        channels_width=2,
        input_size=16,
        activation="modReLU",
        upsampling_layer="bilinear",
        conv_mode="complex",
        residual=True,
        num_blocks=1,
    )
    model_loaded.load_state_dict(torch.load(save_path))

    # Verify that loaded model has residual connections
    found_residual = False
    for name, module in model_loaded.named_modules():
        if hasattr(module, "residual") and module.residual:
            found_residual = True
            break

    assert found_residual, "Loaded model should maintain residual connections"

    # Simple forward pass to ensure it runs without errors
    x = torch.complex(torch.randn(1, 1, 16, 16), torch.randn(1, 1, 16, 16))

    with torch.no_grad():
        output = model_loaded(x)

    assert output.shape == x.shape, "Output shape should match input shape"
