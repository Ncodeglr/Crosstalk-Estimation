import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytest
import torch.nn as nn

from cvnn.evaluate import (
    reconstruct_full_image,
    evaluate_reconstruction,
    evaluate_segmentation,
    evaluate_classification
)


class IdentityModel(torch.nn.Module):
    def forward(self, x):
        return x


class PatchDataset(Dataset):
    def __init__(self, patches):
        # patches: list of numpy arrays shape (C, H, W)
        self.patches = patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        # return a tensor (as batched later)
        return torch.from_numpy(self.patches[idx])


def make_full_loader(num_channels, crop, patch_size):
    # build synthetic full image and patches
    nb_rows = crop["end_row"] - crop["start_row"]
    nb_cols = crop["end_col"] - crop["start_col"]
    full = np.zeros((num_channels, nb_rows, nb_cols), dtype=np.float32)
    # fill full image with increasing integers per pixel
    count = 0
    for r in range(nb_rows):
        for c in range(nb_cols):
            full[:, r, c] = count
            count += 1
    # extract patches in row-major order
    patches = []
    for h in range(0, nb_rows, patch_size):
        for w in range(0, nb_cols, patch_size):
            block = full[:, h : h + patch_size, w : w + patch_size]
            patches.append(block)
    # convert to complex64 for evaluater
    patches = [p.astype(np.complex64) for p in patches]
    ds = PatchDataset(patches)
    return DataLoader(ds, batch_size=1)


def test_reconstruct_full_image_identity():
    # set up config and loader
    num_channels = 1
    patch_size = 2
    crop = {"start_row": 0, "start_col": 0, "end_row": 4, "end_col": 4}
    cfg = {
        "data": {"crop": crop, "num_channels": num_channels, "patch_size": patch_size}
    }
    model = IdentityModel()
    loader = make_full_loader(num_channels, crop, patch_size)
    orig, recon = reconstruct_full_image(
        model,
        loader,
        config=cfg,
        device=torch.device("cpu"),
        nsamples_per_rows=2,  # 4/2 = 2 patches per row
        nsamples_per_cols=2,  # 4/2 = 2 patches per col
    )
    # shapes match full image
    assert orig.shape == (
        num_channels,
        crop["end_row"] - crop["start_row"],
        crop["end_col"] - crop["start_col"],
    )
    assert np.allclose(orig, recon)
    # orig should match manual full
    # rebuild expected full
    expected = np.zeros_like(orig)
    count = 0
    for r in range(orig.shape[1]):
        for c in range(orig.shape[2]):
            expected[:, r, c] = count
            count += 1
    assert np.array_equal(orig, expected)


def test_reconstruction_metrics():
    """Test evaluation with basic configuration."""
    # set up config and loader
    num_channels = 1
    patch_size = 2
    crop = {"start_row": 0, "start_col": 0, "end_row": 4, "end_col": 4}
    model = IdentityModel()
    test_loader = PatchDataset(
        [np.ones((num_channels, patch_size, patch_size), dtype=np.complex64)]
    )
    loader = DataLoader(test_loader, batch_size=1)

    # Basic config for registry-based evaluation
    config = {
        "evaluation": {"metrics": ["mse", "psnr", "ssim"], "pipeline_type": "complex"},
        # Minimal keys expected by the evaluation dispatcher
        "data": {"type": "reconstruction", "dataset": {"name": "PolSFDataset"}, "batch_size": 1},
        "model": {"layer_mode": "complex"},
    }

    metrics = evaluate_reconstruction(
        loader, model, cfg=config, device=torch.device("cpu")
    )

    # check metrics
    assert metrics["mse"] == 0.0
    assert metrics["psnr"] == float("inf")  # PSNR is infinite for identical images
    # For identical complex images, SSIM should be close to 1.0
    assert metrics["ssim"] >= 0.9  # Allow some numerical tolerance


def test_reconstruction_metrics_with_detailed_ssim():
    """Test evaluation with detailed SSIM metrics enabled."""
    num_channels = 1
    patch_size = 4
    model = IdentityModel()

    # Create a test patch with both magnitude and phase structure
    x, y = np.meshgrid(np.linspace(-1, 1, patch_size), np.linspace(-1, 1, patch_size))
    magnitude = np.exp(-(x**2 + y**2))
    phase = np.arctan2(y, x)
    complex_patch = magnitude * np.exp(1j * phase)
    complex_patch = complex_patch.reshape(num_channels, patch_size, patch_size).astype(
        np.complex64
    )

    test_loader = PatchDataset([complex_patch])
    loader = DataLoader(test_loader, batch_size=1)

    # Configuration for detailed SSIM
    config = {
        "evaluation": {
            "metrics": ["mse", "psnr", "ssim", "detailed_ssim"],
            "pipeline_type": "complex",
            "detailed_ssim": True,
        },
        "data": {"type": "reconstruction", "dataset": {"name": "PolSFDataset"}, "batch_size": 1},
        "model": {"layer_mode": "complex"},
    }

    # Test with detailed SSIM
    metrics = evaluate_reconstruction(
        loader,
        model,
        cfg=config,
        device=torch.device("cpu"),
    )

    # Check that basic metrics are included
    assert "mse" in metrics
    assert "psnr" in metrics
    assert "ssim" in metrics

    # For identical images, all SSIM metrics should be high
    if "ssim_magnitude_only" in metrics:
        assert metrics["ssim_magnitude_only"] >= 0.9


def test_reconstruction_metrics_with_adaptive_ssim():
    """Test evaluation with adaptive SSIM method selection."""
    num_channels = 1
    patch_size = 4
    model = IdentityModel()

    # Create test data
    complex_patch = np.random.randn(num_channels, patch_size, patch_size).astype(
        np.complex64
    )
    test_loader = PatchDataset([complex_patch])
    loader = DataLoader(test_loader, batch_size=1)

    # Configuration for adaptive SSIM
    config = {
        "evaluation": {
            "metrics": ["mse", "psnr", "ssim"],
            "pipeline_type": "complex",
            "adaptive_ssim": True,
        },
        "data": {"type": "reconstruction", "dataset": {"name": "PolSFDataset"}, "batch_size": 1},
        "model": {"layer_mode": "complex"},
    }

    # Test with adaptive SSIM
    metrics = evaluate_reconstruction(
        loader,
        model,
        cfg=config,
        device=torch.device("cpu"),
    )

    # Should have basic metrics
    assert "mse" in metrics
    assert "psnr" in metrics
    assert "ssim" in metrics

    # For identity model, metrics should indicate perfect reconstruction
    assert metrics["mse"] == 0.0
    assert metrics["psnr"] == float("inf")




def test_evaluate_reconstruction_error_handling():
    """Test error handling in evaluation functions."""
    num_channels = 1
    patch_size = 2
    model = IdentityModel()

    # Empty dataset should be handled gracefully
    empty_loader = DataLoader(PatchDataset([]), batch_size=1)

    # Create basic config for empty data test
    config = {
        "data": {
            "num_channels": 1,
            "patch_size": 2,
            # include minimal dataset name required by dispatcher
            "type": "reconstruction",
            "dataset": {"name": "PolSFDataset"},
        },
        "model": {"layer_mode": "complex"},
    }

    # Test with empty data - should handle gracefully with config
    try:
        metrics = evaluate_reconstruction(
            empty_loader, model, cfg=config, device=torch.device("cpu")
        )
        # If it succeeds, metrics should be a dict
        assert isinstance(metrics, dict)
    except (ZeroDivisionError, StopIteration, RuntimeError):
        # These exceptions are acceptable for empty data
        pass


def test_evaluate_all_tasks():
    """Test that all evaluation functions can be called with basic configs."""
    num_channels = 1
    patch_size = 2
    model = IdentityModel()

    # Create simple test data
    test_data = np.ones((num_channels, patch_size, patch_size), dtype=np.complex64)
    test_loader = DataLoader(PatchDataset([test_data]), batch_size=1)

    # Test reconstruction evaluation

    # Provide a minimal cfg expected by evaluate_reconstruction
    minimal_cfg = {
        "data": {"type": "reconstruction", "dataset": {"name": "PolSFDataset"}, "batch_size": 1},
        "model": {"layer_mode": "complex"},
    }
    recon_metrics = evaluate_reconstruction(
        test_loader, model, cfg=minimal_cfg, device=torch.device("cpu")
    )
    assert isinstance(recon_metrics, dict)
    assert "mse" in recon_metrics

    # Test segmentation evaluation
    # evaluate_segmentation does not accept a cfg argument; call with loader, model, device
    seg_metrics = evaluate_segmentation(test_loader, model, cfg=minimal_cfg, device=torch.device("cpu"))
    assert isinstance(seg_metrics, dict)

    # Test classification evaluation
    class_metrics = evaluate_classification(test_loader, model, cfg=minimal_cfg, device=torch.device("cpu"))
    assert isinstance(class_metrics, dict)

    # # Test generation evaluation
    # gen_config = {
    #     "evaluation": {
    #         "metrics": ["fid", "inception_score"],
    #         "pipeline_type": "complex"
    #     }
    # }
    # gen_metrics = evaluate_generation(test_loader, model, cfg=gen_config, device=torch.device("cpu"))
    # assert isinstance(gen_metrics, dict)




def test_circular_shift_consistency():
    """Test circular shift consistency function for both reconstruction and segmentation tasks."""
    from cvnn.evaluate import circular_shift_consistency

    device = torch.device("cpu")

    # Test reconstruction task
    # Identity model should have perfect consistency (L2 norm = 0 for identical outputs)
    identity_model = IdentityModel()
    inputs = torch.randn(2, 3, 16, 16, dtype=torch.float32)

    consistency_recon = circular_shift_consistency(
        identity_model, inputs, "reconstruction", device
    )
    # For identity model, L2 norm should be very close to 0 (lower is better)
    assert (
        consistency_recon < 1e-5
    ), f"Identity model should have near-zero consistency score, got {consistency_recon}"

    # Test with complex inputs
    complex_inputs = torch.randn(2, 3, 16, 16, dtype=torch.complex64)
    consistency_complex = circular_shift_consistency(
        identity_model, complex_inputs, "reconstruction", device
    )
    assert (
        consistency_complex < 1e-5
    ), f"Identity model should have near-zero complex reconstruction consistency, got {consistency_complex}"

    # Test segmentation task
    # Create a simple classification model that outputs constant predictions
    class ConstantSegmentationModel(torch.nn.Module):
        def forward(self, x):
            batch_size = x.shape[0]
            height, width = x.shape[-2:]
            # Return constant class 0 for all pixels (perfect consistency expected)
            logits = torch.zeros(batch_size, 5, height, width)  # 5 classes
            logits[:, 0, :, :] = 10.0  # Strong logit for class 0
            return logits

    seg_model = ConstantSegmentationModel()
    consistency_seg = circular_shift_consistency(
        seg_model, inputs, "segmentation", device
    )
    # Constant predictions should have perfect agreement (consistency = 1.0)
    assert (
        consistency_seg == 1.0
    ), f"Constant segmentation model should have perfect consistency, got {consistency_seg}"

    # Test invalid task
    with pytest.raises(ValueError, match="Unsupported task"):
        circular_shift_consistency(identity_model, inputs, "invalid_task", device)
