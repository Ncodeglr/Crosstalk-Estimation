import sys
from pathlib import Path
import pytest

# Ensure src is on PYTHONPATH
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import cvnn  # noqa: F401


@pytest.fixture(autouse=True)
def seed_reproducibility():
    """Set a fixed random seed for reproducibility"""
    import numpy as np
    import torch

    np.random.seed(42)
    torch.manual_seed(42)


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create a temporary directory with a small synthetic dataset"""
    # Example: two classes with one image each
    class_dirs = [tmp_path / "class0", tmp_path / "class1"]
    for d in class_dirs:
        d.mkdir()
        # create a dummy .npy file
        import numpy as np

        arr = np.random.randn(8, 8)
        np.save(d / f"sample_{d.name}.npy", arr)
    return tmp_path


@pytest.fixture
def sample_config(tmp_path):
    """Write a minimal YAML config file"""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
        model:
          type: toy
        training:
          epochs: 1
          batch_size: 2
        """
    )
    return str(cfg)


@pytest.fixture
def complex_input_tensor():
    """Create a complex tensor for testing."""
    import torch

    batch_size, channels, height, width = 2, 3, 8, 8
    return torch.complex(
        torch.randn(batch_size, channels, height, width),
        torch.randn(batch_size, channels, height, width),
    )


@pytest.fixture
def initialize_small_weights():
    """Returns a function to initialize model weights to small values for residual testing."""
    import torch.nn as nn

    def _init_small_weights(model):
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                nn.init.normal_(module.weight, mean=0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    return _init_small_weights
