# CVNN: Complex-Valued Neural Networks for Computer Vision

A modular and extensible framework for computer vision projects with comprehensive support for complex-valued neural networks across multiple tasks: classification, segmentation, reconstruction, and generation.

## 🌟 Key Features

- **🔌 Plugin-Based Architecture**: Automatic experiment discovery from `projects/*/experiment.py`
- **⚙️ Hierarchical Configuration**: Base configuration with task-specific specializations and automatic merging
- **🚀 Unified CLI**: Single command-line interface via `python -m cvnn`
- **🎯 Automatic Task Dispatch**: Configuration-to-experiment mapping via the `task` field
- **📊 Built-in W&B Integration**: Native experiment tracking and logging

## CI & Testing

This project separates fast, PR-friendly smoke tests from the full test-suite to provide quick feedback and full validation on merges.

- Smoke tests: small, fast tests marked with the `smoke` pytest marker. These run on pull requests to catch obvious regressions quickly.
- Full tests: the full `pytest` run which runs the complete test-suite (slower) and is executed on pushes to `main`/`master`.

CI notes
- A GitHub Actions workflow (`.github/workflows/ci-split.yml`) is provided in the repository. It runs the smoke job on PRs and the full job on pushes to `main`/`master`.
- Consider enabling dependency caching for Poetry and adding `pytest-xdist` to speed up the full job on CI.

Run tests locally

```bash
# Run only smoke tests (fast)
pytest -q -m smoke

# Run full test-suite
pytest -q

# Run a single test file
pytest tests/test_unet_encode_decode.py -q

# Run with coverage
pytest --cov=src/cvnn tests/
```

- **🧮 Comprehensive Metrics Registry**: Unified evaluation system for all tasks
- **🔬 Real & Complex-Valued Support**: Full pipeline support for both real and complex neural networks

## 📚 Table of Contents

- [Installation](#installation)
- [Architecture Overview](#architecture-overview)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Task-Specific Guides](#task-specific-guides)
- [Configuration System](#configuration-system)
- [Metrics and Evaluation](#metrics-and-evaluation)
- [Development](#development)
- [Examples](#examples)

## 🚀 Installation

### Prerequisites
- Python 3.12+
- Poetry for dependency management

### Setup

1. **Clone and install dependencies**:
   ```bash
   git clone <repository-url>
   cd cvnn
   poetry install
   ```

2. **Activate the virtual environment**:
   ```bash
   poetry shell
   ```

3. **Verify installation**:
   ```bash
   python -m cvnn --help
   ```

## 🏗️ Architecture Overview

CVNN follows a modular, plugin-based architecture that separates core pipeline logic from task-specific implementations:

```
cvnn/
├── src/cvnn/                    # Core pipeline framework
│   ├── base_experiment.py       # Abstract base experiment class
│   ├── metrics_registry.py      # Unified metrics system
│   ├── evaluate.py              # Task-agnostic evaluation functions
│   ├── train.py                 # Training utilities
│   ├── config.py                # Configuration management
│   ├── data.py                  # Data loading and preprocessing
│   ├── models/                  # Model definitions
│   └── cli/                     # Command-line interface
├── projects/                    # Task-specific implementations
│   ├── classification/experiment.py
│   ├── reconstruction/experiment.py
│   ├── segmentation/experiment.py
│   └── generation/experiment.py
├── configs/                     # Hierarchical configuration files
│   ├── config.yaml              # Base configuration
│   ├── config_classification.yaml
│   ├── config_reconstruction.yaml
│   ├── config_segmentation.yaml
│   └── config_generation.yaml
└── tests/                       # Comprehensive test suite
```

### Key Design Principles

1. **Separation of Concerns**: Core pipeline logic is separate from task-specific code
2. **Configuration-Driven**: All experiments are defined through YAML configurations
3. **Plugin Discovery**: Experiments are automatically discovered and registered
4. **Unified Evaluation**: Single metrics registry handles all task types
5. **Reproducibility**: Built-in seed management and experiment logging

## ⚡ Quick Start

### Running Your First Experiment

```bash
# Full pipeline (train + evaluate + visualize)
python -m cvnn configs/config_reconstruction.yaml --mode full

# Train only
python -m cvnn configs/config_classification.yaml --mode train

# Evaluate existing model
python -m cvnn configs/config_segmentation.yaml \
    --mode eval --resume-logdir logs/segmentation_experiment_*

# Resume training from checkpoint
python -m cvnn configs/config_generation.yaml \
    --mode retrain --resume-logdir logs/generation_*
```

### Basic Configuration Example

```yaml
# config_my_experiment.yaml
task: "reconstruction"  # Auto-dispatches to projects/reconstruction/experiment.py
project_name: "my_cvnn_project"

# Dataset configuration
dataset:
  name: "synthetic_sar"
  path: "datasets/synthetic_sar"
  validation_split: 0.2
  
# Model architecture
model:
  name: "ComplexUNet"
  input_channels: 2
  output_channels: 2
  use_complex: true
  
# Training parameters
training:
  epochs: 100
  batch_size: 16
  learning_rate: 0.001
  
# Evaluation metrics
evaluation:
  metrics: ["mse", "psnr", "ssim"]
```

## 🧠 Core Concepts

### 1. Task-Based Architecture

CVNN supports four main computer vision tasks:

- **Classification**: Image classification with complex-valued features
- **Segmentation**: Pixel-wise classification for image segmentation
- **Reconstruction**: Image reconstruction and denoising
- **Generation**: Generative modeling and synthesis

Each task is implemented as a separate experiment class inheriting from `BaseExperiment`.

### 2. Complex-Valued Neural Networks

CVNN provides full support for complex-valued neural networks through integration with `torchcvnn`:

```python
# Example: Complex-valued convolution
import torchcvnn.nn as cv_nn

class ComplexModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = cv_nn.ComplexConv2d(2, 32, 3)
        self.conv2 = cv_nn.ComplexConv2d(32, 64, 3)
        self.activation = cv_nn.ComplexReLU()
        
    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        return x
```

### 3. Unified Metrics Registry

All tasks use a centralized metrics registry for consistent evaluation:

```python
from cvnn.metrics_registry import MetricsRegistry

# Initialize registry for a task
registry = MetricsRegistry(task="reconstruction", cfg=config)

# Compute metrics
results = registry.compute_metrics(predictions, targets)
# Returns: {"mse": 0.01, "psnr": 42.5, "ssim": 0.95}
```

### 4. Configuration System

Configurations use a hierarchical system with inheritance:

1. **Base config** (`configs/config.yaml`): Common settings
2. **Task-specific configs**: Override and extend base settings
3. **Runtime overrides**: Command-line parameter overrides

## 📊 Task-Specific Guides

### Classification

Image classification with support for complex-valued features and multi-class/multi-label scenarios.

**Key Features:**
- Standard and complex-valued CNN architectures
- Multi-class and multi-label classification
- Comprehensive classification metrics (accuracy, precision, recall, F1)

**Example Configuration:**
```yaml
task: "classification"
dataset:
  name: "complex_mnist"
  num_classes: 10
model:
  name: "ComplexResNet18"
  use_complex: true
evaluation:
  metrics: ["accuracy", "precision", "recall", "f1"]
```

### Reconstruction

Image reconstruction, denoising, and restoration tasks.

**Key Features:**
- Encoder-decoder architectures (U-Net, AutoEncoder)
- Complex-valued reconstruction for SAR/medical imaging
- Advanced reconstruction metrics (PSNR, SSIM, structural similarity)

**Example Configuration:**
```yaml
task: "reconstruction"
dataset:
  name: "synthetic_sar"
  noise_level: 0.1
model:
  name: "ComplexUNet"
  use_complex: true
evaluation:
  metrics: ["mse", "psnr", "ssim", "structural_similarity"]
```

### Segmentation

Pixel-wise classification for semantic and instance segmentation.

**Key Features:**
- U-Net and DeepLab architectures
- Multi-class segmentation support
- Segmentation-specific metrics (IoU, Dice coefficient)

**Example Configuration:**
```yaml
task: "segmentation"
dataset:
  name: "medical_segmentation"
  num_classes: 5
model:
  name: "UNet"
  use_complex: false
evaluation:
  metrics: ["iou", "dice", "pixel_accuracy"]
```

### Generation

Generative modeling for image synthesis and augmentation.

**Key Features:**
- GAN and VAE architectures
- Complex-valued generation
- Generation quality metrics (FID, IS, LPIPS)

**Example Configuration:**
```yaml
task: "generation"
dataset:
  name: "complex_textures"
model:
  name: "ComplexGAN"
  use_complex: true
evaluation:
  metrics: ["fid", "inception_score"]
```

## 🔧 Configuration System

### Configuration Hierarchy

1. **Base Configuration** (`configs/config.yaml`):
   ```yaml
   # Common settings for all tasks
   project_name: "cvnn_experiments"
   seed: 42
   device: "auto"  # auto-detect GPU/CPU
   
   logging:
     use_wandb: true
     log_level: "INFO"
   
   training:
     epochs: 100
     batch_size: 32
     optimizer: "adam"
   ```

2. **Task-Specific Configuration**:
   ```yaml
   # config_reconstruction.yaml
   task: "reconstruction"
   
   # Inherits from base config and adds:
   model:
     name: "ComplexUNet"
     use_complex: true
   
   evaluation:
     metrics: ["mse", "psnr", "ssim"]
   ```

### Configuration Access

```python
from cvnn.config import load_config

# Load configuration with hierarchy resolution
cfg = load_config("configs/config_reconstruction.yaml")

# Access nested configuration
model_name = cfg.model.name
learning_rate = cfg.training.learning_rate
metrics = cfg.evaluation.metrics
```

## 📈 Metrics and Evaluation

### Metrics Registry

The `MetricsRegistry` provides a unified interface for computing task-specific metrics:

```python
from cvnn.metrics_registry import MetricsRegistry

# Initialize for specific task
registry = MetricsRegistry(task="reconstruction", cfg=config)

# Get available metrics
available_metrics = registry.get_available_metrics()
# Returns: ["mse", "mae", "psnr", "ssim", "structural_similarity"]

# Compute metrics
predictions = model(inputs)
results = registry.compute_metrics(predictions, targets)
```

### Supported Metrics by Task

**Classification:**
- `accuracy`: Overall classification accuracy
- `precision`: Precision score (macro/micro/weighted)
- `recall`: Recall score (macro/micro/weighted)
- `f1`: F1 score (macro/micro/weighted)
- `roc_auc`: Area under ROC curve

**Reconstruction:**
- `mse`: Mean Squared Error
- `mae`: Mean Absolute Error
- `psnr`: Peak Signal-to-Noise Ratio
- `ssim`: Structural Similarity Index
- `structural_similarity`: Advanced structural similarity

**Segmentation:**
- `iou`: Intersection over Union (IoU)
- `dice`: Dice coefficient
- `pixel_accuracy`: Pixel-wise accuracy
- `mean_iou`: Mean IoU across classes

**Generation:**
- `fid`: Fréchet Inception Distance
- `inception_score`: Inception Score
- `lpips`: Learned Perceptual Image Patch Similarity

### Custom Metrics

Add custom metrics by extending the registry:

```python
from cvnn.metrics_registry import MetricsRegistry

class CustomMetricsRegistry(MetricsRegistry):
    def compute_custom_metric(self, predictions, targets):
        # Your custom metric implementation
        return custom_score

# Use in your experiment
registry = CustomMetricsRegistry(task="reconstruction", cfg=config)
```

## 🛠️ Development

### Creating a New Task

1. **Create the experiment class**:
   ```python
   # projects/my_task/experiment.py
   from cvnn.base_experiment import BaseExperiment
   
   class MyTaskExperiment(BaseExperiment):
       def __init__(self, cfg):
           super().__init__(cfg)
           
       def prepare_data(self):
           # Implement data preparation
           pass
           
       def run_training(self):
           # Implement training logic
           pass
           
       def run_evaluation(self):
           # Implement evaluation logic
           pass
   ```

2. **Create task configuration**:
   ```yaml
   # configs/config_my_task.yaml
   task: "my_task"
   
   # Task-specific settings
   model:
     name: "MyTaskModel"
   
   evaluation:
     metrics: ["my_custom_metric"]
   ```

3. **Add metrics to registry**:
   ```python
   # Extend MetricsRegistry for your task
   from cvnn.metrics_registry import MetricsRegistry
   
   class MyTaskMetricsRegistry(MetricsRegistry):
       def compute_my_custom_metric(self, predictions, targets):
           # Implement your metric
           return score
   ```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_metrics_registry.py

# Run with coverage
pytest --cov=src/cvnn tests/
```

### Code Style

The project uses:
- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Run linting
flake8 src/ tests/

# Type checking
mypy src/cvnn/
```

## 💡 Examples

### Example 1: Complex-Valued SAR Image Reconstruction

```python
# configs/sar_reconstruction.yaml
task: "reconstruction"
project_name: "sar_denoising"

dataset:
  name: "synthetic_sar"
  path: "datasets/synthetic_sar"
  noise_level: 0.2
  validation_split: 0.2

model:
  name: "ComplexUNet"
  input_channels: 2  # Complex: real + imaginary
  output_channels: 2
  use_complex: true
  encoder_depths: [64, 128, 256, 512]

training:
  epochs: 150
  batch_size: 16
  learning_rate: 0.001
  scheduler: "cosine"

evaluation:
  metrics: ["mse", "psnr", "ssim"]
  save_visualizations: true
```

```bash
# Run the experiment
python -m cvnn configs/sar_reconstruction.yaml --mode full
```

### Example 2: Multi-Class Medical Image Segmentation

```python
# configs/medical_segmentation.yaml
task: "segmentation"
project_name: "medical_organ_segmentation"

dataset:
  name: "medical_ct"
  path: "datasets/medical_ct"
  num_classes: 4  # Background + 3 organs
  validation_split: 0.15

model:
  name: "UNet"
  input_channels: 1  # Grayscale CT
  num_classes: 4
  use_complex: false

training:
  epochs: 200
  batch_size: 8
  learning_rate: 0.0001
  loss_function: "dice_loss"

evaluation:
  metrics: ["dice", "iou", "pixel_accuracy"]
  compute_per_class: true
```

### Example 3: Complex-Valued Image Classification

```python
# configs/complex_classification.yaml
task: "classification"
project_name: "complex_mnist_classification"

dataset:
  name: "complex_mnist"
  path: "datasets/complex_mnist"
  num_classes: 10

model:
  name: "ComplexResNet18"
  use_complex: true
  num_classes: 10
  dropout: 0.1

training:
  epochs: 50
  batch_size: 64
  learning_rate: 0.01
  weight_decay: 0.0001

evaluation:
  metrics: ["accuracy", "precision", "recall", "f1"]
  confusion_matrix: true
```

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/my-feature`
3. **Make your changes** following the code style guidelines
4. **Add tests** for new functionality
5. **Run the test suite**: `pytest`
6. **Submit a pull request**

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [PyTorch](https://pytorch.org/) and [TorchCVNN](https://github.com/ivannz/torch-cvnn)
- Configuration management via [Hydra](https://hydra.cc/)
- Experiment tracking with [Weights & Biases](https://wandb.ai/)
- Complex-valued neural network support from the [CVNN community](https://github.com/NEGU93/cvnn)

---

**Ready to start your complex-valued computer vision journey?** 🚀

```bash
git clone <repository-url>
cd cvnn
poetry install
poetry shell
python -m cvnn configs/config_reconstruction.yaml --mode full
```
