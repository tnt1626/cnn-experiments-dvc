# CNN Experiments with DVC

A PyTorch-based CNN project for CIFAR-10 image classification with DVC (Data Version Control) pipeline management. This repository demonstrates ML experiment tracking and parameter tuning for convolutional neural networks.

## Overview

This project implements a configurable CNN architecture (PinteeCNN) to classify images from the CIFAR-10 dataset. It uses DVC to manage reproducible data pipelines and track experiments with different hyperparameter configurations and architectural improvements.

**Current Performance:** 74.27% test accuracy

## Project Structure

```
├── src/                      # Source code
│   ├── model.py             # CNN architecture definition
│   ├── train.py             # Training script
│   ├── evaluate.py          # Evaluation script
│   └── data_prep.py         # Data preparation and preprocessing
├── data/                    # Data directory
│   ├── raw/                 # Raw CIFAR-10 dataset
│   └── processed/           # Processed training/test data (PyTorch format)
├── models/                  # Trained model weights
│   └── model.pth           # Serialized model
├── dvc.yaml                 # DVC pipeline stages
├── params.yaml              # Configuration parameters
├── pyproject.toml           # Project dependencies
├── metrics.json             # Evaluation metrics
├── training_logs.csv        # Training history
└── main.py                  # Entry point
```

## Setup

### Requirements

- Python >= 3.13
- PyTorch
- DVC >= 3.67.1
- NumPy, YAML, Matplotlib

### Installation

1. Clone the repository:
```bash
git clone <repo-url>
cd cnn-experiments-dvc
```

2. Install dependencies:
```bash
pip install -e .
```

3. Initialize DVC (if not already initialized):
```bash
dvc init
```

## Configuration

All model parameters are defined in `params.yaml`:

### Base Configuration
- `batch_size`: Training batch size (default: 256)
- `epochs`: Number of training epochs (default: 20)
- `learning_rate`: Adam optimizer learning rate (default: 0.001)
- `image_shape`: Input image dimensions [channels, height, width] (default: [3, 32, 32])
- `optimizer`: Optimization algorithm (default: adam)

### Improvements (Feature Flags)
- `use_relu`: Enable ReLU activation (default: true)
- `use_advanced_activation`: Use SiLU activation instead (default: false)
- `use_z_score`: Apply z-score normalization (default: true)
- `use_batch_norm`: Add batch normalization layers (default: false)
- `use_he_initialization`: Use He initialization (default: false)
- `use_skip_connection`: Add skip connections (default: true)
- `reduce_learning_rate`: Reduce LR to 1e-4 (default: false)
- `use_4_blocks`: Use 4 convolutional blocks (default: true)

## Running the Pipeline

### Full Pipeline
Run all stages (data preparation, training, evaluation):
```bash
dvc repro
```

### Individual Stages

Prepare data only:
```bash
dvc repro prepare_data
```

Train model only:
```bash
dvc repro train
```

Evaluate model only:
```bash
dvc repro evaluate
```

### Run Python Scripts Directly
```bash
python src/data_prep.py      # Prepare CIFAR-10 data
python src/train.py          # Train the model
python src/evaluate.py       # Evaluate on test set
```

## Model Architecture

The **PinteeCNN** model is a multi-block convolutional neural network designed for CIFAR-10 classification:

- **Input:** 3×32×32 RGB images
- **Blocks:** Multiple convolutional blocks with:
  - 3×3 convolutions
  - Optional batch normalization
  - Max pooling operations
  - Configurable activation functions (ReLU/SiLU/Sigmoid)
  - Optional skip connections
- **Output:** 10-class softmax predictions

Architecture is configurable via `params.yaml` improvement flags.

## Results

- **Test Accuracy:** 74.27%
- **Test Loss:** 1.433
- **Best Training Accuracy:** 96.93% (at epoch 19)

Results are saved to:
- `metrics.json` - Final evaluation metrics
- `training_logs.csv` - Per-epoch training history with loss and accuracy

## Experiment Tracking with DVC

DVC tracks:
- **Dependencies:** Data files and source code changes
- **Outputs:** Generated models and processed data
- **Metrics:** Evaluation results
- **Plots:** Training curves from `training_logs.csv`

View pipeline DAG:
```bash
dvc dag
```

Check pipeline status:
```bash
dvc status
```

## Key Features

- ✅ Reproducible ML pipeline with DVC
- ✅ Configurable architecture and training parameters
- ✅ Multiple improvement flags for architecture experimentation
- ✅ Automatic metrics tracking and logging
- ✅ GPU support (CUDA if available, otherwise CPU)
- ✅ Training history logging to CSV for analysis
- ✅ Clean modular code organization

## Dependencies

See `pyproject.toml` for complete dependency list:
- `dvc` - Pipeline management
- `torch` & `torchvision` - Deep learning framework
- `numpy` - Numerical computing
- `pyyaml` - Configuration parsing
- `matplotlib` - Visualization

## License

[Add your license here]
