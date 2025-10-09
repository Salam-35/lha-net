# LHA-Net: Lightweight Hierarchical Attention Network for AMOS22 Small Organ Segmentation

This repository contains the official PyTorch implementation of LHA-Net, a novel deep learning architecture designed for the challenging task of small organ segmentation in abdominal CT scans. The model was developed for the AMOS22 (Abdominal Multi-Organ Segmentation) challenge.

## Features

- **Hierarchical Attention:** Employs a hierarchical attention mechanism to focus on features at different scales, improving the segmentation of small and variably-sized organs.
- **Lightweight Design:** Optimized for efficiency with a lightweight backbone and memory-saving techniques, allowing for training on GPUs with moderate memory capacity (e.g., 24GB).
- **Smart Patch Sampling:** A sophisticated patch sampling strategy that prioritizes small organs, boundary regions, and difficult examples, leading to more effective training.
- **Advanced Training Techniques:** Incorporates mixed-precision training, gradient accumulation, and gradient checkpointing to accelerate training and reduce memory footprint.
- **Comprehensive Configuration:** A single YAML file (`configs/lha_net_config.yaml`) allows for easy configuration of all aspects of the model, training, and data processing.
- **Reproducibility:** The code is structured for reproducibility, with a fixed random seed and options for deterministic training.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/lha-net.git
    cd lha-net
    ```

2.  **Create a Python virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Dataset Setup

The model is designed to work with the AMOS22 dataset. You need to download the data and structure it as follows:

```
/path/to/amos22/dataset/
├───imagesTr/
│   ├───amos_0001.nii.gz
│   ├───amos_0002.nii.gz
│   └───...
├───labelsTr/
│   ├───amos_0001.nii.gz
│   ├───amos_0002.nii.gz
│   └───...
├───imagesVa/
│   ├───amos_0501.nii.gz
│   └───...
├───labelsVa/
│   ├───amos_0501.nii.gz
│   └───...
└───dataset.json
```

**Important:** After setting up the dataset, you must update the `paths:data_root` field in the `configs/lha_net_config.yaml` file to point to the root of your dataset directory (e.g., `/path/to/amos22/dataset`).

## Configuration

All aspects of the model, training, and data are controlled by the `configs/lha_net_config.yaml` file. Before running the training, review and adjust the following sections:

-   **`paths`:** Set the paths for the dataset, output directories, and checkpoints.
-   **`system`:** Configure the hardware settings, such as the GPU ID, number of workers, and memory constraints.
-   **`training`:** Adjust training parameters like the number of epochs, batch size, and learning rate.
-   **`model`:** Change the model architecture, such as the backbone type or number of channels.

## Training

The main training script is `scripts/dummy_training.py`. This script is designed to run a "dummy" training loop with synthetic data to verify that the entire pipeline is working correctly. To run the actual training on the AMOS22 dataset, you will need to adapt this script or create a new one.

**1. Run the dummy training script:**

Before starting a full training run, it is highly recommended to run the dummy training script to ensure your environment is set up correctly.

```bash
python scripts/dummy_training.py
```

This will test the model creation, forward and backward passes, and the overall training loop with synthetic data that mimics the properties of the AMOS22 dataset.

**2. (TODO) Create a real training script:**

You will need to create a new training script (e.g., `train.py`) that loads the real AMOS22 dataset instead of the synthetic data. This can be done by replacing the `DummyDataset` with a PyTorch `Dataset` that reads the NIfTI files from the AMOS22 dataset directory.

**3. Start the training:**

Once you have a training script for the real dataset, you can run it:

```bash
python train.py --config configs/lha_net_config.yaml
```

## Evaluation

The `evaluation` module contains scripts to evaluate the performance of a trained model. You can use the `SegmentationMetrics` class in `src/evaluation/metrics.py` to compute various metrics, including:

-   Dice Similarity Coefficient (DSC)
-   Hausdorff Distance (95th percentile)
-   Normalized Surface Distance (NSD)
-   Volume Similarity

You will need to write a script that loads a trained model, runs inference on the validation or test set, and computes these metrics.

## Testing

The project includes a suite of tests in the `tests/` directory. To run all tests, use the `run_all_tests.py` script:

```bash
python scripts/run_all_tests.py
```

This will run unit tests for the data pipeline, loss functions, metrics, and model components.

## Directory Structure

```
lha-net/
├───configs/              # Configuration files (e.g., lha_net_config.yaml)
├───scripts/              # Training and utility scripts
├───src/                  # Source code
│   ├───data/             # Data loading, preprocessing, and augmentation
│   ├───evaluation/       # Evaluation metrics and scripts
│   ├───losses/           # Loss functions
│   ├───models/           # Model architecture (LHA-Net)
│   ├───training/         # Training loop, optimizer, scheduler
│   └───utils/            # Utility functions
├───tests/                # Unit tests
├───requirements.txt      # Python package dependencies
└───README.md             # This file
```
