# LHA-Net Training, Testing, and Inference Guide

This guide explains how to use the LHA-Net training, testing, and inference scripts with the YAML configuration file.

## Table of Contents

1. [Configuration Setup](#configuration-setup)
2. [Training](#training)
3. [Testing/Evaluation](#testingevaluation)
4. [Inference](#inference)
5. [Configuration Options](#configuration-options)
6. [Examples](#examples)

---

## Configuration Setup

All scripts use the YAML configuration file located at `configs/lha_net_config.yaml`. Before running any script, ensure your configuration is properly set up.

### Key Configuration Sections

```yaml
# Model architecture settings
model:
  name: "lha_net"
  in_channels: 1
  num_classes: 16
  base_channels: 32

# Training parameters
training:
  num_epochs: 10
  batch_size: 1
  learning_rate: 0.001

# Data paths (IMPORTANT: Update these!)
paths:
  data_root: "/path/to/your/amos22/data"
  output_dir: "/path/to/output"
  checkpoint_dir: "./checkpoints"
```

**Important**: Update the `paths` section with your actual data locations before running any scripts!

---

## Training

### Basic Training

Train the model using the default configuration:

```bash
python train.py --config configs/lha_net_config.yaml
```

### Training Options

```bash
python train.py [OPTIONS]

Options:
  --config PATH       Path to config file (default: configs/lha_net_config.yaml)
```

### What Training Does

1. **Loads configuration** from YAML file
2. **Creates model** based on model config
3. **Loads datasets** from specified data paths
4. **Initializes optimizer and scheduler** based on training config
5. **Trains for specified epochs** with:
   - Mixed precision training (if enabled)
   - Gradient accumulation
   - Learning rate scheduling
   - Memory optimization
6. **Validates periodically** (controlled by `validation_freq`)
7. **Saves checkpoints**:
   - `latest_checkpoint.pth` - After each validation
   - `best_checkpoint.pth` - When validation Dice improves

### Training Output

The training process creates:

```
output_dir/
├── checkpoints/
│   ├── best_checkpoint.pth
│   └── latest_checkpoint.pth
└── logs/
    └── training_log.txt
```

### Monitoring Training

Training outputs:
- **Epoch summary**: Loss components, Dice scores
- **Per-organ metrics**: Individual organ segmentation performance
- **Learning rate**: Current learning rate
- **GPU memory**: Memory usage (if CUDA available)

### Configuration Options for Training

Key training parameters in `configs/lha_net_config.yaml`:

```yaml
training:
  num_epochs: 10                    # Total training epochs
  batch_size: 1                     # Physical batch size
  effective_batch_size: 16          # Via gradient accumulation
  learning_rate: 0.001              # Initial learning rate

  # Validation frequency
  validation_freq: 1                # Validate every N epochs
  detailed_metrics_freq: 5          # Compute HD95/NSD every N epochs
  save_freq: 5                      # Save checkpoint every N epochs

  # Optimization
  optimizer:
    type: "adamw"
    weight_decay: 0.0001
    differential_lr: true           # Different LR for backbone
    backbone_lr_factor: 0.1

  scheduler:
    type: "cosine_warmup"
    warmup_epochs: 5
    min_lr: 0.000001

  # Mixed precision
  mixed_precision:
    enabled: true

  # Gradient settings
  gradient_accumulation_steps: 8
  max_grad_norm: 1.0
```

---

## Testing/Evaluation

### Basic Evaluation

Evaluate a trained model on validation set:

```bash
python test.py \
  --config configs/lha_net_config.yaml \
  --checkpoint checkpoints/best_checkpoint.pth
```

### Evaluation Options

```bash
python test.py [OPTIONS]

Options:
  --config PATH              Path to config file
  --checkpoint PATH          Path to model checkpoint (required)
  --split {train,val,test}   Dataset split to evaluate (default: val)
  --output-dir PATH          Output directory for results
  --save-predictions         Save prediction masks as .npz files
  --single-case PATH         Evaluate single NIfTI file
```

### Examples

**Evaluate on test set:**
```bash
python test.py \
  --config configs/lha_net_config.yaml \
  --checkpoint checkpoints/best_checkpoint.pth \
  --split test \
  --output-dir results/test_evaluation
```

**Evaluate with saved predictions:**
```bash
python test.py \
  --config configs/lha_net_config.yaml \
  --checkpoint checkpoints/best_checkpoint.pth \
  --save-predictions
```

**Evaluate single case:**
```bash
python test.py \
  --config configs/lha_net_config.yaml \
  --checkpoint checkpoints/best_checkpoint.pth \
  --single-case /path/to/image.nii.gz
```

### Evaluation Output

The evaluation process creates:

```
output_dir/evaluation/
├── evaluation_summary_20250101_120000.json    # Overall metrics
├── detailed_results_20250101_120000.csv       # Per-case results
└── predictions_20250101_120000.npz            # Prediction masks (optional)
```

### Metrics Computed

1. **Dice Score**: Overlap between prediction and ground truth
2. **Hausdorff Distance 95 (HD95)**: Surface distance metric (mm)
3. **Normalized Surface Distance (NSD)**: Surface agreement metric
4. **Volume Similarity**: Volume agreement between prediction and ground truth

### Evaluation Output Format

**Console output:**
```
EVALUATION RESULTS
================================================================================

OVERALL METRICS (n=20 cases):
  Mean Dice Score: 0.8234 ± 0.0456
  Median Dice Score: 0.8301
  Range: [0.7123, 0.9012]

ORGAN GROUP METRICS:
  SMALL organs: 0.7456 ± 0.0821
    Organs: gallbladder, duodenum
  MEDIUM organs: 0.8123 ± 0.0534
    Organs: right_adrenal, left_adrenal

PER-ORGAN METRICS:
  Organ                Dice            HD95 (mm)       NSD
  -------------------- --------------- --------------- ---------------
  liver                0.9234±0.0234   5.23±1.45       0.9123±0.0234
  spleen               0.8934±0.0345   6.12±2.34       0.8812±0.0412
  ...
```

**JSON summary (evaluation_summary_*.json):**
```json
{
  "num_cases": 20,
  "overall_metrics": {
    "mean_dice": 0.8234,
    "std_dice": 0.0456,
    "median_dice": 0.8301
  },
  "per_organ_metrics": {
    "liver": {
      "dice_mean": 0.9234,
      "dice_std": 0.0234,
      "hd95_mean": 5.23,
      "nsd_mean": 0.9123
    }
  }
}
```

**CSV detailed results (detailed_results_*.csv):**
```csv
case_id,mean_dice,median_dice,dice_liver,dice_spleen,...
case_001,0.8234,0.8301,0.9234,0.8934,...
case_002,0.8145,0.8256,0.9123,0.8823,...
```

---

## Inference

Run predictions on new medical images (without ground truth labels).

### Basic Inference

**Single file:**
```bash
python inference.py \
  --config configs/lha_net_config.yaml \
  --checkpoint checkpoints/best_checkpoint.pth \
  --input /path/to/image.nii.gz \
  --output /path/to/output_segmentation.nii.gz
```

**Batch processing:**
```bash
python inference.py \
  --config configs/lha_net_config.yaml \
  --checkpoint checkpoints/best_checkpoint.pth \
  --input /path/to/images_directory/ \
  --output /path/to/output_directory/ \
  --batch-mode
```

### Inference Options

```bash
python inference.py [OPTIONS]

Options:
  --config PATH           Path to config file
  --checkpoint PATH       Path to model checkpoint (required)
  --input PATH            Input image file or directory (required)
  --output PATH           Output file or directory (required)
  --device {auto,cuda,cpu}  Device to use (default: auto)
  --save-probabilities    Save probability maps
  --batch-mode            Process all .nii/.nii.gz files in directory
```

### Examples

**CPU inference (if no GPU):**
```bash
python inference.py \
  --config configs/lha_net_config.yaml \
  --checkpoint checkpoints/best_checkpoint.pth \
  --input image.nii.gz \
  --output segmentation.nii.gz \
  --device cpu
```

**Save probability maps:**
```bash
python inference.py \
  --config configs/lha_net_config.yaml \
  --checkpoint checkpoints/best_checkpoint.pth \
  --input image.nii.gz \
  --output segmentation.nii.gz \
  --save-probabilities
```

**Process entire directory:**
```bash
python inference.py \
  --config configs/lha_net_config.yaml \
  --checkpoint checkpoints/best_checkpoint.pth \
  --input /data/test_images/ \
  --output /data/test_results/ \
  --batch-mode
```

### Inference Output

For each input image, creates:

```
output_directory/
├── case_001_segmentation.nii.gz    # Segmentation mask
├── case_001_stats.json             # Organ volume statistics
├── case_002_segmentation.nii.gz
├── case_002_stats.json
...
```

**Statistics JSON format:**
```json
{
  "volume_shape": [512, 512, 300],
  "organ_voxel_counts": {
    "background": 45678901,
    "liver": 123456,
    "spleen": 45678,
    "pancreas": 23456,
    ...
  },
  "organ_volumes": {
    "liver": 1234567.89,
    "spleen": 456789.12,
    ...
  }
}
```

### Inference Features

1. **Patch-based processing**: Handles large volumes by processing in patches
2. **Overlapping patches**: Uses overlapping patches for smooth predictions
3. **Gaussian blending**: Smoothly blends patch predictions
4. **Memory efficient**: Processes patches in batches to fit GPU memory
5. **Automatic preprocessing**: Applies same preprocessing as training

---

## Configuration Options

### Complete Configuration Structure

```yaml
# Model Configuration
model:
  name: "lha_net"
  type: "lightweight"              # lightweight, standard, high_capacity
  in_channels: 1                   # Input channels (1 for CT)
  num_classes: 16                  # Number of segmentation classes
  backbone_type: "resnet18"        # resnet18, resnet34
  use_lightweight: true            # Use lightweight backbone
  base_channels: 32                # Base number of channels

  # PMSA (Pyramid Multi-Scale Attention) settings
  pmsa_scales: [0.5, 0.75, 1.0, 1.25, 1.5]
  organ_contexts: ["small", "small", "medium", "medium", "large"]

  # Decoder settings
  use_deep_supervision: true       # Enable deep supervision

  # Memory optimization
  memory_efficient: true
  use_gradient_checkpointing: true

# Training Configuration
training:
  # Basic parameters
  num_epochs: 10
  batch_size: 1                    # Physical batch size
  effective_batch_size: 16         # Effective via accumulation
  learning_rate: 0.001

  # Optimizer settings
  optimizer:
    type: "adamw"                  # adam, adamw, sgd
    weight_decay: 0.0001
    betas: [0.9, 0.999]
    eps: 1e-8
    differential_lr: true          # Different LR for backbone
    backbone_lr_factor: 0.1        # Backbone LR multiplier

  # Scheduler settings
  scheduler:
    type: "cosine_warmup"          # cosine_warmup, step, exponential
    warmup_epochs: 5
    min_lr: 1e-6

  # Mixed precision training
  mixed_precision:
    enabled: true
    init_scale: 65536.0
    growth_factor: 2.0
    backoff_factor: 0.5

  # Gradient settings
  gradient_accumulation_steps: 8
  max_grad_norm: 1.0

  # Memory management
  memory_optimization:
    enable_checkpointing: true
    empty_cache_freq: 10           # Clear cache every N batches
    pin_memory: true

  # Validation and saving
  validation_freq: 1               # Validate every N epochs
  detailed_metrics_freq: 5         # Compute HD95/NSD every N epochs
  save_freq: 5                     # Save checkpoint every N epochs

# Loss Function Configuration
loss:
  primary_loss_weight: 1.0
  deep_supervision_weight: 0.4
  size_prediction_weight: 0.1
  routing_weight: 0.05

  # Combo loss component weights
  focal_weight: 0.5
  dice_weight: 0.3
  size_weight: 0.2

  # Focal loss settings
  focal_loss:
    gamma: 2.0
    adaptive_gamma: true
    size_aware_alpha: true
    organ_size_weights:
      small: 3.0
      medium: 2.0
      large: 1.0
      background: 0.1

# Data Configuration
data:
  # Patch sampling
  patch_size: [48, 96, 96]         # [D, H, W]
  overlap_ratio: 0.5               # For inference
  patches_per_volume: 16

  # Smart sampling strategy
  smart_sampling:
    small_organ_bias: 0.7          # Bias towards small organs
    boundary_bias: 0.2             # Bias towards boundaries
    random_bias: 0.1
    min_organ_voxels: 10

  # Preprocessing
  preprocessing:
    target_spacing: [1.5, 1.5, 1.5]  # mm
    intensity_normalization: "z_score"
    clip_range: [-200, 300]        # HU range for CT

  # Augmentation
  augmentation:
    enabled: false
    rotation_range: [-15, 15]
    scaling_range: [0.9, 1.1]
    random_flip: true

# System Configuration
system:
  device: "cuda"                   # cuda, cpu, auto
  gpu_id: 0
  max_gpu_memory_gb: 24
  num_workers: 4
  pin_memory: true
  log_interval: 50

# Evaluation Configuration
evaluation:
  metrics:
    - "dice"
    - "hausdorff_95"
    - "normalized_surface_distance"
    - "volume_similarity"

  # Organ grouping
  organ_groups:
    small: [9, 12]                 # gallbladder, duodenum
    medium: [7, 8]                 # adrenal glands
    large: [1, 2, 3, 4, 5, 6, 10, 11, 13]

# Paths Configuration
paths:
  data_root: "/path/to/amos22/data"
  output_dir: "/path/to/output"
  checkpoint_dir: "./checkpoints"
  log_dir: "./logs"

# Reproducibility
random_seed: 42
deterministic: false
```

---

## Examples

### Complete Training Workflow

```bash
# 1. Update configuration
vim configs/lha_net_config.yaml
# Update data_root and output_dir paths

# 2. Train model
python train.py --config configs/lha_net_config.yaml

# 3. Evaluate on validation set
python test.py \
  --config configs/lha_net_config.yaml \
  --checkpoint /path/to/output/checkpoints/best_checkpoint.pth \
  --split val

# 4. Evaluate on test set
python test.py \
  --config configs/lha_net_config.yaml \
  --checkpoint /path/to/output/checkpoints/best_checkpoint.pth \
  --split test \
  --save-predictions

# 5. Run inference on new data
python inference.py \
  --config configs/lha_net_config.yaml \
  --checkpoint /path/to/output/checkpoints/best_checkpoint.pth \
  --input /path/to/new_images/ \
  --output /path/to/results/ \
  --batch-mode
```

### Quick Test Run

For testing the pipeline with minimal resources:

```yaml
# configs/lha_net_config.yaml - Quick test settings
training:
  num_epochs: 2
  batch_size: 1
  validation_freq: 1
  detailed_metrics_freq: 1

data:
  patch_size: [32, 64, 64]         # Smaller patches
  patches_per_volume: 4            # Fewer patches

model:
  base_channels: 16                # Fewer channels
```

```bash
python train.py --config configs/lha_net_config.yaml
```

### Resume Training

To resume training from a checkpoint, modify the training script or load the checkpoint:

```python
# In train.py, add resume functionality:
if args.resume:
    checkpoint = torch.load(args.resume)
    trainer.model.load_state_dict(checkpoint['model_state_dict'])
    trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    trainer.current_epoch = checkpoint['epoch']
```

---

## Tips and Best Practices

### Training Tips

1. **Start with small patches**: Use smaller patch sizes initially to fit in memory
2. **Use gradient accumulation**: Increase effective batch size without OOM
3. **Monitor validation metrics**: Watch for overfitting
4. **Save best model**: Based on validation Dice score
5. **Use mixed precision**: Reduces memory usage and speeds up training

### Memory Management

If encountering OOM errors:

1. Reduce `batch_size` (already at 1 in default config)
2. Reduce `patch_size`: `[32, 64, 64]` instead of `[48, 96, 96]`
3. Reduce `base_channels`: 16 or 24 instead of 32
4. Enable gradient checkpointing: Already enabled by default
5. Reduce `num_workers`: Set to 2 or 0

### Inference Tips

1. **Use batch_mode**: For multiple images
2. **Adjust overlap_ratio**: Higher overlap = smoother predictions but slower
3. **Save statistics**: Useful for clinical analysis
4. **Check GPU memory**: Monitor with `nvidia-smi`

---

## Troubleshooting

### Common Issues

**Issue: CUDA Out of Memory**
- Reduce batch_size or patch_size
- Enable memory_efficient mode
- Clear GPU cache: `torch.cuda.empty_cache()`

**Issue: Training too slow**
- Enable mixed precision training
- Reduce num_workers if CPU is bottleneck
- Use smaller detailed_metrics_freq (compute HD95/NSD less often)

**Issue: Poor validation performance**
- Check data preprocessing
- Verify augmentation settings
- Adjust learning rate
- Train for more epochs

**Issue: Config file not found**
- Check file path: `configs/lha_net_config.yaml`
- Use absolute path if relative path fails

---

## Additional Resources

- **AMOS22 Dataset**: https://amos22.grand-challenge.org/
- **Project README**: See `README.md` for architecture details
- **Metrics Guide**: See `METRICS_GUIDE.md` for detailed metrics explanation
- **Quick Start**: See `QUICK_START.md` for getting started

---

## Support

For issues or questions:
1. Check this guide first
2. Review configuration file comments
3. Check existing test scripts in `tests/` directory
4. Review model architecture in `src/models/`

---

**Last Updated**: 2025-10-13
