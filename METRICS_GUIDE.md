# Comprehensive Metrics Tracking Guide

This guide explains the comprehensive metrics tracking system implemented in the LHA-Net training pipeline.

## Overview

The training system now tracks:
- **All loss components** (total, focal, dice, size, routing) for each epoch
- **Per-organ metrics** (Dice, HD95, NSD) for each validation epoch
- **Complete training history** stored in checkpoint files
- **Detailed console output** showing all metrics during training

## Metrics Tracked

### Training Metrics (Every Epoch)
1. **Total Loss**: Combined loss value
2. **Focal Loss**: Adaptive focal loss component
3. **Dice Loss**: Dice coefficient loss component
4. **Size Loss**: Size-aware loss component
5. **Routing Loss**: Expert routing loss component

### Validation Metrics (Every Validation Epoch)
1. **Loss Components**: Same as training
2. **Mean Dice Score**: Average across all organs (excluding background)
3. **Per-Organ Metrics**:
   - **Dice Score** (mean ± std): Overlap metric
   - **Hausdorff Distance 95** (mean ± std): Surface distance metric in mm
   - **Normalized Surface Distance** (mean ± std): Surface accuracy metric

### Organs Tracked
The system tracks metrics for all 16 classes:
- background
- liver
- right_kidney
- spleen
- pancreas
- aorta
- ivc (inferior vena cava)
- right_adrenal
- left_adrenal
- gallbladder
- esophagus
- stomach
- duodenum
- left_kidney
- class_14
- class_15

## Training Output

### During Training
Each epoch shows:
```
================================================================================
EPOCH X/Y SUMMARY
================================================================================

TRAINING METRICS:
  Total Loss: 0.XXXX
  Loss Components:
    total_loss: 0.XXXX
    focal_loss: 0.XXXX
    dice_loss: 0.XXXX
    size_loss: 0.XXXX
    routing_loss: 0.XXXX

VALIDATION METRICS:  (every N epochs based on validation_freq)
  Total Loss: 0.XXXX
  Mean Dice Score: 0.XXXX
  Loss Components:
    ...

PER-ORGAN METRICS:
  Organ                Dice         HD95 (mm)       NSD
  -------------------- ------------ --------------- ------------
  liver                0.XXXX±0.XX  XX.XX±XX.XX     0.XXXX±0.XX
  right_kidney         0.XXXX±0.XX  XX.XX±XX.XX     0.XXXX±0.XX
  ...

Learning Rate: 0.XXXXXX
================================================================================
```

### Final Summary
At the end of training:
```
================================================================================
TRAINING SUMMARY
================================================================================

Training Loss Progression:
  Epoch 1: 0.XXXX
  Epoch 2: 0.XXXX
  ...

Validation Metrics Progression:
  Epoch 5: Loss=0.XXXX, Dice=0.XXXX
  Epoch 10: Loss=0.XXXX, Dice=0.XXXX
  ...

Best Performance (Epoch X):
  Mean Dice Score: 0.XXXX

  Best Per-Organ Dice Scores:
    liver               : 0.XXXX±0.XXXX
    spleen              : 0.XXXX±0.XXXX
    ...
================================================================================
```

## Checkpoint Files

All checkpoint files (.pth) now contain:

```python
{
    'epoch': current_epoch,
    'model_state_dict': ...,
    'optimizer_state_dict': ...,
    'scheduler_state_dict': ...,
    'best_dice': float,
    'config': dict,

    # NEW: Comprehensive metrics history
    'metrics_history': {
        'train': {
            'epochs': [1, 2, 3, ...],
            'losses': [0.XXX, 0.XXX, ...],
            'loss_components': [
                {'total_loss': X, 'focal_loss': X, 'dice_loss': X, ...},
                ...
            ]
        },
        'val': {
            'epochs': [5, 10, ...],
            'losses': [0.XXX, 0.XXX, ...],
            'dice_scores': [0.XXX, 0.XXX, ...],
            'per_organ_metrics': [
                {
                    'liver': {'dice_mean': X, 'dice_std': X, 'hd95_mean': X, ...},
                    'spleen': {...},
                    ...
                },
                ...
            ]
        }
    },

    # NEW: Current epoch metrics
    'current_epoch_metrics': {
        'epoch': X,
        'train_loss': X,
        'train_loss_components': {...},
        'val_loss': X,
        'val_dice': X,
        'val_loss_components': {...},
        'organ_metrics': {...}
    }
}
```

## Analyzing Metrics

### Using the Analysis Script

The `analyze_metrics.py` script provides comprehensive analysis of checkpoint metrics.

#### Basic Analysis
```bash
python analyze_metrics.py /path/to/checkpoint.pth
```

This will display:
- Complete training loss history
- Validation metrics for all epochs
- Per-organ metrics for each validation epoch
- Best performance summary

#### Export to JSON
```bash
python analyze_metrics.py /path/to/checkpoint.pth --export metrics.json
```

This exports all metrics to a JSON file for further analysis or plotting.

#### Compare Multiple Checkpoints
```bash
python analyze_metrics.py checkpoint1.pth checkpoint2.pth checkpoint3.pth --compare
```

This compares metrics across multiple checkpoints.

### Programmatic Access

You can also load and analyze metrics in Python:

```python
import torch

# Load checkpoint
checkpoint = torch.load('checkpoint.pth', map_location='cpu')

# Access metrics history
metrics_history = checkpoint['metrics_history']

# Training metrics
train_epochs = metrics_history['train']['epochs']
train_losses = metrics_history['train']['losses']
train_loss_components = metrics_history['train']['loss_components']

# Validation metrics
val_epochs = metrics_history['val']['epochs']
val_dice_scores = metrics_history['val']['dice_scores']
val_organ_metrics = metrics_history['val']['per_organ_metrics']

# Example: Get liver Dice scores across epochs
liver_dice_scores = [
    epoch_metrics['liver']['dice_mean']
    for epoch_metrics in val_organ_metrics
]

# Example: Get best performing organ at epoch 5
epoch_5_metrics = val_organ_metrics[0]  # First validation (epoch 5)
best_organ = max(epoch_5_metrics.items(),
                 key=lambda x: x[1].get('dice_mean', 0))
print(f"Best organ: {best_organ[0]} with Dice: {best_organ[1]['dice_mean']:.4f}")
```

## Configuration

The metrics tracking is controlled by these config parameters:

```yaml
training:
  validation_freq: 5  # Compute detailed metrics every N epochs
  save_freq: 10       # Save checkpoints every N epochs

evaluation:
  metrics:
    - "dice"
    - "hausdorff_95"
    - "normalized_surface_distance"
    - "volume_similarity"

data:
  preprocessing:
    target_spacing: [1.5, 1.5, 1.5]  # Used for distance metrics
```

## Visualization

You can create plots from the exported JSON:

```python
import json
import matplotlib.pyplot as plt

# Load exported metrics
with open('metrics.json', 'r') as f:
    data = json.load(f)

metrics_history = data['metrics_history']

# Plot training loss
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(metrics_history['train']['epochs'],
         metrics_history['train']['losses'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

# Plot validation dice
plt.subplot(1, 3, 2)
plt.plot(metrics_history['val']['epochs'],
         metrics_history['val']['dice_scores'])
plt.xlabel('Epoch')
plt.ylabel('Dice Score')
plt.title('Validation Dice')

# Plot per-organ dice for last validation
plt.subplot(1, 3, 3)
last_epoch_metrics = metrics_history['val']['per_organ_metrics'][-1]
organs = [o for o in last_epoch_metrics.keys() if o != 'background']
dice_scores = [last_epoch_metrics[o]['dice_mean'] for o in organs]
plt.barh(organs, dice_scores)
plt.xlabel('Dice Score')
plt.title('Per-Organ Dice (Last Epoch)')
plt.tight_layout()
plt.savefig('metrics_summary.png')
```

## Best Practices

1. **Monitor All Metrics**: Don't just look at overall Dice - check per-organ metrics to identify which organs are challenging

2. **Check Loss Components**: If total loss is decreasing but Dice isn't improving, check which loss component is dominating

3. **Track HD95 and NSD**: These metrics can reveal issues with boundary accuracy even when Dice is high

4. **Export Regularly**: Use `--export` to save metrics for comparison and plotting

5. **Compare Checkpoints**: Use `--compare` to evaluate different training runs or hyperparameter settings

## Troubleshooting

### Metrics Calculation is Slow
- HD95 and NSD calculations are computationally expensive
- Consider reducing validation frequency if needed
- The metrics are only computed during validation (not training)

### Some Organs Show "N/A" for HD95/NSD
- This happens when an organ is not present in the ground truth or prediction
- It's normal for some samples to not contain all organs

### Checkpoint File is Large
- Storing complete metrics history increases checkpoint size
- This is intentional for comprehensive analysis
- Old checkpoints without metrics can be deleted

## Summary

The comprehensive metrics tracking system provides:
- ✅ Complete training history in checkpoint files
- ✅ Per-organ metrics for all validation epochs
- ✅ All loss components tracked and stored
- ✅ Detailed console output during training
- ✅ Analysis tools for checkpoint evaluation
- ✅ Easy export to JSON for custom analysis

This enables thorough analysis of model performance and helps identify areas for improvement!
