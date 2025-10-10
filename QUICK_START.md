# Quick Start: Comprehensive Metrics Tracking

## What You'll See Now

### Every Epoch (Fast Validation)
```
Running fast validation (Dice only)...

================================================================================
EPOCH 1/10 SUMMARY
================================================================================

TRAINING METRICS:
  Total Loss: 7.9207
  Loss Components:
    total_loss: 7.9207
    focal_loss: 0.2564
    dice_loss: 0.9951
    size_loss: 0.1234
    routing_loss: 0.0456

VALIDATION METRICS:
  Total Loss: 6.5432
  Mean Dice Score: 0.6234
  Loss Components:
    total_loss: 6.5432
    focal_loss: 0.1987
    dice_loss: 0.8765

PER-ORGAN METRICS (Dice only):
  Organ                Dice         HD95 (mm)       NSD
  -------------------- ------------ --------------- ------------
  liver                0.7234±0.02  N/A             N/A
  right_kidney         0.6891±0.03  N/A             N/A
  spleen               0.7012±0.02  N/A             N/A
  pancreas             0.5234±0.04  N/A             N/A
  aorta                0.6789±0.03  N/A             N/A
  ivc                  0.6456±0.03  N/A             N/A
  right_adrenal        0.4567±0.05  N/A             N/A
  left_adrenal         0.4321±0.05  N/A             N/A
  gallbladder          0.5678±0.04  N/A             N/A
  esophagus            0.6234±0.03  N/A             N/A
  stomach              0.6789±0.03  N/A             N/A
  duodenum             0.5123±0.04  N/A             N/A
  left_kidney          0.6891±0.03  N/A             N/A

Learning Rate: 0.000041
================================================================================
```

### Every 5th Epoch (Detailed Validation)
```
Running detailed validation (Dice + HD95 + NSD)...

================================================================================
EPOCH 5/10 SUMMARY
================================================================================

TRAINING METRICS:
  Total Loss: 4.2345
  Loss Components:
    total_loss: 4.2345
    focal_loss: 0.1234
    dice_loss: 0.5678
    size_loss: 0.0789
    routing_loss: 0.0234

VALIDATION METRICS:
  Total Loss: 3.8765
  Mean Dice Score: 0.7834
  Loss Components:
    total_loss: 3.8765
    focal_loss: 0.0987
    dice_loss: 0.4321

PER-ORGAN METRICS:
  Organ                Dice         HD95 (mm)       NSD
  -------------------- ------------ --------------- ------------
  liver                0.8523±0.02  12.34±3.21      0.9234±0.01
  right_kidney         0.8201±0.03  15.67±4.52      0.8956±0.02
  spleen               0.8401±0.02  13.89±3.78      0.9102±0.02
  pancreas             0.6789±0.04  25.43±6.12      0.7654±0.03
  aorta                0.8123±0.03  18.23±5.01      0.8734±0.02
  ivc                  0.7956±0.03  19.45±5.23      0.8567±0.02
  right_adrenal        0.5678±0.05  32.67±8.45      0.6234±0.04
  left_adrenal         0.5432±0.05  34.12±9.01      0.6012±0.04
  gallbladder          0.7234±0.04  21.34±6.34      0.8123±0.03
  esophagus            0.7678±0.03  17.89±4.89      0.8456±0.02
  stomach              0.8012±0.03  16.23±4.67      0.8689±0.02
  duodenum             0.6456±0.04  28.67±7.23      0.7234±0.03
  left_kidney          0.8201±0.03  15.67±4.52      0.8956±0.02

Learning Rate: 0.000087
================================================================================
```

## Configuration

Current settings in `configs/lha_net_config.yaml`:

```yaml
training:
  validation_freq: 1              # Validate EVERY epoch
  detailed_metrics_freq: 5        # HD95 & NSD every 5 epochs
  save_freq: 5                    # Save checkpoint every 5 epochs
```

### Adjust Based on Your Needs:

**For faster training:**
```yaml
validation_freq: 2              # Validate every 2 epochs
detailed_metrics_freq: 10       # Detailed metrics every 10 epochs
```

**For more detailed tracking:**
```yaml
validation_freq: 1              # Every epoch
detailed_metrics_freq: 1        # ALL metrics every epoch (SLOW!)
```

**For quick experiments:**
```yaml
validation_freq: 5              # Every 5 epochs
detailed_metrics_freq: 10       # Detailed every 10 epochs
```

## What Metrics Mean

### Dice Score (0-1, higher is better)
- Measures overlap between prediction and ground truth
- **0.9+**: Excellent
- **0.8-0.9**: Very good
- **0.7-0.8**: Good
- **0.5-0.7**: Moderate
- **<0.5**: Poor

### Hausdorff Distance 95 (mm, lower is better)
- Measures maximum surface distance between prediction and ground truth
- **<10mm**: Excellent
- **10-20mm**: Good
- **20-30mm**: Moderate
- **>30mm**: Poor

### Normalized Surface Distance (0-1, higher is better)
- Percentage of surface points within tolerance (1mm)
- **>0.9**: Excellent boundary accuracy
- **0.8-0.9**: Good boundary accuracy
- **0.7-0.8**: Moderate boundary accuracy
- **<0.7**: Poor boundary accuracy

## Performance Tips

### Why Two Validation Modes?

1. **Fast Mode (Dice only)**: ~30 seconds per epoch
   - Quick feedback on training progress
   - Per-organ Dice scores
   - All loss components

2. **Detailed Mode (Dice + HD95 + NSD)**: ~3-5 minutes per epoch
   - Comprehensive quality assessment
   - Surface distance metrics
   - Boundary accuracy metrics

### Recommended Settings

**During initial training (finding hyperparameters):**
```yaml
validation_freq: 2
detailed_metrics_freq: 10
```

**During final training (publication results):**
```yaml
validation_freq: 1
detailed_metrics_freq: 5
```

**Quick debugging:**
```yaml
validation_freq: 5
detailed_metrics_freq: 20
```

## Analyzing Results

After training, analyze your checkpoint:

```bash
# View all metrics
python analyze_metrics.py checkpoints/best_checkpoint.pth

# Export for plotting
python analyze_metrics.py checkpoints/best_checkpoint.pth --export results.json

# Compare multiple runs
python analyze_metrics.py run1.pth run2.pth run3.pth --compare
```

## Common Issues

### "Validation is too slow!"
- Increase `detailed_metrics_freq` to 10 or 20
- HD95 and NSD calculations are the slowest part

### "I want metrics every epoch!"
- Set both `validation_freq: 1` and `detailed_metrics_freq: 1`
- Be prepared for ~3-5 min validation time per epoch

### "Some organs show N/A"
- Normal when HD95/NSD aren't computed (fast mode)
- Also happens if organ is missing from sample

## Summary

You now get:
- ✅ **All loss components** every epoch
- ✅ **Per-organ Dice scores** every epoch (fast)
- ✅ **Detailed metrics (HD95, NSD)** every 5 epochs
- ✅ **Everything saved** in checkpoint files
- ✅ **Beautiful formatted output** in console

Your training will show meaningful metrics from epoch 1 onwards!
