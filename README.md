# Generated Data Image Training

Visual classifier for 9-class anomaly detection using CNN (ResNet50) on generated time series plots.

## Overview

This system:
1. Samples CSVs from Generated Data folder (200 per class for training, 1000 total for testing)
2. Generates time series plots from CSVs
3. Trains ResNet50 to classify anomalies visually from plots
4. Evaluates on 9 classes with comprehensive metrics

## Classes (9 Total)

- collective_anomaly
- contextual_anomaly
- deterministic_trend
- mean_shift
- point_anomaly
- stochastic_trend
- trend_shift
- variance_shift
- volatility

## Quick Start

### 1. Install Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pillow tqdm numpy pandas matplotlib scikit-learn
```

### 2. Train Model

```bash
cd "C:\Users\user\Desktop\STATIONARY\generateddataimagestraining"
python train.py
```

This will:
- Sample 200 CSVs per class for training (1800 total)
- Sample ~111 CSVs per class for testing (1000 total)
- Generate plots from all sampled CSVs
- Train ResNet50 for 30 epochs
- Save best model to `trained_models/best_model.pth`

## Output Files

After training:

1. **trained_models/best_model.pth**: Best model checkpoint
2. **trained_models/model_info.json**: Model metadata and confusion matrix
3. **trained_models/training_history.json**: Training curves
4. **generated_plots/train/**: Training plot images (organized by class)
5. **generated_plots/test/**: Test plot images (organized by class)

## Configuration

Edit [config.py](config.py) to change:

```python
TRAIN_SAMPLES_PER_CLASS = 200  # CSVs per class for training
TEST_SAMPLES_TOTAL = 1000       # Total CSVs for testing
NUM_EPOCHS = 30                 # Training epochs
BATCH_SIZE = 32                 # Batch size
LEARNING_RATE = 0.001           # Learning rate
IMAGE_SIZE = 224                # Input image size
BACKBONE = 'resnet50'           # CNN backbone
```

## Expected Results

- Training time: ~15-30 minutes (GPU), ~1-2 hours (CPU)
- Plot generation: ~5-10 minutes for 2800 plots
- Expected accuracy: 75-95% (9-class classification)

## File Structure

```
generateddataimagestraining/
├── README.md                # This file
├── config.py                # Configuration
├── data_sampler.py          # CSV sampling logic
├── plot_generator.py        # Plot generation from CSVs
├── data_loader.py           # Image data loading
├── model.py                 # ResNet50 architecture
├── train.py                 # Main training script
├── generated_plots/         # Generated plot images
│   ├── train/               # Training plots (by class)
│   └── test/                # Test plots (by class)
└── trained_models/          # Model outputs
    ├── best_model.pth
    ├── model_info.json
    └── training_history.json
```

## Comparison: Generated Data vs Combinations

| Metric | Combinations (Visual Model) | Generated Data (This) |
|--------|-----------------------------|-----------------------|
| **Classes** | 5 classes | 9 classes |
| **Training samples** | 420 (balanced) | 1800 (200 per class) |
| **Test samples** | 84 | 1000 |
| **Source** | Pre-existing plots | Generated from CSVs |
| **Use case** | Existing combinations | Full Generated Data |

## Troubleshooting

### CUDA out of memory

Reduce batch size in [config.py](config.py):
```python
BATCH_SIZE = 16  # or 8
```

### Plot generation errors

Check CSV format - should have 'data' or 'value' column, or use first numerical column.

### Low accuracy

- Increase epochs to 50
- Check plot quality in `generated_plots/`
- Verify class balance in sampling
# ResNet50-
