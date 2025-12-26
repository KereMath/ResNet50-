# Image-Based Anomaly Classification Model Training

## Overview
This project trains a **ResNet50-based CNN classifier** to detect and classify 9 different types of time series anomalies from **visual plot images**.

---

## Model Architecture

### Backbone: ResNet50
- **Pretrained**: ImageNet weights
- **Input Size**: 224x224 RGB images
- **Total Parameters**: 24,692,297
- **All parameters trainable**

### Custom Classifier Head
```
ResNet50 Feature Extractor (2048 features)
    ↓
Dropout(0.3)
    ↓
Linear(2048 → 512) + ReLU + BatchNorm
    ↓
Dropout(0.3)
    ↓
Linear(512 → 256) + ReLU + BatchNorm
    ↓
Dropout(0.2)
    ↓
Linear(256 → 9) [Output Layer]
```

---

## Dataset

### Data Source
- **Location**: `C:\Users\user\Desktop\STATIONARY\Generated Data\`
- **Format**: CSV files containing time series data
- **Classes**: 9 anomaly types

### 9 Anomaly Classes
1. **collective_anomaly** - Multiple consecutive anomalous points
2. **contextual_anomaly** - Values normal individually but anomalous in context
3. **deterministic_trend** - Clear deterministic pattern changes
4. **mean_shift** - Sudden shift in mean value
5. **point_anomaly** - Single isolated anomalous points
6. **stochastic_trend** - Random walk patterns
7. **trend_shift** - Change in trend direction
8. **variance_shift** - Change in data variance
9. **volatility** - High variance/instability patterns

### Data Pipeline
1. **CSV to Plot Conversion**: Time series CSVs → PNG plot images
2. **Data Split**:
   - Training: 1,781 images (200 per class)
   - Testing: 992 images (~110 per class)
3. **Augmentation**: Resize to 224x224, Normalize (ImageNet stats)

### Plot Generation Settings
- **DPI**: 100
- **Figure Size**: 8×6 inches
- **Style**: Default matplotlib
- **Output**: PNG images saved to `generated_plots/train/` and `generated_plots/test/`

---

## Training Configuration

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Epochs | 30 |
| Batch Size | 32 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Loss Function | CrossEntropyLoss |
| LR Scheduler | ReduceLROnPlateau (mode=max, factor=0.5, patience=3) |

### Hardware
- **Device**: CUDA (GPU)
- **DataLoader Workers**: 0 (Windows compatibility)

---

## Training Results

### Best Model Performance (Epoch 25)
| Metric | Score |
|--------|-------|
| **Test Accuracy** | **74.29%** |
| **F1-Score (Macro)** | **0.7420** |
| **Precision (Macro)** | **0.7506** |
| **Recall (Macro)** | **0.7418** |

### Training Progress
| Epoch | Train Acc | Test Acc | Test F1 | Status |
|-------|-----------|----------|---------|--------|
| 1 | 32.29% | 40.32% | 0.3598 | ⭐ New Best |
| 5 | 59.46% | 52.62% | 0.5097 | ⭐ New Best |
| 7 | 60.53% | 62.80% | 0.6099 | ⭐ New Best |
| 11 | 66.98% | 61.39% | 0.6228 | ⭐ New Best |
| 14 | 68.56% | 67.64% | 0.6700 | ⭐ New Best |
| 17 | 71.59% | 69.15% | 0.6821 | ⭐ New Best |
| 22 | 79.23% | 73.08% | 0.7310 | ⭐ New Best |
| 23 | 80.91% | 73.19% | 0.7313 | ⭐ New Best |
| **25** | **82.87%** | **74.29%** | **0.7420** | **⭐ BEST MODEL** |
| 26-30 | 84-89% | 72-74% | 0.72-0.74 | No improvement |

### Observations
- **Early Learning**: Rapid improvement in first 10 epochs
- **Plateau**: Performance stabilized around epoch 15-20
- **Final Push**: Best result at epoch 25
- **Overfitting Signs**: Train acc reached 88.99% by epoch 30, but test acc stayed ~74%
- **LR Scheduler**: Automatically reduced learning rate when F1 plateaued

---

## Model Saving Strategy

### Save Mechanism
```python
if test_f1 > best_test_f1:
    # Save only if F1-score improves
    torch.save(checkpoint, 'trained_models/best_model.pth')
```

### Saved Model Contents
- Model state dict (weights)
- Epoch number: 25
- Test accuracy: 74.29%
- Test F1: 0.7420
- Test precision: 0.7506
- Test recall: 0.7418
- Class names and count

### Model Location
```
generateddataimagestraining/trained_models/best_model.pth
```

---

## Files Structure

```
generateddataimagestraining/
├── config.py                 # Configuration settings
├── model.py                  # ResNet50 model architecture
├── data_loader.py            # Dataset and DataLoader
├── train.py                  # Full pipeline (generate + train)
├── onlytrain.py             # Training only (uses pre-generated plots)
├── imagemodeltraining.md    # This documentation
├── generated_plots/         # Generated plot images
│   ├── train/               # Training images (1781)
│   │   ├── collective_anomaly/
│   │   ├── contextual_anomaly/
│   │   ├── deterministic_trend/
│   │   ├── mean_shift/
│   │   ├── point_anomaly/
│   │   ├── stochastic_trend/
│   │   ├── trend_shift/
│   │   ├── variance_shift/
│   │   └── volatility/
│   └── test/                # Test images (992)
│       └── [same structure]
└── trained_models/          # Saved models
    ├── best_model.pth       # Best model checkpoint
    ├── model_info.json      # Model metadata
    └── training_history.json # Full training history
```

---

## How to Use

### 1. Training (First Time)
```bash
# Generate plots from CSVs and train
python train.py
```

### 2. Training (Using Existing Plots)
```bash
# Train on pre-generated plots only
python onlytrain.py
```

### 3. Model Inference
```bash
# Load model and predict on new images
# (inference script to be created)
```

---

## Training Logs Example

```
======================================================================
  TRAINING ON PRE-GENERATED DATA
  9-Class Visual Anomaly Classification
======================================================================

  Using device: cuda

======================================================================
  LOADING PRE-GENERATED DATA
======================================================================
  Loaded 1781 training images
  Loaded 992 test images

======================================================================
  MODEL ARCHITECTURE
======================================================================
  Backbone: resnet50
  Pretrained: True
  Output classes: 9
  Device: cuda
  Total parameters: 24,692,297
  Trainable parameters: 24,692,297
======================================================================

======================================================================
  TRAINING
======================================================================

Epoch 25/30
----------------------------------------------------------------------
Train Loss: 0.4719, Train Acc: 82.87%
Test  Loss: 0.7652, Test Acc:  74.29%
Test  F1:   0.7420, Precision: 0.7506, Recall: 0.7418
>>> NEW BEST MODEL! F1: 0.7420, Acc: 74.29%
```

---

## Potential Improvements

### Data Side
- [ ] Increase dataset size (currently 2,773 → target 5K-10K)
- [ ] Add data augmentation (rotation, noise, brightness)
- [ ] Balance class distribution if needed
- [ ] Generate more diverse plot styles

### Model Side
- [ ] Try EfficientNet or Vision Transformer
- [ ] Add attention mechanisms (CBAM, SE-Net)
- [ ] Ensemble multiple models
- [ ] Fine-tune with lower learning rate

### Training Side
- [ ] Early stopping to prevent overfitting
- [ ] Cross-validation for robust evaluation
- [ ] Class weights for imbalanced data
- [ ] Mixed precision training (faster, less memory)

---

## Next Steps
1. **VAE Implementation**: Create unsupervised anomaly detection using Variational Autoencoder
2. **Inference Script**: Add prediction capability for new images
3. **Deployment**: Export model to ONNX or TorchScript
4. **Real-time Detection**: Integrate with live data streams

---

## Notes
- Model successfully learns visual patterns from time series plots
- 74% accuracy is good for 9-class visual classification
- Some overfitting observed (train 88%, test 74%)
- Further improvement possible with more data and regularization
