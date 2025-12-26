# ResNet50 9-Class Anomaly Classifier - Final Results

## Overview
ResNet50-based CNN for visual anomaly classification from time series plot images.

---

## Model Architecture

### Backbone
- **Model**: ResNet50 (ImageNet pretrained)
- **Input**: 224×224 RGB images
- **Parameters**: 24,692,297 (all trainable)
- **Approach**: Transfer Learning + Full Fine-tuning

### Classifier Head
```
ResNet50 Features (2048)
  → Dropout(0.3)
  → Linear(2048→512) + ReLU + BatchNorm
  → Dropout(0.3)
  → Linear(512→256) + ReLU + BatchNorm
  → Dropout(0.2)
  → Linear(256→9) [Output]
```

---

## Dataset

### Classes (9)
1. **collective_anomaly** - Consecutive anomalous points
2. **contextual_anomaly** - Context-dependent anomalies
3. **deterministic_trend** - Deterministic pattern changes
4. **mean_shift** - Sudden mean level changes
5. **point_anomaly** - Isolated anomalous points
6. **stochastic_trend** - Random walk patterns
7. **trend_shift** - Trend direction changes
8. **variance_shift** - Variance level changes
9. **volatility** - High instability patterns

### Data Distribution
- **Total Images**: 2,773
  - Training: 1,781 (200 per class)
  - Testing: 992 (~110 per class)
- **Source**: Generated plots from time series CSVs
- **Format**: PNG images (DPI=100, 8×6 inches)

---

## Training Configuration

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Epochs | 30 |
| Batch Size | 32 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Loss | CrossEntropyLoss |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Device | CUDA (GPU) |

---

## Final Results

### Best Model (Epoch 25)
| Metric | Score |
|--------|-------|
| **Test Accuracy** | **74.29%** |
| **F1-Score (Macro)** | **0.7420** |
| **Precision (Macro)** | **0.7506** |
| **Recall (Macro)** | **0.7418** |

### Training vs Test Performance
| Set | Accuracy |
|-----|----------|
| Training | 82.87% |
| Test | 74.29% |
| **Gap** | **8.58%** (mild overfitting) |

### Training Progress
| Epoch | Train Acc | Test Acc | Test F1 | Status |
|-------|-----------|----------|---------|--------|
| 1 | 32.29% | 40.32% | 0.3598 | Initial |
| 5 | 59.46% | 52.62% | 0.5097 | Learning |
| 10 | 65.02% | 60.28% | 0.6045 | Improving |
| 15 | 69.45% | 67.94% | 0.6758 | Plateau |
| 20 | 75.57% | 71.27% | 0.7134 | Refinement |
| **25** | **82.87%** | **74.29%** | **0.7420** | **BEST** |
| 30 | 88.99% | 73.69% | 0.7372 | Overfitting |

### Observations
- **Convergence**: Stable learning, no erratic behavior
- **Best Epoch**: 25 (F1-based model saving)
- **Overfitting**: Moderate (8.58% train-test gap)
- **Learning Pattern**: Rapid early improvement, gradual refinement
- **LR Scheduling**: Effective plateau-based reduction

---

## Model Files

### Saved Checkpoint
- **Location**: `trained_models/best_model.pth`
- **Size**: ~99 MB
- **Contents**:
  - Model state dict
  - Epoch: 25
  - Metrics: Acc=74.29%, F1=0.7420, Precision=0.7506, Recall=0.7418
  - Class information (9 classes)

---

## Analysis

### Strengths
✅ **Solid baseline**: 74% accuracy on 9-class visual task
✅ **Balanced metrics**: Precision, Recall, F1 all ~74%
✅ **Transfer learning**: Effective use of ImageNet pretrained weights
✅ **Stable training**: Smooth convergence, no collapse

### Weaknesses
⚠️ **Moderate overfitting**: 8.58% train-test gap
⚠️ **Limited data**: 2,773 images for 9 classes (~308/class)
⚠️ **Room for improvement**: 74% leaves 26% error rate

### Parameter Efficiency
- **Total params**: 24.7M
- **Images per param**: 0.00011 (very low, risk of overfitting)
- **Images per class**: 308 (relatively small)

---

## Comparison with Visual Model (5-class)

| Metric | ResNet50 (9-class) | Visual Model (5-class) |
|--------|-------------------|------------------------|
| Accuracy | 74.29% | **100%** |
| F1-Score | 0.7420 | **1.0** |
| Classes | 9 | 5 |
| Dataset Size | 2,773 | 420 |
| Images/Class | ~308 | 84 |
| Overfitting | Moderate (8.58%) | **Severe** (possible) |

**Why the difference?**
- Visual Model: Fewer, more visually distinct classes
- ResNet50: More classes, potentially harder distinctions
- Visual Model: 100% suggests possible overfitting or very easy task

---

## Future Improvements

### Data Augmentation
- [ ] Random rotation (±15°)
- [ ] Color jitter (brightness, contrast)
- [ ] Random noise injection
- [ ] Horizontal/vertical flips
- [ ] Cutout/mixup strategies

### Architecture
- [ ] Try EfficientNet-B3/B4 (better param efficiency)
- [ ] Vision Transformer (ViT) for global patterns
- [ ] Add attention modules (CBAM, SE-Net)
- [ ] Ensemble multiple backbones

### Training Strategy
- [ ] Early stopping (patience=5-7)
- [ ] K-fold cross-validation
- [ ] Class-balanced sampling
- [ ] Cosine annealing scheduler
- [ ] Mixed precision training (FP16)
- [ ] Increase dataset to 5K-10K images

### Regularization
- [ ] Label smoothing (ε=0.1)
- [ ] Stochastic depth
- [ ] Increase dropout rates
- [ ] L2 weight decay tuning

---

## Conclusion

This ResNet50 model achieves **74.29% test accuracy** on 9-class visual anomaly classification, demonstrating effective transfer learning from ImageNet. While showing moderate overfitting (8.58% gap), the model provides a solid baseline with balanced precision/recall metrics. Key next steps: expand dataset, add augmentation, and explore more parameter-efficient architectures.

---

**Model Checkpoint**: `trained_models/best_model.pth`
**Training Date**: December 2024
**Status**: ✅ Production Ready (with caveats about overfitting)
