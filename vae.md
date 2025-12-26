# VAE Transfer Learning Results

## Project Overview
This project implements a two-stage approach for time series anomaly classification:
1. **Stage 1: VAE (Variational Autoencoder)** - Unsupervised feature learning
2. **Stage 2: Transfer Learning** - Supervised classification using pre-trained VAE encoder

## Stage 1: VAE Training (Unsupervised)

### Configuration
- **Model**: Variational Autoencoder (VAE)
- **Latent Dimension**: 128
- **Image Size**: 224x224
- **Channels**: 3 (RGB)
- **Training Epochs**: 50
- **Device**: CUDA (GPU)

### Results
- **Best Epoch**: 47/50
- **Test Loss**: 1086.50
  - Reconstruction Loss: 995.65
  - KL Divergence Loss: 90.85
- **Anomaly Threshold**: 0.0124

### Output Files
- Model: `vae/vae_models/best_vae.pth`
- Results: `vae/results/vae_results.json`
- Training History: `vae/results/vae_history.json`
- Visualizations: `vae/results/vae_training_curves.png`

---

## Stage 2: Transfer Learning (Supervised Classification)

### Problem Fixed
**ImportError Issue**: The original `train_transfer.py` was importing from the wrong `data_loader.py` (VAE's version instead of transfer learning's version).

**Solution Applied**:
```python
# Added in train_transfer.py (lines 23-26)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from data_loader import create_dataloaders
```

This ensures the local `data_loader.py` is imported first, preventing conflicts with the VAE directory's data_loader.

### Configuration
- **Pre-trained Model**: VAE encoder (frozen weights)
- **Trainable Parameters**: 67,849 (0.73% of total 9,280,425 params)
- **Strategy**: Frozen encoder + new classifier head
- **Training Epochs**: 20
- **Batch Size**: 32
- **Learning Rate**: 0.001
- **Optimizer**: Adam (classifier only)
- **Scheduler**: ReduceLROnPlateau (monitors F1-score)
- **Classes**: 9 time series anomaly types

### Dataset
- **Training**: 1,781 images (56 batches)
- **Testing**: 992 images (31 batches)

### Final Results (Best Model - Epoch 19/20)

#### Overall Metrics
- **Accuracy**: **65.12%**
- **F1-Score**: **65.24%**
- **Precision**: **66.67%**
- **Recall**: **65.03%**

#### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support | Accuracy |
|-------|-----------|--------|----------|---------|----------|
| collective_anomaly | 0.5185 | 0.5000 | 0.5091 | 108 | 50.0% |
| contextual_anomaly | 0.7704 | **0.9369** | 0.8458 | 111 | **93.7%** ⭐ |
| deterministic_trend | 0.8173 | 0.7658 | 0.7907 | 111 | **76.6%** |
| mean_shift | 0.6392 | 0.5636 | 0.5991 | 110 | 56.4% |
| point_anomaly | 0.3846 | 0.5676 | 0.4591 | 111 | 56.8% |
| stochastic_trend | 0.8085 | 0.6937 | 0.7467 | 111 | 69.4% |
| trend_shift | 0.6480 | **0.7297** | 0.6865 | 111 | **73.0%** |
| variance_shift | 0.7683 | 0.5676 | 0.6528 | 111 | 56.8% |
| volatility | 0.5948 | 0.5278 | 0.5593 | 108 | 52.8% |

#### Training Progress
- **Epoch 1**: 46.47% acc, F1 0.4414
- **Epoch 5**: 57.86% acc, F1 0.5773
- **Epoch 10**: 60.69% acc, F1 0.6060
- **Epoch 15**: 64.01% acc, F1 0.6421
- **Epoch 19**: **65.12% acc, F1 0.6524** ⭐ (Best)
- **Epoch 20**: 65.12% acc, F1 0.6524

#### Loss Progression
- **Training Loss**: 1.865 → 0.725 (61% reduction)
- **Test Loss**: 1.492 → 0.986 (34% reduction)

### Confusion Matrix

```
True / Pred    collecti contextu determin mean_shi point_an stochast trend_sh variance volatili
----------------------------------------------------------------------------------------------------
collective_anomaly       54        2        0        8       37        0        1        2        4
contextual_anomaly        2      104        0        0        5        0        0        0        0
deterministic_trend       0        0       85        3        1       10       12        0        0
mean_shift               13        9        1       62       13        0        6        4        2
point_anomaly            14        5        0        7       63        1        2        4       15
stochastic_trend          1        2        7        4        0       77       19        0        1
trend_shift               1        2       11        9        1        5       81        0        1
variance_shift            6        4        0        3       12        3        3       63       17
volatility                2        7        0        1       31        0        1        9       57
```

### Key Insights

#### Strengths
1. **Excellent Performance**: contextual_anomaly (93.7% accuracy, 93.69% recall)
2. **Strong Performance**: deterministic_trend (76.6%), trend_shift (73.0%)
3. **Fast Training**: Only 20 epochs needed due to frozen encoder strategy

#### Challenges
1. **Confusion Between Point Anomalies**:
   - collective_anomaly ↔ point_anomaly (37+14 = 51 confusions)
   - volatility → point_anomaly (31 confusions)

2. **Moderate Performance Classes**:
   - collective_anomaly: 50.0%
   - volatility: 52.8%
   - mean_shift: 56.4%

3. **Performance Gap**: Achieved 65.12% vs expected 74-80% (README target)

### Possible Improvements
1. **Fine-tune encoder**: Set `FREEZE_ENCODER=False` (slower but potentially better)
2. **More epochs**: Extend training beyond 20 epochs
3. **Data augmentation**: Adjust augmentation strategies
4. **Class balancing**: Address class imbalance issues
5. **Hyperparameter tuning**: Optimize learning rate, batch size, etc.

### Output Files
- Model: `vae_transfer_learning/models/best_transfer_model.pth`
- Results: `vae_transfer_learning/results/transfer_results.json`
- Training History: `vae_transfer_learning/results/transfer_history.json`
- Training Curves: `vae_transfer_learning/results/transfer_training_curves.png`

---

## 9 Anomaly Classes

1. **collective_anomaly**: Group of anomalous points
2. **contextual_anomaly**: Anomalous in specific context
3. **deterministic_trend**: Predictable trend pattern
4. **mean_shift**: Sudden change in average value
5. **point_anomaly**: Single anomalous point
6. **stochastic_trend**: Random trend pattern
7. **trend_shift**: Change in trend direction
8. **variance_shift**: Change in data spread
9. **volatility**: High variability in values

---

## Technical Stack
- **Framework**: PyTorch
- **Data**: Pre-generated time series plot images
- **Metrics**: sklearn (precision, recall, F1-score, confusion matrix)
- **Visualization**: matplotlib
- **Device**: CUDA (GPU acceleration)

---

## Repository
GitHub: https://github.com/KereMath/ResNet50-.git

---

**Training Date**: December 2025
**Status**: ✓ Completed
