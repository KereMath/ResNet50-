# VAE + Classifier for 9-Class Anomaly Classification

## Overview
**VAE Classifier** combines:
1. **Variational Autoencoder (VAE)** - Learns compressed representations
2. **Classification Head** - Predicts 9 anomaly types

This model performs **both reconstruction and classification** simultaneously.

---

## Architecture

```
Input Image (224x224x3)
    ↓
┌─────────────────┐
│    ENCODER      │ Conv layers
└─────────────────┘
    ↓
Latent Space (mu, logvar) - 128 dimensions
    ↓
    ├──────────────────────────┐
    ↓                          ↓
┌─────────────────┐    ┌──────────────────┐
│    DECODER      │    │ CLASSIFIER HEAD  │
│ Reconstruct IMG │    │   9 Classes      │
└─────────────────┘    └──────────────────┘
    ↓                          ↓
Reconstruction          Class Prediction
(224x224x3)            (9 probabilities)
```

---

## Loss Function

**Total Loss = Reconstruction Loss + KL Divergence + Classification Loss**

```python
total_loss = recon_weight * MSE(original, reconstructed)
           + kl_weight * KL(N(mu, sigma) || N(0, 1))
           + class_weight * CrossEntropy(predicted, true_label)
```

- **Reconstruction Loss**: How well it recreates the image
- **KL Divergence**: Keeps latent space smooth and regular
- **Classification Loss**: How well it predicts the correct class

---

## vs Other Approaches

| Model | Reconstruction | Classification | Unsupervised |
|-------|---------------|----------------|--------------|
| **Pure VAE** | ✅ Yes | ❌ No (only anomaly score) | ✅ Yes |
| **ResNet50** | ❌ No | ✅ Yes (9 classes) | ❌ No |
| **VAE + Classifier** | ✅ Yes | ✅ Yes (9 classes) | ❌ No |

---

## Files

```
vae/
├── vae_classifier.py              # VAE + Classifier architecture
├── data_loader_supervised.py      # Supervised data loading (with labels)
├── train_vae_classifier.py        # Training script
├── inference_vae_classifier.py    # Prediction script
├── README_CLASSIFIER.md           # This file
├── vae_models/
│   └── best_vae_classifier.pth    # Trained model
└── results/
    ├── vae_classifier_results.json
    ├── vae_classifier_history.json
    └── vae_classifier_predictions/
```

---

## Usage

### 1. Train VAE Classifier

```bash
cd C:\Users\user\Desktop\STATIONARY\generateddataimagestraining\vae
python train_vae_classifier.py
```

**Training Configuration:**
- Epochs: 30 (adjustable in config.py)
- Batch Size: 32
- Learning Rate: 0.0001
- Latent Dimension: 128
- Loss Weights: recon=1.0, kl=1.0, class=1.0

**Expected Output:**
```
======================================================================
  VAE + CLASSIFIER TRAINING
  9-Class Time Series Anomaly Classification
======================================================================

Epoch 25/30
----------------------------------------------------------------------
Train - Loss: 1234.56 (Recon: 1200.00, KL: 30.00, Class: 0.4567), Acc: 76.50%
Test  - Loss: 1256.78 (Recon: 1220.00, KL: 32.00, Class: 0.4789), Acc: 74.20%
Test  - F1: 0.7350, Precision: 0.7450, Recall: 0.7250
>>> NEW BEST MODEL! F1: 0.7350, Acc: 74.20%
```

---

### 2. Inference - Single Image

```python
from inference_vae_classifier import load_vae_classifier, predict_single_image
import config

device = config.DEVICE

# Load model
model, class_names = load_vae_classifier(
    'vae_models/best_vae_classifier.pth',
    device
)

# Predict single image
predicted_class, class_probs, reconstruction = predict_single_image(
    image_path='path/to/image.png',
    model=model,
    class_names=class_names,
    device=device,
    save_visualization='result.png'  # Saves visualization
)

print(f"Predicted Class: {predicted_class}")
print(f"Confidence: {class_probs[predicted_class]*100:.1f}%")

# Top 3 predictions
for i, (cls, prob) in enumerate(list(class_probs.items())[:3]):
    print(f"{i+1}. {cls}: {prob*100:.1f}%")
```

---

### 3. Inference - Batch Prediction

```python
from inference_vae_classifier import batch_predict

results = batch_predict(
    image_dir='path/to/images/',
    model=model,
    class_names=class_names,
    device=device,
    output_dir='results/predictions/'
)

# Results: [(image_name, predicted_class, confidence), ...]
```

---

### 4. Inference - Direct from CSV

```python
from inference_vae_classifier import predict_from_csv

predicted_class, class_probs = predict_from_csv(
    csv_path='path/to/timeseries.csv',
    model=model,
    class_names=class_names,
    device=device,
    save_plot=True  # Saves generated plot
)

print(f"Predicted: {predicted_class}")
print(f"Confidence: {class_probs[predicted_class]*100:.1f}%")
```

---

## Output Example

### Prediction Result:
```
Image: anomaly_sample_001.png
Predicted: mean_shift

Top 3 probabilities:
  1. mean_shift              : 87.3%
  2. trend_shift             : 8.2%
  3. variance_shift          : 2.1%
```

### Visualization Saved:
- **Left**: Original image
- **Middle**: Reconstructed image
- **Right**: Class probabilities (bar chart)

---

## Expected Performance

Based on similar architecture and dataset size:

| Metric | Expected Range |
|--------|---------------|
| **Accuracy** | 72-78% |
| **F1-Score** | 0.70-0.76 |
| **Precision** | 0.72-0.78 |
| **Recall** | 0.70-0.75 |

**Note:** May be similar to or slightly better than ResNet50 (74.29%) due to:
- ✅ Richer latent representations from VAE
- ✅ Reconstruction task acts as regularization
- ❌ More complex optimization (3 losses)

---

## Advantages over ResNet50

### 1. **Reconstruction Ability**
- Can visualize what model "sees"
- Helps debug misclassifications
- Useful for explainability

### 2. **Latent Space**
- Compressed 128-dim representation
- Can cluster similar anomalies
- Enables anomaly detection + classification

### 3. **Regularization**
- Reconstruction task prevents overfitting
- KL divergence ensures smooth latent space

### 4. **Flexibility**
- Can use latent features for other tasks
- Can generate synthetic anomaly images
- Can interpolate between anomaly types

---

## Disadvantages

### 1. **Training Time**
- Slower than ResNet50 (3 losses to optimize)
- More epochs needed for convergence

### 2. **Complexity**
- More hyperparameters to tune
- Harder to debug

### 3. **Memory**
- Larger model (encoder + decoder + classifier)
- Needs more GPU memory

---

## Hyperparameter Tuning

### Loss Weights

```python
# config.py
RECONSTRUCTION_WEIGHT = 1.0  # Increase if reconstruction is poor
KL_WEIGHT = 1.0              # Increase for smoother latent space
# Classification weight is always 1.0
```

**Recommendations:**
- If classification accuracy low → Increase class weight (try 2.0)
- If reconstruction poor → Increase recon weight (try 2.0)
- If latent space collapse → Increase KL weight (try 1.5)

### Latent Dimension

```python
LATENT_DIM = 128  # Default
```

- **Smaller (64)**: Faster, more compression, may lose info
- **Larger (256)**: Better representation, slower training

### Learning Rate

```python
LEARNING_RATE = 0.0001  # Default (conservative)
```

- If training slow → Try 0.0003
- If unstable → Try 0.00005

---

## Comparison Table

| Feature | Pure VAE | ResNet50 | VAE Classifier |
|---------|----------|----------|----------------|
| **Training Time** | Fast | Medium | Slow |
| **Accuracy** | N/A | 74.29% | ~72-78% |
| **9 Classes** | ❌ | ✅ | ✅ |
| **Reconstruction** | ✅ | ❌ | ✅ |
| **Latent Features** | ✅ | ❌ | ✅ |
| **Explainability** | Medium | Low | High |
| **Memory Usage** | Medium | Low | High |

---

## When to Use?

### Use VAE Classifier If:
- ✅ You want both classification AND reconstruction
- ✅ You need latent representations for clustering/analysis
- ✅ You want explainable predictions (via reconstruction)
- ✅ You have enough GPU memory and training time

### Use ResNet50 If:
- ✅ You only need classification (no reconstruction)
- ✅ You want faster training
- ✅ You have limited GPU memory
- ✅ 74% accuracy is sufficient

### Use Pure VAE If:
- ✅ You don't have labels (unsupervised)
- ✅ You only need anomaly detection (not classification)
- ✅ You want to detect novel/unknown anomalies

---

## Troubleshooting

### Classification Accuracy Low
1. Increase classification weight: `class_weight=2.0`
2. Train longer (50 epochs)
3. Reduce reconstruction weight: `recon_weight=0.5`

### Reconstruction Quality Poor
1. Increase reconstruction weight: `recon_weight=2.0`
2. Increase latent dimension: `LATENT_DIM=256`
3. Decrease KL weight: `kl_weight=0.5`

### Training Unstable
1. Lower learning rate: `LEARNING_RATE=0.00005`
2. Increase batch size: `BATCH_SIZE=64`
3. Add gradient clipping in training script

---

## Next Steps

1. **Train the model**: `python train_vae_classifier.py`
2. **Evaluate results**: Check `results/vae_classifier_results.json`
3. **Compare with ResNet50**: See which performs better
4. **Use for inference**: Load best model and predict new images

---

## Notes

- Model combines best of both worlds: VAE's representation learning + supervised classification
- Useful when you need both reconstruction AND class labels
- More complex than pure ResNet50 but potentially more powerful
- Good for research and when explainability matters
