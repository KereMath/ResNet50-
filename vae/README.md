# VAE-based Anomaly Detection

## Overview
This implementation uses a **Variational Autoencoder (VAE)** for **unsupervised anomaly detection** in time series plot images.

Unlike the supervised ResNet50 classifier (which predicts anomaly types), the VAE learns to reconstruct normal images and detects anomalies through **reconstruction error**.

---

## How It Works

### Training Phase
1. **Input**: All plot images (ignores labels)
2. **VAE learns** to compress images into a latent space and reconstruct them
3. **Normal patterns** are learned through reconstruction
4. **Anomaly threshold** is computed from reconstruction errors on training data

### Inference Phase
1. **Input**: New image
2. **VAE reconstructs** the image
3. **Compute reconstruction error** (MSE between original and reconstructed)
4. **If error > threshold** → **ANOMALY**, else **NORMAL**

---

## VAE Architecture

### Encoder (Image → Latent)
```
Input: 224x224x3
  ↓
Conv2D(3→32) + BN + ReLU → 112x112x32
  ↓
Conv2D(32→64) + BN + ReLU → 56x56x64
  ↓
Conv2D(64→128) + BN + ReLU → 28x28x128
  ↓
Conv2D(128→256) + BN + ReLU → 14x14x256
  ↓
Conv2D(256→512) + BN + ReLU → 7x7x512
  ↓
Flatten → 25088
  ↓
FC → mu (128), logvar (128)
  ↓
Reparameterization: z = mu + std * epsilon
  ↓
Latent Vector (128-dim)
```

### Decoder (Latent → Image)
```
Latent Vector (128-dim)
  ↓
FC → 7x7x512
  ↓
ConvTranspose2D(512→256) + BN + ReLU → 14x14x256
  ↓
ConvTranspose2D(256→128) + BN + ReLU → 28x28x128
  ↓
ConvTranspose2D(128→64) + BN + ReLU → 56x56x64
  ↓
ConvTranspose2D(64→32) + BN + ReLU → 112x112x32
  ↓
ConvTranspose2D(32→3) + Sigmoid → 224x224x3
  ↓
Reconstructed Image
```

---

## Loss Function

**VAE Loss = Reconstruction Loss + KL Divergence**

```python
# Reconstruction Loss (how well we reconstruct the image)
recon_loss = MSE(original, reconstructed)

# KL Divergence (regularization - keeps latent space smooth)
kl_loss = KL(N(mu, sigma) || N(0, 1))

# Total Loss
total_loss = recon_loss + beta * kl_loss
```

- **Beta = 1.0** (standard VAE)
- For **beta-VAE**, increase beta for more disentangled latent space

---

## Configuration

See [config.py](config.py):

| Parameter | Value | Description |
|-----------|-------|-------------|
| Latent Dim | 128 | Size of latent representation |
| Image Size | 224×224 | Input/output image size |
| Epochs | 50 | Training epochs |
| Batch Size | 32 | Batch size |
| Learning Rate | 0.0001 | Adam optimizer LR |
| Beta | 1.0 | KL divergence weight |
| Anomaly Threshold | 95th percentile | Top 5% errors = anomaly |

---

## Data

Uses the **same pre-generated plot images** from the supervised classifier:

```
generateddataimagestraining/generated_plots/
├── train/  (1781 images)
└── test/   (992 images)
```

**Key Difference**: VAE ignores class labels during training (unsupervised).

---

## Files

```
vae/
├── config.py              # Configuration
├── vae_model.py          # VAE architecture
├── data_loader.py        # Unsupervised data loading
├── train_vae.py          # Training script
├── inference_vae.py      # Anomaly detection inference
├── README.md             # This file
├── vae_models/           # Saved models
│   └── best_vae.pth
├── reconstructions/      # Sample reconstructions during training
│   ├── reconstruction_epoch_005.png
│   ├── reconstruction_epoch_010.png
│   └── ...
└── results/              # Training results
    ├── vae_results.json
    ├── training_history.json
    ├── training_curves.png
    └── test_predictions/
```

---

## Usage

### 1. Train VAE

```bash
cd C:\Users\user\Desktop\STATIONARY\generateddataimagestraining\vae
python train_vae.py
```

**Output:**
- `vae_models/best_vae.pth` - Best model checkpoint
- `reconstructions/` - Sample reconstructions per 5 epochs
- `results/training_curves.png` - Training progress
- `results/vae_results.json` - Final metrics and threshold

### 2. Inference (Anomaly Detection)

```python
from inference_vae import load_vae_model, load_anomaly_threshold, predict_anomaly
import config

device = config.DEVICE

# Load model and threshold
model = load_vae_model('vae_models/best_vae.pth', device)
threshold = load_anomaly_threshold('results/vae_results.json')

# Predict on single image
is_anomaly, error = predict_anomaly(
    image_path='path/to/image.png',
    model=model,
    threshold=threshold,
    device=device,
    save_comparison='result.png'  # Saves original, reconstructed, difference
)

print(f"Anomaly: {is_anomaly}, Error: {error:.6f}")
```

### 3. Batch Prediction

```python
from inference_vae import batch_predict

results = batch_predict(
    image_dir='path/to/images/',
    model=model,
    threshold=threshold,
    device=device,
    output_dir='results/predictions/'
)
```

---

## Expected Results

### Training
- **Reconstruction Loss**: Should decrease smoothly
- **KL Divergence**: Should stabilize around 20-50
- **Total Loss**: Should converge after 30-40 epochs

### Anomaly Detection
- **Threshold**: Typically around 0.001 - 0.01 (depends on data)
- **Detection Rate**: Varies based on threshold percentile
  - 95th percentile → 5% false positive rate
  - 99th percentile → 1% false positive rate

---

## VAE vs Supervised Classifier

| Aspect | VAE (This) | ResNet50 Classifier |
|--------|------------|---------------------|
| **Learning** | Unsupervised | Supervised |
| **Labels** | Not used | Required |
| **Output** | Anomaly score (error) | 9 class probabilities |
| **Detection** | High error = anomaly | Direct classification |
| **Strength** | Finds novel anomalies | Accurate on known types |
| **Weakness** | No anomaly type info | Can't detect new types |
| **Use Case** | General anomaly screening | Specific anomaly diagnosis |

---

## Combining Both Approaches

**Recommended Pipeline:**
1. **VAE (Stage 1)**: Quick anomaly screening
   - Fast reconstruction-based detection
   - Catches both known and unknown anomalies
2. **Classifier (Stage 2)**: Anomaly type identification
   - Only run on images flagged by VAE
   - Classifies into 9 specific types

```python
# Two-stage pipeline
error, _ = vae_predict(image)

if error > vae_threshold:
    # Anomaly detected by VAE
    anomaly_type = classifier_predict(image)
    print(f"Anomaly Type: {anomaly_type}")
else:
    print("Normal (no anomaly)")
```

---

## Hyperparameter Tuning

### Latent Dimension
- **Smaller (64)**: Faster, more compression, higher reconstruction error
- **Larger (256)**: Slower, less compression, lower reconstruction error
- **Current (128)**: Good balance

### Beta (KL Weight)
- **Beta = 0.5**: More focus on reconstruction
- **Beta = 1.0**: Standard VAE (current)
- **Beta = 2.0**: More disentangled latent space

### Threshold Percentile
- **90th**: More sensitive (10% false positives)
- **95th**: Balanced (5% false positives) - **current**
- **99th**: Conservative (1% false positives)

---

## Troubleshooting

### Poor Reconstruction Quality
- Increase latent dimension (128 → 256)
- Decrease beta (1.0 → 0.5)
- Train longer (50 → 100 epochs)

### All Images Flagged as Anomalies
- Lower threshold percentile (95 → 90)
- Check if training data is too limited

### No Anomalies Detected
- Increase threshold percentile (95 → 99)
- VAE might be over-fitting

---

## Future Improvements

- [ ] **Conditional VAE (CVAE)**: Use labels during training
- [ ] **Beta-VAE**: Tune beta for better disentanglement
- [ ] **Attention VAE**: Add attention mechanism
- [ ] **Adversarial VAE**: GAN-like discriminator
- [ ] **Hierarchical VAE**: Multi-scale latent representations

---

## References

- Original VAE Paper: [Auto-Encoding Variational Bayes (Kingma & Welling, 2013)](https://arxiv.org/abs/1312.6114)
- Beta-VAE: [beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl)

---

## Notes

- VAE is **unsupervised** - doesn't know what "anomaly" means
- It learns what "normal" looks like and flags deviations
- Works best when training data is mostly normal
- If training data has many anomalies, they'll be learned as "normal"
- Threshold selection is critical - tune based on your use case
