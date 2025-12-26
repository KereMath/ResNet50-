"""
VAE Configuration for Anomaly Detection
Unsupervised learning approach - uses only visual plots
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
GENERATED_DATA_DIR = BASE_DIR.parent / "Generated Data"
OUTPUT_DIR = Path(__file__).parent

# Use the same generated plots from the classifier training
PLOTS_DIR = BASE_DIR / "generated_plots"
TRAIN_PLOTS_DIR = PLOTS_DIR / "train"
TEST_PLOTS_DIR = PLOTS_DIR / "test"

# Output directories for VAE
MODELS_DIR = OUTPUT_DIR / "vae_models"
RECONSTRUCTIONS_DIR = OUTPUT_DIR / "reconstructions"
RESULTS_DIR = OUTPUT_DIR / "results"

# Create directories
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RECONSTRUCTIONS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 9 classes (for reference, but VAE is unsupervised)
CLASSES = [
    'collective_anomaly',
    'contextual_anomaly',
    'deterministic_trend',
    'mean_shift',
    'point_anomaly',
    'stochastic_trend',
    'trend_shift',
    'variance_shift',
    'volatility'
]

# VAE Architecture
IMAGE_SIZE = 224
LATENT_DIM = 128  # Latent space dimension
CHANNELS = 3  # RGB images

# Encoder: 224x224x3 -> 128 latent
# Decoder: 128 latent -> 224x224x3 reconstruction

# Training hyperparameters
NUM_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
BETA = 1.0  # KL divergence weight (standard VAE)

# Loss weights
RECONSTRUCTION_WEIGHT = 1.0
KL_WEIGHT = BETA

# Device
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# DataLoader
NUM_WORKERS = 0  # Windows compatibility

# Anomaly detection threshold
# Will be computed from training data reconstruction errors
ANOMALY_THRESHOLD_PERCENTILE = 95  # Top 5% reconstruction errors = anomaly
