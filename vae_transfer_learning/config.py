"""
Transfer Learning Configuration
Uses pre-trained VAE encoder + new classifier head
"""
from pathlib import Path
import sys

# Add parent directories to path
parent_dir = Path(__file__).parent.parent
vae_dir = parent_dir / "vae"
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(vae_dir))

# Paths
BASE_DIR = parent_dir
OUTPUT_DIR = Path(__file__).parent

# Use the same generated plots
PLOTS_DIR = BASE_DIR / "generated_plots"
TRAIN_PLOTS_DIR = PLOTS_DIR / "train"
TEST_PLOTS_DIR = PLOTS_DIR / "test"

# Pre-trained VAE model
PRETRAINED_VAE_PATH = vae_dir / "vae_models" / "best_vae.pth"

# Output directories
MODELS_DIR = OUTPUT_DIR / "models"
RESULTS_DIR = OUTPUT_DIR / "results"

# Create directories
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 9 classes
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

# Transfer learning settings
IMAGE_SIZE = 224
LATENT_DIM = 128  # Must match pre-trained VAE
CHANNELS = 3

# Training hyperparameters (faster since only training classifier)
NUM_EPOCHS = 20  # Less epochs needed
BATCH_SIZE = 32
LEARNING_RATE = 0.001  # Higher LR since only classifier trains

# Freeze VAE encoder?
FREEZE_ENCODER = True  # True = only train classifier (fast)
                       # False = fine-tune encoder too (slow)

# Device
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# DataLoader
NUM_WORKERS = 0  # Windows compatibility
