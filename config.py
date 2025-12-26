"""
Generated Data Image Training - Configuration
Generates plots from CSV files and trains visual classifier
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
GENERATED_DATA_DIR = BASE_DIR / "Generated Data"
OUTPUT_DIR = Path(__file__).parent

# Output directories
PLOTS_DIR = OUTPUT_DIR / "generated_plots"
TRAIN_PLOTS_DIR = PLOTS_DIR / "train"
TEST_PLOTS_DIR = PLOTS_DIR / "test"
MODELS_DIR = OUTPUT_DIR / "trained_models"

# 9 classes (all from Generated Data)
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

# Folder name mapping (handle case differences)
CLASS_FOLDERS = {
    'collective_anomaly': 'collective_anomaly',
    'contextual_anomaly': 'contextual_anomaly',
    'deterministic_trend': 'deterministic_trend',
    'mean_shift': 'mean_shift',
    'point_anomaly': 'point_anomaly',
    'stochastic_trend': 'Stochastic Trend',  # Note: capital letters in folder name
    'trend_shift': 'trend_shift',
    'variance_shift': 'variance_shift',
    'volatility': 'Volatility'  # Note: capital letter in folder name
}

# Class to index mapping
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: cls for idx, cls in enumerate(CLASSES)}

# Data sampling
TRAIN_SAMPLES_PER_CLASS = 200  # 200 CSVs per class for training
TEST_SAMPLES_TOTAL = 1000  # 1000 CSVs total for testing (~111 per class)
RANDOM_STATE = 42

# Plot generation settings
PLOT_DPI = 100
PLOT_SIZE = (8, 6)
PLOT_STYLE = 'default'

# Training hyperparameters
NUM_EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.001
IMAGE_SIZE = 224

# Model architecture
BACKBONE = 'resnet50'
PRETRAINED = True

# Device
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataloader
NUM_WORKERS = 0  # Set to 0 for Windows compatibility
