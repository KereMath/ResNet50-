"""
Transfer Learning Model
Pre-trained VAE Encoder + New Classifier Head
"""
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add VAE directory to path
vae_dir = Path(__file__).parent.parent / "vae"
sys.path.insert(0, str(vae_dir))

from vae_model import Encoder


class VAETransferClassifier(nn.Module):
    """
    Transfer learning model:
    - Uses pre-trained VAE encoder (frozen or fine-tuned)
    - Adds new classifier head for 9-class prediction
    """

    def __init__(self, latent_dim=128, num_classes=9, channels=3, freeze_encoder=True):
        super(VAETransferClassifier, self).__init__()

        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.freeze_encoder = freeze_encoder

        # Pre-trained encoder (will load weights later)
        self.encoder = Encoder(latent_dim, channels)

        # New classifier head
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def freeze_encoder_weights(self):
        """Freeze encoder weights (only train classifier)"""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("  Encoder weights FROZEN (only classifier will train)")

    def unfreeze_encoder_weights(self):
        """Unfreeze encoder weights (fine-tune entire model)"""
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("  Encoder weights UNFROZEN (fine-tuning entire model)")

    def forward(self, x):
        """
        Forward pass

        Returns:
            class_logits: Classification logits (9 classes)
        """
        # Encode (use mu, ignore logvar for classification)
        mu, _ = self.encoder(x)

        # Classify
        class_logits = self.classifier(mu)

        return class_logits


def create_transfer_model(pretrained_vae_path, latent_dim=128, num_classes=9,
                         channels=3, freeze_encoder=True, device='cpu'):
    """
    Create transfer learning model with pre-trained VAE encoder

    Args:
        pretrained_vae_path: Path to pre-trained VAE model
        latent_dim: Latent dimension (must match pre-trained VAE)
        num_classes: Number of output classes
        channels: Number of image channels
        freeze_encoder: If True, freeze encoder weights
        device: Device to use

    Returns:
        model: Transfer learning model on device
    """
    # Create model
    model = VAETransferClassifier(latent_dim, num_classes, channels, freeze_encoder)

    # Load pre-trained VAE encoder weights
    print("\n" + "=" * 70)
    print("  LOADING PRE-TRAINED VAE ENCODER")
    print("=" * 70)

    checkpoint = torch.load(pretrained_vae_path, map_location=device)

    # Extract encoder weights from checkpoint
    encoder_state_dict = {}
    for key, value in checkpoint['model_state_dict'].items():
        if key.startswith('encoder.'):
            # Remove 'encoder.' prefix
            new_key = key[8:]
            encoder_state_dict[new_key] = value

    # Load encoder weights
    model.encoder.load_state_dict(encoder_state_dict)
    print(f"  Loaded encoder weights from: {pretrained_vae_path}")
    print(f"  VAE trained for {checkpoint['epoch']+1} epochs")
    print(f"  VAE test loss: {checkpoint['test_loss']:.4f}")

    # Freeze or unfreeze encoder
    if freeze_encoder:
        model.freeze_encoder_weights()
    else:
        model.unfreeze_encoder_weights()

    # Move to device
    model = model.to(device)

    print("\n" + "=" * 70)
    print("  TRANSFER LEARNING MODEL")
    print("=" * 70)
    print(f"  Latent dimension: {latent_dim}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Encoder frozen: {freeze_encoder}")
    print(f"  Device: {device}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    classifier_params = sum(p.numel() for p in model.classifier.parameters())

    print(f"\n  Parameter Count:")
    print(f"    Total parameters: {total_params:,}")
    print(f"    Trainable parameters: {trainable_params:,}")
    print(f"    Encoder parameters: {encoder_params:,}")
    print(f"    Classifier parameters: {classifier_params:,}")

    if freeze_encoder:
        print(f"\n  Training Strategy: FAST")
        print(f"    Only {classifier_params:,} classifier params will train")
        print(f"    Encoder ({encoder_params:,} params) stays frozen")
    else:
        print(f"\n  Training Strategy: FINE-TUNING")
        print(f"    All {trainable_params:,} params will train")

    print("=" * 70)

    return model


if __name__ == "__main__":
    # Test model
    import config

    if not config.PRETRAINED_VAE_PATH.exists():
        print(f"ERROR: Pre-trained VAE not found at {config.PRETRAINED_VAE_PATH}")
        print("Please train the VAE first using vae/train_vae.py")
    else:
        model = create_transfer_model(
            pretrained_vae_path=config.PRETRAINED_VAE_PATH,
            latent_dim=config.LATENT_DIM,
            num_classes=len(config.CLASSES),
            channels=config.CHANNELS,
            freeze_encoder=config.FREEZE_ENCODER,
            device=config.DEVICE
        )

        # Test forward pass
        dummy_input = torch.randn(4, 3, 224, 224).to(config.DEVICE)
        class_logits = model(dummy_input)

        print(f"\nTest forward pass:")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Class logits shape: {class_logits.shape}")

        # Test prediction
        predicted_classes = torch.argmax(class_logits, dim=1)
        print(f"  Predicted classes: {predicted_classes.cpu().numpy()}")
