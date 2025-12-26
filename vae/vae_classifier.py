"""
VAE + Classifier Head
Uses VAE latent space for 9-class classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from vae_model import Encoder, Decoder


class VAEClassifier(nn.Module):
    """
    VAE with classification head

    Architecture:
    Input Image -> Encoder -> Latent (mu, logvar)
                              |
                              +-> Decoder -> Reconstructed Image
                              |
                              +-> Classifier -> 9 Classes
    """

    def __init__(self, latent_dim=128, num_classes=9, channels=3):
        super(VAEClassifier, self).__init__()

        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # VAE components
        self.encoder = Encoder(latent_dim, channels)
        self.decoder = Decoder(latent_dim, channels)

        # Classification head (from latent space)
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

    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        """
        Forward pass

        Returns:
            x_recon: Reconstructed image
            mu: Latent mean
            logvar: Latent log-variance
            class_logits: Classification logits (9 classes)
        """
        # Encode
        mu, logvar = self.encoder(x)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode (reconstruction)
        x_recon = self.decoder(z)

        # Classify (use mu for stable classification)
        class_logits = self.classifier(mu)

        return x_recon, mu, logvar, class_logits

    def predict_class(self, x):
        """Predict class only (for inference)"""
        mu, _ = self.encoder(x)
        class_logits = self.classifier(mu)
        return class_logits


def vae_classifier_loss(x, x_recon, mu, logvar, class_logits, labels,
                        recon_weight=1.0, kl_weight=1.0, class_weight=1.0):
    """
    Combined loss: Reconstruction + KL + Classification

    Args:
        x: Original images
        x_recon: Reconstructed images
        mu: Latent mean
        logvar: Latent log-variance
        class_logits: Classification logits
        labels: Ground truth labels (0-8)
        recon_weight: Weight for reconstruction loss
        kl_weight: Weight for KL divergence
        class_weight: Weight for classification loss

    Returns:
        total_loss, recon_loss, kl_loss, class_loss
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(x_recon, x, reduction='sum') / x.size(0)

    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

    # Classification loss (CrossEntropy)
    class_loss = F.cross_entropy(class_logits, labels)

    # Total loss
    total_loss = (recon_weight * recon_loss +
                  kl_weight * kl_loss +
                  class_weight * class_loss)

    return total_loss, recon_loss, kl_loss, class_loss


def create_vae_classifier(latent_dim=128, num_classes=9, channels=3, device='cpu'):
    """
    Create VAE Classifier model

    Args:
        latent_dim: Latent space dimension
        num_classes: Number of output classes
        channels: Number of image channels
        device: Device to use

    Returns:
        model: VAE Classifier on device
    """
    model = VAEClassifier(latent_dim, num_classes, channels)
    model = model.to(device)

    print("\n" + "=" * 70)
    print("  VAE + CLASSIFIER MODEL")
    print("=" * 70)
    print(f"  Latent dimension: {latent_dim}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Image channels: {channels}")
    print(f"  Device: {device}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    classifier_params = sum(p.numel() for p in model.classifier.parameters())

    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Encoder parameters: {encoder_params:,}")
    print(f"  Decoder parameters: {decoder_params:,}")
    print(f"  Classifier parameters: {classifier_params:,}")
    print("=" * 70)

    return model


if __name__ == "__main__":
    # Test model
    import config

    model = create_vae_classifier(
        latent_dim=config.LATENT_DIM,
        num_classes=len(config.CLASSES),
        channels=config.CHANNELS,
        device=config.DEVICE
    )

    # Test forward pass
    dummy_input = torch.randn(4, 3, 224, 224).to(config.DEVICE)
    dummy_labels = torch.randint(0, 9, (4,)).to(config.DEVICE)

    x_recon, mu, logvar, class_logits = model(dummy_input)

    print(f"\nTest forward pass:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Reconstructed shape: {x_recon.shape}")
    print(f"  Latent mu shape: {mu.shape}")
    print(f"  Latent logvar shape: {logvar.shape}")
    print(f"  Class logits shape: {class_logits.shape}")

    # Test loss
    total_loss, recon_loss, kl_loss, class_loss = vae_classifier_loss(
        dummy_input, x_recon, mu, logvar, class_logits, dummy_labels
    )

    print(f"\nTest loss computation:")
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  Reconstruction loss: {recon_loss.item():.4f}")
    print(f"  KL loss: {kl_loss.item():.4f}")
    print(f"  Classification loss: {class_loss.item():.4f}")

    # Test prediction
    predicted_classes = torch.argmax(class_logits, dim=1)
    print(f"\nTest prediction:")
    print(f"  True labels: {dummy_labels.cpu().numpy()}")
    print(f"  Predicted: {predicted_classes.cpu().numpy()}")
