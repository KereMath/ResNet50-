"""
Variational Autoencoder (VAE) for Anomaly Detection
Architecture: Convolutional VAE for image reconstruction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Encoder network: Image -> Latent distribution parameters"""

    def __init__(self, latent_dim=128, channels=3):
        super(Encoder, self).__init__()

        # Convolutional layers
        # 224x224x3 -> 112x112x32
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # 112x112x32 -> 56x56x64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # 56x56x64 -> 28x28x128
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # 28x28x128 -> 14x14x256
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # 14x14x256 -> 7x7x512
        self.conv5 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(512)

        # Flatten: 7x7x512 = 25088
        self.flatten_size = 7 * 7 * 512

        # Latent parameters (mean and log-variance)
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)

    def forward(self, x):
        # Encoder forward pass
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))
        h = F.relu(self.bn4(self.conv4(h)))
        h = F.relu(self.bn5(self.conv5(h)))

        # Flatten
        h = h.view(h.size(0), -1)

        # Get latent distribution parameters
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar


class Decoder(nn.Module):
    """Decoder network: Latent -> Reconstructed image"""

    def __init__(self, latent_dim=128, channels=3):
        super(Decoder, self).__init__()

        self.latent_dim = latent_dim
        self.initial_size = 7
        self.initial_channels = 512

        # Linear layer to expand latent vector
        self.fc = nn.Linear(latent_dim, self.initial_size * self.initial_size * self.initial_channels)

        # Transposed convolutional layers (upsampling)
        # 7x7x512 -> 14x14x256
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(256)

        # 14x14x256 -> 28x28x128
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        # 28x28x128 -> 56x56x64
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # 56x56x64 -> 112x112x32
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(32)

        # 112x112x32 -> 224x224x3
        self.deconv5 = nn.ConvTranspose2d(32, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, z):
        # Expand latent vector
        h = self.fc(z)
        h = h.view(h.size(0), self.initial_channels, self.initial_size, self.initial_size)

        # Decoder forward pass
        h = F.relu(self.bn1(self.deconv1(h)))
        h = F.relu(self.bn2(self.deconv2(h)))
        h = F.relu(self.bn3(self.deconv3(h)))
        h = F.relu(self.bn4(self.deconv4(h)))

        # Final layer with sigmoid activation (output in [0, 1])
        x_recon = torch.sigmoid(self.deconv5(h))

        return x_recon


class VAE(nn.Module):
    """Complete VAE model"""

    def __init__(self, latent_dim=128, channels=3):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim, channels)
        self.decoder = Decoder(latent_dim, channels)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = mu + std * epsilon"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        # Encode
        mu, logvar = self.encoder(x)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode
        x_recon = self.decoder(z)

        return x_recon, mu, logvar

    def reconstruct(self, x):
        """Reconstruct input without reparameterization (for inference)"""
        mu, _ = self.encoder(x)
        x_recon = self.decoder(mu)
        return x_recon

    def generate(self, num_samples, device):
        """Generate new samples from random latent vectors"""
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decoder(z)
        return samples


def vae_loss(x, x_recon, mu, logvar, reconstruction_weight=1.0, kl_weight=1.0):
    """
    VAE loss = Reconstruction loss + KL divergence

    Args:
        x: Original images
        x_recon: Reconstructed images
        mu: Latent mean
        logvar: Latent log-variance
        reconstruction_weight: Weight for reconstruction loss
        kl_weight: Weight for KL divergence (beta in beta-VAE)

    Returns:
        total_loss, reconstruction_loss, kl_loss
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(x_recon, x, reduction='sum') / x.size(0)

    # KL divergence loss
    # KL(N(mu, sigma) || N(0, 1))
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

    # Total loss
    total_loss = reconstruction_weight * recon_loss + kl_weight * kl_loss

    return total_loss, recon_loss, kl_loss


def create_vae(latent_dim=128, channels=3, device='cpu'):
    """
    Create and initialize VAE model

    Args:
        latent_dim: Dimension of latent space
        channels: Number of image channels
        device: Device to use

    Returns:
        model: VAE model on device
    """
    model = VAE(latent_dim=latent_dim, channels=channels)
    model = model.to(device)

    print("\n" + "=" * 70)
    print("  VAE MODEL ARCHITECTURE")
    print("=" * 70)
    print(f"  Latent dimension: {latent_dim}")
    print(f"  Image channels: {channels}")
    print(f"  Image size: 224x224")
    print(f"  Device: {device}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())

    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Encoder parameters: {encoder_params:,}")
    print(f"  Decoder parameters: {decoder_params:,}")
    print("=" * 70)

    return model


if __name__ == "__main__":
    # Test VAE
    import config

    model = create_vae(
        latent_dim=config.LATENT_DIM,
        channels=config.CHANNELS,
        device=config.DEVICE
    )

    # Test forward pass
    dummy_input = torch.randn(4, 3, 224, 224).to(config.DEVICE)
    x_recon, mu, logvar = model(dummy_input)

    print(f"\nTest forward pass:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Reconstructed shape: {x_recon.shape}")
    print(f"  Latent mu shape: {mu.shape}")
    print(f"  Latent logvar shape: {logvar.shape}")

    # Test loss
    total_loss, recon_loss, kl_loss = vae_loss(dummy_input, x_recon, mu, logvar)
    print(f"\nTest loss computation:")
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  Reconstruction loss: {recon_loss.item():.4f}")
    print(f"  KL loss: {kl_loss.item():.4f}")

    # Test generation
    samples = model.generate(4, config.DEVICE)
    print(f"\nTest generation:")
    print(f"  Generated samples shape: {samples.shape}")
