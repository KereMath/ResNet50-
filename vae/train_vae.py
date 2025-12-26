"""
Train VAE for Anomaly Detection
Unsupervised learning - learns to reconstruct images
Anomalies detected via high reconstruction error
"""
import torch
import torch.optim as optim
import numpy as np
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import config
from vae_model import create_vae, vae_loss
from data_loader import create_vae_dataloaders


def train_epoch(model, train_loader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for images in pbar:
        images = images.to(device)

        # Forward pass
        optimizer.zero_grad()
        x_recon, mu, logvar = model(images)

        # Compute loss
        loss, recon_loss, kl_loss = vae_loss(
            images, x_recon, mu, logvar,
            reconstruction_weight=config.RECONSTRUCTION_WEIGHT,
            kl_weight=config.KL_WEIGHT
        )

        # Backward pass
        loss.backward()
        optimizer.step()

        # Accumulate losses
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
        num_batches += 1

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'recon': f'{recon_loss.item():.4f}',
            'kl': f'{kl_loss.item():.4f}'
        })

    return (total_loss / num_batches,
            total_recon_loss / num_batches,
            total_kl_loss / num_batches)


def evaluate(model, test_loader, device):
    """Evaluate model on test set"""
    model.eval()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    num_batches = 0

    reconstruction_errors = []

    with torch.no_grad():
        for images in tqdm(test_loader, desc="Evaluating", leave=False):
            images = images.to(device)

            # Forward pass
            x_recon, mu, logvar = model(images)

            # Compute loss
            loss, recon_loss, kl_loss = vae_loss(
                images, x_recon, mu, logvar,
                reconstruction_weight=config.RECONSTRUCTION_WEIGHT,
                kl_weight=config.KL_WEIGHT
            )

            # Accumulate losses
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            num_batches += 1

            # Per-sample reconstruction error (for anomaly detection)
            per_sample_error = torch.mean((images - x_recon) ** 2, dim=[1, 2, 3])
            reconstruction_errors.extend(per_sample_error.cpu().numpy())

    avg_loss = total_loss / num_batches
    avg_recon_loss = total_recon_loss / num_batches
    avg_kl_loss = total_kl_loss / num_batches

    return avg_loss, avg_recon_loss, avg_kl_loss, reconstruction_errors


def save_reconstructions(model, test_loader, device, epoch, num_samples=8):
    """Save sample reconstructions for visualization"""
    model.eval()

    with torch.no_grad():
        # Get one batch
        images = next(iter(test_loader))
        images = images[:num_samples].to(device)

        # Reconstruct
        x_recon = model.reconstruct(images)

        # Convert to numpy
        images_np = images.cpu().numpy()
        x_recon_np = x_recon.cpu().numpy()

        # Create visualization
        fig, axes = plt.subplots(2, num_samples, figsize=(20, 5))

        for i in range(num_samples):
            # Original
            axes[0, i].imshow(np.transpose(images_np[i], (1, 2, 0)))
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Original', fontsize=12)

            # Reconstructed
            axes[1, i].imshow(np.transpose(x_recon_np[i], (1, 2, 0)))
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('Reconstructed', fontsize=12)

        plt.tight_layout()
        save_path = config.RECONSTRUCTIONS_DIR / f'reconstruction_epoch_{epoch:03d}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Saved reconstructions to: {save_path}")


def main():
    """Main training function"""
    print("\n" + "=" * 70)
    print("  VAE TRAINING FOR ANOMALY DETECTION")
    print("  Unsupervised Learning - Reconstruction-Based")
    print("=" * 70)

    device = config.DEVICE
    print(f"\n  Using device: {device}")

    # Create dataloaders
    train_loader, test_loader = create_vae_dataloaders(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS
    )

    # Create VAE model
    model = create_vae(
        latent_dim=config.LATENT_DIM,
        channels=config.CHANNELS,
        device=device
    )

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Training loop
    print("\n" + "=" * 70)
    print("  TRAINING")
    print("=" * 70)

    best_test_loss = float('inf')
    history = {
        'train_loss': [], 'train_recon_loss': [], 'train_kl_loss': [],
        'test_loss': [], 'test_recon_loss': [], 'test_kl_loss': []
    }

    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        print("-" * 70)

        # Train
        train_loss, train_recon, train_kl = train_epoch(
            model, train_loader, optimizer, device
        )

        # Evaluate
        test_loss, test_recon, test_kl, recon_errors = evaluate(
            model, test_loader, device
        )

        # Update scheduler
        scheduler.step(test_loss)

        # Save history
        history['train_loss'].append(train_loss)
        history['train_recon_loss'].append(train_recon)
        history['train_kl_loss'].append(train_kl)
        history['test_loss'].append(test_loss)
        history['test_recon_loss'].append(test_recon)
        history['test_kl_loss'].append(test_kl)

        print(f"Train - Loss: {train_loss:.4f}, Recon: {train_recon:.4f}, KL: {train_kl:.4f}")
        print(f"Test  - Loss: {test_loss:.4f}, Recon: {test_recon:.4f}, KL: {test_kl:.4f}")

        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_loss': test_loss,
                'test_recon_loss': test_recon,
                'test_kl_loss': test_kl,
                'latent_dim': config.LATENT_DIM,
                'channels': config.CHANNELS
            }
            torch.save(checkpoint, config.MODELS_DIR / 'best_vae.pth')
            print(f">>> NEW BEST MODEL! Test Loss: {test_loss:.4f}")

        # Save reconstructions every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_reconstructions(model, test_loader, device, epoch + 1)

    # Final evaluation
    print("\n" + "=" * 70)
    print("  FINAL EVALUATION")
    print("=" * 70)

    # Load best model
    checkpoint = torch.load(config.MODELS_DIR / 'best_vae.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate on test set
    test_loss, test_recon, test_kl, recon_errors = evaluate(
        model, test_loader, device
    )

    print(f"\n  Best Model Performance:")
    print(f"    Total Loss: {test_loss:.4f}")
    print(f"    Reconstruction Loss: {test_recon:.4f}")
    print(f"    KL Divergence: {test_kl:.4f}")

    # Compute anomaly detection threshold
    recon_errors = np.array(recon_errors)
    threshold = np.percentile(recon_errors, config.ANOMALY_THRESHOLD_PERCENTILE)

    print(f"\n  Reconstruction Error Statistics:")
    print(f"    Mean: {np.mean(recon_errors):.6f}")
    print(f"    Std:  {np.std(recon_errors):.6f}")
    print(f"    Min:  {np.min(recon_errors):.6f}")
    print(f"    Max:  {np.max(recon_errors):.6f}")
    print(f"    Anomaly Threshold ({config.ANOMALY_THRESHOLD_PERCENTILE}th percentile): {threshold:.6f}")

    # Save final reconstructions
    save_reconstructions(model, test_loader, device, config.NUM_EPOCHS, num_samples=16)

    # Plot training curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Total loss
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['test_loss'], label='Test')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Total Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Reconstruction loss
    axes[1].plot(history['train_recon_loss'], label='Train')
    axes[1].plot(history['test_recon_loss'], label='Test')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Reconstruction Loss')
    axes[1].set_title('Reconstruction Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # KL divergence
    axes[2].plot(history['train_kl_loss'], label='Train')
    axes[2].plot(history['test_kl_loss'], label='Test')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('KL Divergence')
    axes[2].set_title('KL Divergence')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(config.RESULTS_DIR / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  Training curves saved to: {config.RESULTS_DIR / 'training_curves.png'}")

    # Save results
    results = {
        'best_epoch': checkpoint['epoch'],
        'best_test_loss': test_loss,
        'best_test_recon_loss': test_recon,
        'best_test_kl_loss': test_kl,
        'anomaly_threshold': float(threshold),
        'latent_dim': config.LATENT_DIM,
        'num_epochs': config.NUM_EPOCHS,
        'batch_size': config.BATCH_SIZE,
        'learning_rate': config.LEARNING_RATE,
        'reconstruction_error_stats': {
            'mean': float(np.mean(recon_errors)),
            'std': float(np.std(recon_errors)),
            'min': float(np.min(recon_errors)),
            'max': float(np.max(recon_errors))
        }
    }

    with open(config.RESULTS_DIR / 'vae_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    with open(config.RESULTS_DIR / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Best Test Loss: {test_loss:.4f}")
    print(f"  Model saved to: {config.MODELS_DIR / 'best_vae.pth'}")
    print(f"  Results saved to: {config.RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
