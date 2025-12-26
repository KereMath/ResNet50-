"""
VAE Inference - Anomaly Detection using Reconstruction Error
Load trained VAE and detect anomalies in new images
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from pathlib import Path
import json

import config
from vae_model import VAE


def load_vae_model(model_path, device):
    """Load trained VAE model"""
    checkpoint = torch.load(model_path, map_location=device)

    model = VAE(
        latent_dim=checkpoint['latent_dim'],
        channels=checkpoint['channels']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print("\n" + "=" * 70)
    print("  VAE MODEL LOADED")
    print("=" * 70)
    print(f"  Model path: {model_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Test Loss: {checkpoint['test_loss']:.4f}")
    print(f"  Latent Dim: {checkpoint['latent_dim']}")
    print("=" * 70)

    return model


def load_anomaly_threshold(results_path):
    """Load anomaly threshold from training results"""
    with open(results_path, 'r') as f:
        results = json.load(f)

    threshold = results['anomaly_threshold']
    print(f"\n  Anomaly Threshold: {threshold:.6f}")

    return threshold


def compute_reconstruction_error(model, image_tensor, device):
    """
    Compute reconstruction error for an image

    Args:
        model: VAE model
        image_tensor: Image tensor (1, C, H, W)
        device: Device

    Returns:
        reconstruction_error: MSE between original and reconstructed
        reconstructed_image: Reconstructed image tensor
    """
    model.eval()

    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        reconstructed = model.reconstruct(image_tensor)

        # Compute MSE
        error = torch.mean((image_tensor - reconstructed) ** 2).item()

    return error, reconstructed


def predict_anomaly(image_path, model, threshold, device, save_comparison=None):
    """
    Predict if an image is anomalous

    Args:
        image_path: Path to image
        model: VAE model
        threshold: Anomaly threshold
        device: Device
        save_comparison: Path to save comparison image (optional)

    Returns:
        is_anomaly: Boolean
        reconstruction_error: Float
    """
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Compute reconstruction error
    error, reconstructed = compute_reconstruction_error(model, image_tensor, device)

    # Determine if anomaly
    is_anomaly = error > threshold

    # Save comparison if requested
    if save_comparison:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original
        axes[0].imshow(np.transpose(image_tensor[0].cpu().numpy(), (1, 2, 0)))
        axes[0].set_title('Original', fontsize=14)
        axes[0].axis('off')

        # Reconstructed
        axes[1].imshow(np.transpose(reconstructed[0].cpu().numpy(), (1, 2, 0)))
        axes[1].set_title('Reconstructed', fontsize=14)
        axes[1].axis('off')

        # Difference
        diff = np.abs(image_tensor[0].cpu().numpy() - reconstructed[0].cpu().numpy())
        diff = np.transpose(diff, (1, 2, 0))
        axes[2].imshow(diff)
        axes[2].set_title(f'Difference (Error: {error:.6f})', fontsize=14)
        axes[2].axis('off')

        result_text = "ANOMALY DETECTED!" if is_anomaly else "Normal"
        fig.suptitle(result_text, fontsize=16, fontweight='bold',
                     color='red' if is_anomaly else 'green')

        plt.tight_layout()
        plt.savefig(save_comparison, dpi=150, bbox_inches='tight')
        plt.close()

    return is_anomaly, error


def batch_predict(image_dir, model, threshold, device, output_dir=None):
    """
    Predict anomalies for all images in a directory

    Args:
        image_dir: Directory containing images
        model: VAE model
        threshold: Anomaly threshold
        device: Device
        output_dir: Directory to save results (optional)

    Returns:
        results: List of (image_name, is_anomaly, error)
    """
    image_dir = Path(image_dir)
    results = []

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("  BATCH ANOMALY DETECTION")
    print("=" * 70)

    # Get all images
    image_paths = list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg'))

    for img_path in image_paths:
        save_path = None
        if output_dir:
            save_path = output_dir / f"{img_path.stem}_result.png"

        is_anomaly, error = predict_anomaly(
            img_path, model, threshold, device, save_comparison=save_path
        )

        results.append((img_path.name, is_anomaly, error))

        status = "ANOMALY" if is_anomaly else "NORMAL"
        print(f"  {img_path.name:<40} {status:<10} Error: {error:.6f}")

    # Summary
    num_anomalies = sum(1 for _, is_anomaly, _ in results if is_anomaly)
    print("\n" + "=" * 70)
    print(f"  Total Images: {len(results)}")
    print(f"  Anomalies Detected: {num_anomalies}")
    print(f"  Normal: {len(results) - num_anomalies}")
    print("=" * 70)

    return results


def main():
    """Example usage"""
    print("\n" + "=" * 70)
    print("  VAE ANOMALY DETECTION - INFERENCE")
    print("=" * 70)

    device = config.DEVICE

    # Load model
    model_path = config.MODELS_DIR / 'best_vae.pth'
    results_path = config.RESULTS_DIR / 'vae_results.json'

    if not model_path.exists():
        print(f"\n  ERROR: Model not found at {model_path}")
        print("  Please train the VAE first using train_vae.py")
        return

    model = load_vae_model(model_path, device)
    threshold = load_anomaly_threshold(results_path)

    # Example: Test on test set
    print("\n  Testing on test set images...")

    test_output_dir = config.RESULTS_DIR / 'test_predictions'
    test_output_dir.mkdir(parents=True, exist_ok=True)

    # Get sample images from test set
    test_images_dir = config.TEST_PLOTS_DIR / config.CLASSES[0]  # Just test on one class
    if test_images_dir.exists():
        results = batch_predict(
            test_images_dir,
            model,
            threshold,
            device,
            output_dir=test_output_dir
        )

        # Save results to JSON
        results_dict = [
            {'image': name, 'is_anomaly': bool(is_anom), 'error': float(err)}
            for name, is_anom, err in results
        ]

        with open(test_output_dir / 'predictions.json', 'w') as f:
            json.dump(results_dict, f, indent=2)

        print(f"\n  Results saved to: {test_output_dir}")
    else:
        print(f"\n  Test directory not found: {test_images_dir}")


if __name__ == "__main__":
    main()
