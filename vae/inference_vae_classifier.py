"""
VAE Classifier Inference - Predict anomaly class from image
Uses trained VAE + Classifier model for 9-class prediction
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from pathlib import Path
import json

import config
from vae_classifier import VAEClassifier


def load_vae_classifier(model_path, device):
    """Load trained VAE Classifier model"""
    checkpoint = torch.load(model_path, map_location=device)

    model = VAEClassifier(
        latent_dim=checkpoint['latent_dim'],
        num_classes=checkpoint['num_classes'],
        channels=config.CHANNELS
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print("\n" + "=" * 70)
    print("  VAE CLASSIFIER MODEL LOADED")
    print("=" * 70)
    print(f"  Model path: {model_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Test Accuracy: {checkpoint['test_acc']*100:.2f}%")
    print(f"  Test F1-Score: {checkpoint['test_f1']:.4f}")
    print(f"  Number of classes: {checkpoint['num_classes']}")
    print("=" * 70)

    return model, checkpoint['classes']


def predict_single_image(image_path, model, class_names, device, save_visualization=None):
    """
    Predict anomaly class for a single image

    Args:
        image_path: Path to image
        model: VAE Classifier model
        class_names: List of class names
        device: Device
        save_visualization: Path to save visualization (optional)

    Returns:
        predicted_class: Class name
        class_probabilities: Dictionary {class_name: probability}
        reconstruction: Reconstructed image
    """
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        # Forward pass
        x_recon, mu, logvar, class_logits = model(image_tensor)

        # Get predictions
        probabilities = torch.softmax(class_logits, dim=1)[0]
        predicted_idx = torch.argmax(probabilities).item()
        predicted_class = class_names[predicted_idx]

        # Get class probabilities
        class_probs = {
            class_names[i]: float(probabilities[i])
            for i in range(len(class_names))
        }

    # Sort probabilities
    sorted_probs = dict(sorted(class_probs.items(), key=lambda x: x[1], reverse=True))

    # Visualization
    if save_visualization:
        fig = plt.figure(figsize=(16, 5))
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.5])

        # Original image
        ax1 = fig.add_subplot(gs[0])
        ax1.imshow(np.transpose(image_tensor[0].cpu().numpy(), (1, 2, 0)))
        ax1.set_title('Original Image', fontsize=14, fontweight='bold')
        ax1.axis('off')

        # Reconstructed image
        ax2 = fig.add_subplot(gs[1])
        ax2.imshow(np.transpose(x_recon[0].cpu().numpy(), (1, 2, 0)))
        ax2.set_title('Reconstructed', fontsize=14, fontweight='bold')
        ax2.axis('off')

        # Class probabilities (bar chart)
        ax3 = fig.add_subplot(gs[2])
        classes = list(sorted_probs.keys())
        probs = list(sorted_probs.values())

        # Color bars: green for highest, red for others
        colors = ['green' if i == 0 else 'steelblue' for i in range(len(classes))]

        bars = ax3.barh(classes, probs, color=colors)
        ax3.set_xlabel('Probability', fontsize=12)
        ax3.set_title(f'Predicted Class: {predicted_class}', fontsize=14, fontweight='bold')
        ax3.set_xlim(0, 1)

        # Add probability values
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            ax3.text(prob + 0.02, i, f'{prob:.3f}', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(save_visualization, dpi=150, bbox_inches='tight')
        plt.close()

    return predicted_class, sorted_probs, x_recon


def batch_predict(image_dir, model, class_names, device, output_dir=None):
    """
    Predict anomaly classes for all images in a directory

    Args:
        image_dir: Directory containing images
        model: VAE Classifier model
        class_names: List of class names
        device: Device
        output_dir: Directory to save results (optional)

    Returns:
        results: List of (image_name, predicted_class, top_probability)
    """
    image_dir = Path(image_dir)
    results = []

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("  BATCH CLASSIFICATION")
    print("=" * 70)

    # Get all images
    image_paths = list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg'))

    for img_path in image_paths:
        save_path = None
        if output_dir:
            save_path = output_dir / f"{img_path.stem}_prediction.png"

        predicted_class, class_probs, _ = predict_single_image(
            img_path, model, class_names, device, save_visualization=save_path
        )

        top_prob = class_probs[predicted_class]
        results.append((img_path.name, predicted_class, top_prob))

        print(f"  {img_path.name:<40} → {predicted_class:<25} ({top_prob*100:.1f}%)")

    # Summary by class
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Total Images: {len(results)}")

    class_counts = {}
    for _, pred_class, _ in results:
        class_counts[pred_class] = class_counts.get(pred_class, 0) + 1

    print(f"\n  Predictions by Class:")
    for cls in sorted(class_counts.keys(), key=lambda x: class_counts[x], reverse=True):
        count = class_counts[cls]
        percentage = (count / len(results)) * 100
        print(f"    {cls:<25} : {count:3d} ({percentage:5.1f}%)")

    print("=" * 70)

    return results


def predict_from_csv(csv_path, model, class_names, device, save_plot=True):
    """
    Predict anomaly class from a CSV file (time series data)

    Steps:
    1. Load CSV
    2. Generate plot image
    3. Predict class from image
    4. Return result

    Args:
        csv_path: Path to CSV file
        model: VAE Classifier model
        class_names: List of class names
        device: Device
        save_plot: Whether to save generated plot

    Returns:
        predicted_class, class_probabilities
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    from io import BytesIO

    # Load CSV
    df = pd.read_csv(csv_path)

    # Generate plot (in memory)
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot time series
    if 'timestamp' in df.columns and 'value' in df.columns:
        ax.plot(df['timestamp'], df['value'], linewidth=1.5)
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Value')
    elif len(df.columns) >= 2:
        ax.plot(df.iloc[:, 0], df.iloc[:, 1], linewidth=1.5)
        ax.set_xlabel(df.columns[0])
        ax.set_ylabel(df.columns[1])
    else:
        ax.plot(df.iloc[:, 0], linewidth=1.5)
        ax.set_xlabel('Index')
        ax.set_ylabel('Value')

    ax.grid(True, alpha=0.3)
    ax.set_title('Time Series Data')

    # Save to memory
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)

    # Optionally save to file
    if save_plot:
        plot_path = csv_path.replace('.csv', '_plot.png')
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')

    plt.close()

    # Load image from memory and predict
    image = Image.open(buf).convert('RGB')

    # Preprocess
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    model.eval()
    with torch.no_grad():
        _, _, _, class_logits = model(image_tensor)
        probabilities = torch.softmax(class_logits, dim=1)[0]
        predicted_idx = torch.argmax(probabilities).item()
        predicted_class = class_names[predicted_idx]

        class_probs = {
            class_names[i]: float(probabilities[i])
            for i in range(len(class_names))
        }

    sorted_probs = dict(sorted(class_probs.items(), key=lambda x: x[1], reverse=True))

    return predicted_class, sorted_probs


def main():
    """Example usage"""
    print("\n" + "=" * 70)
    print("  VAE CLASSIFIER - INFERENCE")
    print("  9-Class Time Series Anomaly Classification")
    print("=" * 70)

    device = config.DEVICE

    # Load model
    model_path = config.MODELS_DIR / 'best_vae_classifier.pth'

    if not model_path.exists():
        print(f"\n  ERROR: Model not found at {model_path}")
        print("  Please train the VAE Classifier first using train_vae_classifier.py")
        return

    model, class_names = load_vae_classifier(model_path, device)

    # Example: Test on test set images
    print("\n  Testing on sample images...")

    output_dir = config.RESULTS_DIR / 'vae_classifier_predictions'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test on first class
    test_class_dir = config.TEST_PLOTS_DIR / config.CLASSES[0]
    if test_class_dir.exists():
        # Get first few images
        test_images = list(test_class_dir.glob('*.png'))[:5]

        for img_path in test_images:
            save_path = output_dir / f"{img_path.stem}_result.png"
            predicted_class, class_probs, _ = predict_single_image(
                img_path, model, class_names, device, save_visualization=save_path
            )

            print(f"\n  Image: {img_path.name}")
            print(f"  Predicted: {predicted_class}")
            print(f"  Top 3 probabilities:")
            for i, (cls, prob) in enumerate(list(class_probs.items())[:3]):
                print(f"    {i+1}. {cls:<25} : {prob*100:5.1f}%")

        print(f"\n  Predictions saved to: {output_dir}")


if __name__ == "__main__":
    main()
