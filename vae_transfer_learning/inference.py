"""
Inference for Transfer Learning Model
Predict 9-class anomaly type from images
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from pathlib import Path
import json

import config
from transfer_model import VAETransferClassifier


def load_transfer_model(model_path, device):
    """Load trained transfer model"""
    checkpoint = torch.load(model_path, map_location=device)

    model = VAETransferClassifier(
        latent_dim=checkpoint['latent_dim'],
        num_classes=checkpoint['num_classes'],
        channels=config.CHANNELS,
        freeze_encoder=False  # Set to False for inference
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print("\n" + "=" * 70)
    print("  TRANSFER MODEL LOADED")
    print("=" * 70)
    print(f"  Model path: {model_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Test Accuracy: {checkpoint['test_acc']*100:.2f}%")
    print(f"  Test F1-Score: {checkpoint['test_f1']:.4f}")
    print(f"  Encoder was frozen: {checkpoint.get('freeze_encoder', True)}")
    print("=" * 70)

    return model, checkpoint['classes']


def predict_image(image_path, model, class_names, device, save_viz=None):
    """
    Predict anomaly class for single image

    Args:
        image_path: Path to image
        model: Transfer model
        class_names: List of class names
        device: Device
        save_viz: Path to save visualization

    Returns:
        predicted_class: Class name
        class_probs: Dictionary {class: probability}
    """
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        pred_class = class_names[pred_idx]

        class_probs = {
            class_names[i]: float(probs[i])
            for i in range(len(class_names))
        }

    sorted_probs = dict(sorted(class_probs.items(), key=lambda x: x[1], reverse=True))

    # Visualization
    if save_viz:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Image
        ax1.imshow(np.transpose(image_tensor[0].cpu().numpy(), (1, 2, 0)))
        ax1.set_title(f'Input Image', fontsize=14, fontweight='bold')
        ax1.axis('off')

        # Probabilities
        classes = list(sorted_probs.keys())
        probs_list = list(sorted_probs.values())
        colors = ['green' if i == 0 else 'steelblue' for i in range(len(classes))]

        ax2.barh(classes, probs_list, color=colors)
        ax2.set_xlabel('Probability', fontsize=12)
        ax2.set_title(f'Predicted: {pred_class}', fontsize=14, fontweight='bold')
        ax2.set_xlim(0, 1)

        for i, (bar, prob) in enumerate(zip(ax2.patches, probs_list)):
            ax2.text(prob + 0.02, i, f'{prob:.3f}', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(save_viz, dpi=150, bbox_inches='tight')
        plt.close()

    return pred_class, sorted_probs


def batch_predict(image_dir, model, class_names, device, output_dir=None):
    """Predict all images in directory"""
    image_dir = Path(image_dir)
    results = []

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("  BATCH PREDICTION")
    print("=" * 70)

    image_paths = list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg'))

    for img_path in image_paths:
        save_path = None
        if output_dir:
            save_path = output_dir / f"{img_path.stem}_pred.png"

        pred_class, class_probs = predict_image(
            img_path, model, class_names, device, save_viz=save_path
        )

        top_prob = class_probs[pred_class]
        results.append((img_path.name, pred_class, top_prob))

        print(f"  {img_path.name:<40} → {pred_class:<25} ({top_prob*100:.1f}%)")

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Total: {len(results)} images")

    class_counts = {}
    for _, cls, _ in results:
        class_counts[cls] = class_counts.get(cls, 0) + 1

    print(f"\n  Predictions by Class:")
    for cls in sorted(class_counts.keys(), key=lambda x: class_counts[x], reverse=True):
        count = class_counts[cls]
        pct = (count / len(results)) * 100
        print(f"    {cls:<25} : {count:3d} ({pct:5.1f}%)")

    print("=" * 70)

    return results


def main():
    """Example usage"""
    print("\n" + "=" * 70)
    print("  TRANSFER MODEL INFERENCE")
    print("  9-Class Anomaly Classification")
    print("=" * 70)

    device = config.DEVICE

    model_path = config.MODELS_DIR / 'best_transfer_model.pth'

    if not model_path.exists():
        print(f"\n  ERROR: Model not found at {model_path}")
        print("  Please train first: python train_transfer.py")
        return

    model, class_names = load_transfer_model(model_path, device)

    # Test on sample
    test_class_dir = config.TEST_PLOTS_DIR / config.CLASSES[0]
    if test_class_dir.exists():
        output_dir = config.RESULTS_DIR / 'predictions'
        output_dir.mkdir(exist_ok=True)

        test_images = list(test_class_dir.glob('*.png'))[:5]

        for img_path in test_images:
            save_path = output_dir / f"{img_path.stem}_result.png"
            pred_class, class_probs = predict_image(
                img_path, model, class_names, device, save_viz=save_path
            )

            print(f"\n  Image: {img_path.name}")
            print(f"  Predicted: {pred_class}")
            print(f"  Top 3:")
            for i, (cls, prob) in enumerate(list(class_probs.items())[:3]):
                print(f"    {i+1}. {cls:<25} : {prob*100:5.1f}%")

        print(f"\n  Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
