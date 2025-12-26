"""
Train VAE + Classifier for 9-Class Time Series Anomaly Classification
Combines reconstruction (VAE) with supervised classification
"""
import torch
import torch.optim as optim
import numpy as np
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

import config
from vae_classifier import create_vae_classifier, vae_classifier_loss
from data_loader_supervised import create_supervised_dataloaders


def train_epoch(model, train_loader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    total_class_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        x_recon, mu, logvar, class_logits = model(images)

        # Compute loss
        loss, recon_loss, kl_loss, class_loss = vae_classifier_loss(
            images, x_recon, mu, logvar, class_logits, labels,
            recon_weight=config.RECONSTRUCTION_WEIGHT,
            kl_weight=config.KL_WEIGHT,
            class_weight=1.0  # Classification weight
        )

        # Backward pass
        loss.backward()
        optimizer.step()

        # Accumulate losses
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
        total_class_loss += class_loss.item()

        # Accuracy
        _, predicted = torch.max(class_logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix({
            'loss': f'{loss.item():.2f}',
            'acc': f'{100 * correct / total:.1f}%'
        })

    num_batches = len(train_loader)
    return (total_loss / num_batches,
            total_recon_loss / num_batches,
            total_kl_loss / num_batches,
            total_class_loss / num_batches,
            correct / total)


def evaluate(model, test_loader, device):
    """Evaluate model on test set"""
    model.eval()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    total_class_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            x_recon, mu, logvar, class_logits = model(images)

            # Compute loss
            loss, recon_loss, kl_loss, class_loss = vae_classifier_loss(
                images, x_recon, mu, logvar, class_logits, labels,
                recon_weight=config.RECONSTRUCTION_WEIGHT,
                kl_weight=config.KL_WEIGHT,
                class_weight=1.0
            )

            # Accumulate losses
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            total_class_loss += class_loss.item()

            # Accuracy
            _, predicted = torch.max(class_logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    num_batches = len(test_loader)

    # Compute metrics
    accuracy = correct / total
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return (total_loss / num_batches,
            total_recon_loss / num_batches,
            total_kl_loss / num_batches,
            total_class_loss / num_batches,
            accuracy, precision, recall, f1,
            all_preds, all_labels)


def save_reconstructions(model, test_loader, device, epoch, num_samples=8):
    """Save sample reconstructions"""
    model.eval()

    with torch.no_grad():
        images, labels = next(iter(test_loader))
        images = images[:num_samples].to(device)
        labels = labels[:num_samples]

        # Forward pass
        x_recon, _, _, class_logits = model(images)
        _, predicted = torch.max(class_logits, 1)

        # Convert to numpy
        images_np = images.cpu().numpy()
        x_recon_np = x_recon.cpu().numpy()

        # Create visualization
        fig, axes = plt.subplots(2, num_samples, figsize=(20, 5))

        for i in range(num_samples):
            # Original
            axes[0, i].imshow(np.transpose(images_np[i], (1, 2, 0)))
            axes[0, i].axis('off')
            true_class = config.CLASSES[labels[i]]
            axes[0, i].set_title(f'True: {true_class[:10]}', fontsize=8)

            # Reconstructed
            axes[1, i].imshow(np.transpose(x_recon_np[i], (1, 2, 0)))
            axes[1, i].axis('off')
            pred_class = config.CLASSES[predicted[i]]
            color = 'green' if predicted[i] == labels[i] else 'red'
            axes[1, i].set_title(f'Pred: {pred_class[:10]}', fontsize=8, color=color)

        plt.tight_layout()
        save_path = config.RECONSTRUCTIONS_DIR / f'vae_classifier_epoch_{epoch:03d}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def main():
    """Main training function"""
    print("\n" + "=" * 70)
    print("  VAE + CLASSIFIER TRAINING")
    print("  9-Class Time Series Anomaly Classification")
    print("  Reconstruction + Classification")
    print("=" * 70)

    device = config.DEVICE
    print(f"\n  Using device: {device}")

    # Create dataloaders
    train_loader, test_loader = create_supervised_dataloaders(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS
    )

    # Create model
    model = create_vae_classifier(
        latent_dim=config.LATENT_DIM,
        num_classes=len(config.CLASSES),
        channels=config.CHANNELS,
        device=device
    )

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )

    # Training loop
    print("\n" + "=" * 70)
    print("  TRAINING")
    print("=" * 70)

    best_f1 = 0.0
    best_acc = 0.0
    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': [],
        'test_precision': [], 'test_recall': [], 'test_f1': []
    }

    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        print("-" * 70)

        # Train
        train_loss, train_recon, train_kl, train_class, train_acc = train_epoch(
            model, train_loader, optimizer, device
        )

        # Evaluate
        (test_loss, test_recon, test_kl, test_class,
         test_acc, test_prec, test_rec, test_f1,
         test_preds, test_labels_arr) = evaluate(model, test_loader, device)

        # Update scheduler
        scheduler.step(test_f1)

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['test_precision'].append(test_prec)
        history['test_recall'].append(test_rec)
        history['test_f1'].append(test_f1)

        print(f"Train - Loss: {train_loss:.2f} (Recon: {train_recon:.2f}, KL: {train_kl:.2f}, Class: {train_class:.4f}), Acc: {train_acc*100:.2f}%")
        print(f"Test  - Loss: {test_loss:.2f} (Recon: {test_recon:.2f}, KL: {test_kl:.2f}, Class: {test_class:.4f}), Acc: {test_acc*100:.2f}%")
        print(f"Test  - F1: {test_f1:.4f}, Precision: {test_prec:.4f}, Recall: {test_rec:.4f}")

        # Save best model
        if test_f1 > best_f1:
            best_f1 = test_f1
            best_acc = test_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'test_f1': test_f1,
                'test_precision': test_prec,
                'test_recall': test_rec,
                'latent_dim': config.LATENT_DIM,
                'num_classes': len(config.CLASSES),
                'classes': config.CLASSES
            }
            torch.save(checkpoint, config.MODELS_DIR / 'best_vae_classifier.pth')
            print(f">>> NEW BEST MODEL! F1: {test_f1:.4f}, Acc: {test_acc*100:.2f}%")

        # Save reconstructions every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_reconstructions(model, test_loader, device, epoch + 1)

    # Final evaluation
    print("\n" + "=" * 70)
    print("  FINAL EVALUATION (BEST MODEL)")
    print("=" * 70)

    # Load best model
    checkpoint = torch.load(config.MODELS_DIR / 'best_vae_classifier.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate
    (final_loss, final_recon, final_kl, final_class,
     final_acc, final_prec, final_rec, final_f1,
     final_preds, final_labels) = evaluate(model, test_loader, device)

    print(f"\n  Overall Metrics:")
    print(f"    Accuracy:  {final_acc*100:.2f}%")
    print(f"    Precision: {final_prec:.4f}")
    print(f"    Recall:    {final_rec:.4f}")
    print(f"    F1-Score:  {final_f1:.4f}")

    # Per-class metrics
    print(f"\n  Per-Class Metrics:")
    print(f"  {'-'*70}")
    print(f"  {'Class':<25} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print(f"  {'-'*70}")

    final_labels_arr = np.array(final_labels)
    final_preds_arr = np.array(final_preds)

    for idx, cls_name in enumerate(config.CLASSES):
        cls_mask = final_labels_arr == idx
        support = cls_mask.sum()

        if support > 0:
            binary_labels = (final_labels_arr == idx).astype(int)
            binary_preds = (final_preds_arr == idx).astype(int)

            cls_prec = precision_score(binary_labels, binary_preds, average='binary', zero_division=0)
            cls_rec = recall_score(binary_labels, binary_preds, average='binary', zero_division=0)
            cls_f1 = f1_score(binary_labels, binary_preds, average='binary', zero_division=0)

            print(f"  {cls_name:<25} {cls_prec:>10.4f} {cls_rec:>10.4f} {cls_f1:>10.4f} {support:>10}")

    # Confusion Matrix
    cm = confusion_matrix(final_labels, final_preds)
    print(f"\n  Confusion Matrix:")
    print(f"  {'-'*70}")
    header = "True / Pred"
    print(f"  {header:<20}", end='')
    for cls in config.CLASSES:
        print(f"{cls[:8]:>10}", end='')
    print()
    print(f"  {'-'*70}")
    for i, true_cls in enumerate(config.CLASSES):
        print(f"  {true_cls:<20}", end='')
        for j in range(len(config.CLASSES)):
            print(f"{cm[i][j]:>10}", end='')
        print()
    print(f"  {'-'*70}")

    # Save final reconstructions
    save_reconstructions(model, test_loader, device, config.NUM_EPOCHS, num_samples=16)

    # Save results
    results = {
        'best_epoch': checkpoint['epoch'],
        'best_test_acc': float(final_acc),
        'best_test_f1': float(final_f1),
        'best_test_precision': float(final_prec),
        'best_test_recall': float(final_rec),
        'num_classes': len(config.CLASSES),
        'classes': config.CLASSES,
        'confusion_matrix': cm.tolist()
    }

    with open(config.RESULTS_DIR / 'vae_classifier_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    with open(config.RESULTS_DIR / 'vae_classifier_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # Plot training curves
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Accuracy
    axes[0, 0].plot(history['train_acc'], label='Train')
    axes[0, 0].plot(history['test_acc'], label='Test')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # F1-Score
    axes[0, 1].plot(history['test_f1'], label='Test F1', color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1-Score')
    axes[0, 1].set_title('F1-Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Total Loss
    axes[1, 0].plot(history['train_loss'], label='Train')
    axes[1, 0].plot(history['test_loss'], label='Test')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Total Loss')
    axes[1, 0].set_title('Total Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Precision & Recall
    axes[1, 1].plot(history['test_precision'], label='Precision', color='blue')
    axes[1, 1].plot(history['test_recall'], label='Recall', color='orange')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Precision & Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(config.RESULTS_DIR / 'vae_classifier_training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Best F1-Score: {best_f1:.4f}")
    print(f"  Best Accuracy: {best_acc*100:.2f}%")
    print(f"  Model saved to: {config.MODELS_DIR / 'best_vae_classifier.pth'}")
    print(f"  Results saved to: {config.RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
