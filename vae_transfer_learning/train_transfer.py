"""
Train Transfer Learning Model
Fast training: Uses pre-trained VAE encoder + trains only classifier
"""
import torch
import torch.nn as nn
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
from transfer_model import create_transfer_model
from data_loader import create_dataloaders


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })

    return running_loss / total, correct / total


def evaluate(model, test_loader, criterion, device):
    """Evaluate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return epoch_loss, epoch_acc, precision, recall, f1, all_preds, all_labels


def main():
    """Main training function"""
    print("\n" + "=" * 70)
    print("  TRANSFER LEARNING: VAE ENCODER + CLASSIFIER")
    print("  9-Class Time Series Anomaly Classification")
    print("  Fast Training with Pre-trained Features")
    print("=" * 70)

    device = config.DEVICE
    print(f"\n  Using device: {device}")

    # Check if pre-trained VAE exists
    if not config.PRETRAINED_VAE_PATH.exists():
        print(f"\n  ERROR: Pre-trained VAE not found at {config.PRETRAINED_VAE_PATH}")
        print("  Please train the VAE first using vae/train_vae.py")
        return

    # Create dataloaders
    train_loader, test_loader = create_dataloaders(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS
    )

    # Create transfer learning model
    model = create_transfer_model(
        pretrained_vae_path=config.PRETRAINED_VAE_PATH,
        latent_dim=config.LATENT_DIM,
        num_classes=len(config.CLASSES),
        channels=config.CHANNELS,
        freeze_encoder=config.FREEZE_ENCODER,
        device=device
    )

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()

    # Only optimize trainable parameters
    if config.FREEZE_ENCODER:
        # Only classifier parameters
        optimizer = optim.Adam(model.classifier.parameters(), lr=config.LEARNING_RATE)
        print("\n  Optimizer: Training classifier only (fast)")
    else:
        # All parameters
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        print("\n  Optimizer: Fine-tuning entire model (slow)")

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

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, test_prec, test_rec, test_f1, test_preds, test_labels_arr = evaluate(
            model, test_loader, criterion, device
        )

        scheduler.step(test_f1)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['test_precision'].append(test_prec)
        history['test_recall'].append(test_rec)
        history['test_f1'].append(test_f1)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        print(f"Test  Loss: {test_loss:.4f}, Test Acc:  {test_acc*100:.2f}%")
        print(f"Test  F1:   {test_f1:.4f}, Precision: {test_prec:.4f}, Recall: {test_rec:.4f}")

        # Save best model
        if test_f1 > best_f1:
            best_f1 = test_f1
            best_acc = test_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'test_acc': test_acc,
                'test_f1': test_f1,
                'test_precision': test_prec,
                'test_recall': test_rec,
                'num_classes': len(config.CLASSES),
                'classes': config.CLASSES,
                'latent_dim': config.LATENT_DIM,
                'freeze_encoder': config.FREEZE_ENCODER
            }
            torch.save(checkpoint, config.MODELS_DIR / 'best_transfer_model.pth')
            print(f">>> NEW BEST MODEL! F1: {test_f1:.4f}, Acc: {test_acc*100:.2f}%")

    # Final evaluation
    print("\n" + "=" * 70)
    print("  FINAL EVALUATION (BEST MODEL)")
    print("=" * 70)

    checkpoint = torch.load(config.MODELS_DIR / 'best_transfer_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    final_loss, final_acc, final_prec, final_rec, final_f1, final_preds, final_labels = evaluate(
        model, test_loader, criterion, device
    )

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

    # Save results
    results = {
        'best_epoch': checkpoint['epoch'],
        'best_test_acc': float(final_acc),
        'best_test_f1': float(final_f1),
        'best_test_precision': float(final_prec),
        'best_test_recall': float(final_rec),
        'num_classes': len(config.CLASSES),
        'classes': config.CLASSES,
        'freeze_encoder': config.FREEZE_ENCODER,
        'confusion_matrix': cm.tolist()
    }

    with open(config.RESULTS_DIR / 'transfer_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    with open(config.RESULTS_DIR / 'transfer_history.json', 'w') as f:
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

    # Loss
    axes[1, 0].plot(history['train_loss'], label='Train')
    axes[1, 0].plot(history['test_loss'], label='Test')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Loss')
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
    plt.savefig(config.RESULTS_DIR / 'transfer_training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Best F1-Score: {best_f1:.4f}")
    print(f"  Best Accuracy: {best_acc*100:.2f}%")
    print(f"  Model saved to: {config.MODELS_DIR / 'best_transfer_model.pth'}")
    print(f"  Results saved to: {config.RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
