"""
Training Script - Train ResNet50 on Pre-Generated Data Plots
9-class visual anomaly classification (training only, no image generation)
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

import config
from data_loader import create_dataloaders
from model import create_model


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
    """Evaluate model and compute detailed metrics"""
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

    # Compute metrics
    epoch_loss = running_loss / total
    epoch_acc = correct / total

    # Compute F1, Precision, Recall (macro average)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return epoch_loss, epoch_acc, precision, recall, f1, all_preds, all_labels


def load_generated_data():
    """Load pre-generated plot paths and labels from generated_plots directory"""
    from pathlib import Path

    plots_dir = config.PLOTS_DIR

    if not plots_dir.exists():
        raise FileNotFoundError(
            f"Generated plots directory not found: {plots_dir}\n"
            "Please ensure the plots have been generated."
        )

    # Create dictionaries for each class
    train_plot_paths = {class_name: [] for class_name in config.CLASSES}
    train_labels = {class_name: [] for class_name in config.CLASSES}
    test_plot_paths = {class_name: [] for class_name in config.CLASSES}
    test_labels = {class_name: [] for class_name in config.CLASSES}

    # Load training data
    train_dir = plots_dir / 'train'
    if train_dir.exists():
        for class_idx, class_name in enumerate(config.CLASSES):
            class_dir = train_dir / class_name
            if class_dir.exists():
                for img_path in sorted(class_dir.glob('*.png')):
                    train_plot_paths[class_name].append(str(img_path))
                    train_labels[class_name].append(class_idx)

    # Load test data
    test_dir = plots_dir / 'test'
    if test_dir.exists():
        for class_idx, class_name in enumerate(config.CLASSES):
            class_dir = test_dir / class_name
            if class_dir.exists():
                for img_path in sorted(class_dir.glob('*.png')):
                    test_plot_paths[class_name].append(str(img_path))
                    test_labels[class_name].append(class_idx)

    # Count total images
    total_train = sum(len(paths) for paths in train_plot_paths.values())
    total_test = sum(len(paths) for paths in test_plot_paths.values())

    if total_train == 0 or total_test == 0:
        raise ValueError(
            f"No images found in {plots_dir}\n"
            "Please ensure the plots have been generated properly."
        )

    return train_plot_paths, train_labels, test_plot_paths, test_labels


def main():
    """Main training function"""
    print("\n" + "=" * 70)
    print("  TRAINING ON PRE-GENERATED DATA")
    print("  9-Class Visual Anomaly Classification")
    print("=" * 70)

    # Create output directories
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    device = config.DEVICE
    print(f"\n  Using device: {device}")

    # Load pre-generated data
    print("\n" + "=" * 70)
    print("  LOADING PRE-GENERATED DATA")
    print("=" * 70)

    try:
        train_plot_paths, train_labels, test_plot_paths, test_labels = load_generated_data()
        total_train = sum(len(paths) for paths in train_plot_paths.values())
        total_test = sum(len(paths) for paths in test_plot_paths.values())
        print(f"  Loaded {total_train} training images")
        print(f"  Loaded {total_test} test images")
    except (FileNotFoundError, ValueError) as e:
        print(f"\n  ERROR: {e}")
        print("\n  Please ensure you have the generated data from a previous run.")
        return

    # Create dataloaders
    print("\n" + "=" * 70)
    print("  CREATING DATALOADERS")
    print("=" * 70)
    train_loader, test_loader = create_dataloaders(
        train_plot_paths, train_labels,
        test_plot_paths, test_labels
    )

    # Create model
    print("\n" + "=" * 70)
    print("  CREATING MODEL")
    print("=" * 70)
    model = create_model(
        num_classes=len(config.CLASSES),
        pretrained=config.PRETRAINED,
        backbone=config.BACKBONE,
        device=device
    )

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )

    # Training loop
    print("\n" + "=" * 70)
    print("  TRAINING")
    print("=" * 70)

    best_test_f1 = 0.0
    best_test_acc = 0.0
    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': [],
        'test_precision': [], 'test_recall': [], 'test_f1': []
    }

    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        print("-" * 70)

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, test_precision, test_recall, test_f1, test_preds, test_labels_arr = evaluate(
            model, test_loader, criterion, device
        )

        scheduler.step(test_f1)  # Use F1 for scheduling

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['test_precision'].append(test_precision)
        history['test_recall'].append(test_recall)
        history['test_f1'].append(test_f1)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        print(f"Test  Loss: {test_loss:.4f}, Test Acc:  {test_acc*100:.2f}%")
        print(f"Test  F1:   {test_f1:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")

        # Save best model (based on F1 score)
        if test_f1 > best_test_f1:
            best_test_f1 = test_f1
            best_test_acc = test_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'test_acc': test_acc,
                'test_f1': test_f1,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'num_classes': len(config.CLASSES),
                'classes': config.CLASSES
            }
            torch.save(checkpoint, config.MODELS_DIR / 'best_model.pth')
            print(f">>> NEW BEST MODEL! F1: {test_f1:.4f}, Acc: {test_acc*100:.2f}%")

    # Final evaluation on best model
    print("\n" + "=" * 70)
    print("  FINAL EVALUATION (BEST MODEL)")
    print("=" * 70)

    # Load best model
    checkpoint = torch.load(config.MODELS_DIR / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate
    _, final_acc, final_prec, final_rec, final_f1, final_preds, final_labels = evaluate(
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

    # Compute per-class metrics
    final_labels_arr = np.array(final_labels)
    final_preds_arr = np.array(final_preds)

    for idx, cls_name in enumerate(config.CLASSES):
        # Get all samples where true label is this class
        cls_mask = final_labels_arr == idx
        support = cls_mask.sum()

        if support > 0:
            # Create binary labels: 1 for this class, 0 for others
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
    model_info = {
        'num_classes': len(config.CLASSES),
        'classes': config.CLASSES,
        'best_test_acc': best_test_acc,
        'best_test_f1': best_test_f1,
        'train_samples': len(train_loader.dataset),
        'test_samples': len(test_loader.dataset),
        'confusion_matrix': cm.tolist()
    }

    with open(config.MODELS_DIR / 'model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)

    with open(config.MODELS_DIR / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Best F1-Score: {best_test_f1:.4f}")
    print(f"  Best Accuracy: {best_test_acc*100:.2f}%")
    print(f"  Model saved to: {config.MODELS_DIR / 'best_model.pth'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
