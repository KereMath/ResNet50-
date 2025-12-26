"""
Data Loader - Load generated plot images for training
"""
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import warnings
warnings.filterwarnings('ignore')

import config


class PlotDataset(Dataset):
    """Dataset for loading generated plot images"""

    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths: List of paths to plot images
            labels: List of corresponding label indices
            transform: Optional transform to apply to images
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Get label
        label = self.labels[idx]

        return image, label


def get_image_transforms(augment=True):
    """Get image transformation pipeline"""
    if augment:
        # Training transforms with augmentation
        transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        # Test transforms (no augmentation)
        transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    return transform


def create_dataloaders(train_plot_paths, train_labels, test_plot_paths, test_labels):
    """
    Create train and test dataloaders

    Args:
        train_plot_paths: Dictionary {class_name: [plot_paths]}
        train_labels: Dictionary {class_name: [label_indices]}
        test_plot_paths: Dictionary {class_name: [plot_paths]}
        test_labels: Dictionary {class_name: [label_indices]}

    Returns:
        train_loader, test_loader
    """
    # Flatten dictionaries to lists
    train_paths_list = []
    train_labels_list = []
    for class_name in config.CLASSES:
        train_paths_list.extend(train_plot_paths[class_name])
        train_labels_list.extend(train_labels[class_name])

    test_paths_list = []
    test_labels_list = []
    for class_name in config.CLASSES:
        test_paths_list.extend(test_plot_paths[class_name])
        test_labels_list.extend(test_labels[class_name])

    # Create datasets
    train_dataset = PlotDataset(
        train_paths_list,
        train_labels_list,
        transform=get_image_transforms(augment=True)
    )

    test_dataset = PlotDataset(
        test_paths_list,
        test_labels_list,
        transform=get_image_transforms(augment=False)
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    print("\n" + "=" * 70)
    print("  DATALOADERS CREATED")
    print("=" * 70)
    print(f"  Train: {len(train_dataset)} images, {len(train_loader)} batches")
    print(f"  Test:  {len(test_dataset)} images, {len(test_loader)} batches")
    print("=" * 70)

    return train_loader, test_loader
