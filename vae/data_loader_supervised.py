"""
Supervised Data Loader for VAE Classifier
Uses labels (unlike unsupervised VAE)
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import config


class SupervisedPlotDataset(Dataset):
    """
    Dataset for supervised VAE classifier training
    Returns (image, label) pairs
    """

    def __init__(self, image_paths_dict, labels_dict, transform=None):
        """
        Args:
            image_paths_dict: Dictionary {class_name: [image_paths]}
            labels_dict: Dictionary {class_name: [label_indices]}
            transform: Image transformations
        """
        # Flatten to lists
        self.image_paths = []
        self.labels = []

        for class_name in config.CLASSES:
            if class_name in image_paths_dict:
                self.image_paths.extend(image_paths_dict[class_name])
                self.labels.extend(labels_dict[class_name])

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(augment=False):
    """
    Get image transforms

    Note: For VAE, we don't use ImageNet normalization
    We keep images in [0, 1] range for reconstruction
    """
    if augment:
        transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(degrees=5),
            transforms.ToTensor(),  # [0, 1]
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),  # [0, 1]
        ])

    return transform


def load_data_from_plots():
    """
    Load image paths and labels from generated_plots directory

    Returns:
        train_paths_dict, train_labels_dict, test_paths_dict, test_labels_dict
    """
    train_paths_dict = {cls: [] for cls in config.CLASSES}
    train_labels_dict = {cls: [] for cls in config.CLASSES}
    test_paths_dict = {cls: [] for cls in config.CLASSES}
    test_labels_dict = {cls: [] for cls in config.CLASSES}

    # Load training data
    train_dir = config.TRAIN_PLOTS_DIR
    if train_dir.exists():
        for class_idx, class_name in enumerate(config.CLASSES):
            class_dir = train_dir / class_name
            if class_dir.exists():
                for img_path in sorted(class_dir.glob('*.png')):
                    train_paths_dict[class_name].append(str(img_path))
                    train_labels_dict[class_name].append(class_idx)

    # Load test data
    test_dir = config.TEST_PLOTS_DIR
    if test_dir.exists():
        for class_idx, class_name in enumerate(config.CLASSES):
            class_dir = test_dir / class_name
            if class_dir.exists():
                for img_path in sorted(class_dir.glob('*.png')):
                    test_paths_dict[class_name].append(str(img_path))
                    test_labels_dict[class_name].append(class_idx)

    return train_paths_dict, train_labels_dict, test_paths_dict, test_labels_dict


def create_supervised_dataloaders(batch_size=32, num_workers=0):
    """
    Create dataloaders for supervised VAE classifier training

    Returns:
        train_loader, test_loader
    """
    # Load data
    train_paths, train_labels, test_paths, test_labels = load_data_from_plots()

    # Create datasets
    train_dataset = SupervisedPlotDataset(
        train_paths, train_labels,
        transform=get_transforms(augment=True)
    )

    test_dataset = SupervisedPlotDataset(
        test_paths, test_labels,
        transform=get_transforms(augment=False)
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    print("\n" + "=" * 70)
    print("  SUPERVISED DATALOADERS CREATED")
    print("=" * 70)
    print(f"  Train: {len(train_dataset)} images, {len(train_loader)} batches")
    print(f"  Test:  {len(test_dataset)} images, {len(test_loader)} batches")
    print("=" * 70)

    return train_loader, test_loader


if __name__ == "__main__":
    # Test dataloader
    train_loader, test_loader = create_supervised_dataloaders(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS
    )

    # Get a batch
    images, labels = next(iter(train_loader))
    print(f"\nTest batch:")
    print(f"  Images shape: {images.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Images min/max: {images.min():.4f} / {images.max():.4f}")
    print(f"  Labels: {labels[:10].numpy()}")
