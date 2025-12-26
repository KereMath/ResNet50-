"""
Data Loader for Transfer Learning
Same as VAE supervised loader but in transfer learning folder
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import config


class TransferDataset(Dataset):
    """Dataset for transfer learning"""

    def __init__(self, image_paths_dict, labels_dict, transform=None):
        """
        Args:
            image_paths_dict: Dictionary {class_name: [image_paths]}
            labels_dict: Dictionary {class_name: [label_indices]}
            transform: Image transformations
        """
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

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(augment=False):
    """Get image transforms (no normalization for VAE)"""
    if augment:
        transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(degrees=5),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
        ])

    return transform


def load_data():
    """Load image paths and labels from plots directory"""
    train_paths = {cls: [] for cls in config.CLASSES}
    train_labels = {cls: [] for cls in config.CLASSES}
    test_paths = {cls: [] for cls in config.CLASSES}
    test_labels = {cls: [] for cls in config.CLASSES}

    # Load training data
    train_dir = config.TRAIN_PLOTS_DIR
    if train_dir.exists():
        for class_idx, class_name in enumerate(config.CLASSES):
            class_dir = train_dir / class_name
            if class_dir.exists():
                for img_path in sorted(class_dir.glob('*.png')):
                    train_paths[class_name].append(str(img_path))
                    train_labels[class_name].append(class_idx)

    # Load test data
    test_dir = config.TEST_PLOTS_DIR
    if test_dir.exists():
        for class_idx, class_name in enumerate(config.CLASSES):
            class_dir = test_dir / class_name
            if class_dir.exists():
                for img_path in sorted(class_dir.glob('*.png')):
                    test_paths[class_name].append(str(img_path))
                    test_labels[class_name].append(class_idx)

    return train_paths, train_labels, test_paths, test_labels


def create_dataloaders(batch_size=32, num_workers=0):
    """Create dataloaders"""
    train_paths, train_labels, test_paths, test_labels = load_data()

    train_dataset = TransferDataset(
        train_paths, train_labels,
        transform=get_transforms(augment=True)
    )

    test_dataset = TransferDataset(
        test_paths, test_labels,
        transform=get_transforms(augment=False)
    )

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
    print("  DATALOADERS CREATED")
    print("=" * 70)
    print(f"  Train: {len(train_dataset)} images, {len(train_loader)} batches")
    print(f"  Test:  {len(test_dataset)} images, {len(test_loader)} batches")
    print("=" * 70)

    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = create_dataloaders(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS
    )

    images, labels = next(iter(train_loader))
    print(f"\nTest batch:")
    print(f"  Images shape: {images.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Images min/max: {images.min():.4f} / {images.max():.4f}")
