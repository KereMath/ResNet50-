"""
Data Loader for VAE Training
Loads pre-generated plot images (unsupervised - ignores labels)
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import config


class VAEImageDataset(Dataset):
    """
    Dataset for VAE training
    Unlike supervised learning, we don't care about labels
    We just load all images for reconstruction training
    """

    def __init__(self, image_paths, transform=None):
        """
        Args:
            image_paths: List of paths to images
            transform: Image transformations
        """
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # For VAE, we return the image twice (input and target are the same)
        return image


def get_all_image_paths(plots_dir):
    """
    Get all image paths from the plots directory
    Ignores class labels - just gets all images

    Args:
        plots_dir: Directory containing class subdirectories with images

    Returns:
        image_paths: List of all image paths
    """
    image_paths = []

    if not plots_dir.exists():
        raise FileNotFoundError(f"Plots directory not found: {plots_dir}")

    # Traverse all class directories
    for class_dir in plots_dir.iterdir():
        if class_dir.is_dir():
            # Get all PNG images from this class
            for img_path in class_dir.glob('*.png'):
                image_paths.append(str(img_path))

    return sorted(image_paths)


def create_vae_dataloaders(batch_size=32, num_workers=0):
    """
    Create dataloaders for VAE training

    Args:
        batch_size: Batch size
        num_workers: Number of dataloader workers

    Returns:
        train_loader, test_loader
    """
    # Image transformations (same as classifier, but for reconstruction)
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),  # Converts to [0, 1]
        # Note: No normalization with ImageNet stats for VAE
        # We want to reconstruct the actual pixel values [0, 1]
    ])

    # Get all image paths
    train_paths = get_all_image_paths(config.TRAIN_PLOTS_DIR)
    test_paths = get_all_image_paths(config.TEST_PLOTS_DIR)

    print("\n" + "=" * 70)
    print("  CREATING VAE DATALOADERS")
    print("=" * 70)

    # Create datasets
    train_dataset = VAEImageDataset(train_paths, transform=transform)
    test_dataset = VAEImageDataset(test_paths, transform=transform)

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
    print("  DATALOADERS CREATED")
    print("=" * 70)
    print(f"  Train: {len(train_dataset)} images, {len(train_loader)} batches")
    print(f"  Test:  {len(test_dataset)} images, {len(test_loader)} batches")
    print("=" * 70)

    return train_loader, test_loader


if __name__ == "__main__":
    # Test dataloader
    train_loader, test_loader = create_vae_dataloaders(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS
    )

    # Get a batch
    images = next(iter(train_loader))
    print(f"\nTest batch:")
    print(f"  Images shape: {images.shape}")
    print(f"  Images min/max: {images.min():.4f} / {images.max():.4f}")
