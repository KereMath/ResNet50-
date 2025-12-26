"""
Data Sampler - Sample CSVs from Generated Data
"""
import numpy as np
from pathlib import Path
from collections import Counter
import config


def sample_csv_files():
    """
    Sample CSV files from Generated Data folder

    Returns:
        train_files: Dictionary {class_name: [csv_paths]}
        test_files: Dictionary {class_name: [csv_paths]}
    """
    print("\n" + "=" * 70)
    print("  SAMPLING CSV FILES FROM GENERATED DATA")
    print("=" * 70)

    np.random.seed(config.RANDOM_STATE)

    train_files = {cls: [] for cls in config.CLASSES}
    test_files = {cls: [] for cls in config.CLASSES}

    all_class_files = {}

    # First pass: collect all CSV files for each class
    print("\n  Scanning for CSV files...")
    for class_name in config.CLASSES:
        folder_name = config.CLASS_FOLDERS[class_name]
        class_dir = config.GENERATED_DATA_DIR / folder_name

        if not class_dir.exists():
            print(f"    WARNING: {folder_name} folder not found!")
            continue

        # Recursively find all CSV files
        csv_files = list(class_dir.glob("**/*.csv"))

        if len(csv_files) == 0:
            print(f"    WARNING: No CSV files in {folder_name}!")
            continue

        all_class_files[class_name] = csv_files
        print(f"    {class_name:<25}: {len(csv_files):5d} CSV files found")

    # Sample training data (200 per class)
    print(f"\n  Sampling training data ({config.TRAIN_SAMPLES_PER_CLASS} per class)...")
    for class_name, csv_files in all_class_files.items():
        n_available = len(csv_files)
        n_train = min(config.TRAIN_SAMPLES_PER_CLASS, n_available)

        # Sample for training
        sampled_train = np.random.choice(csv_files, size=n_train, replace=False).tolist()
        train_files[class_name] = sampled_train

        print(f"    {class_name:<25}: {len(sampled_train):3d} samples")

    # Sample test data (1000 total, balanced across classes)
    print(f"\n  Sampling test data ({config.TEST_SAMPLES_TOTAL} total)...")
    test_per_class = config.TEST_SAMPLES_TOTAL // len(all_class_files)

    for class_name, csv_files in all_class_files.items():
        # Get files not used in training
        remaining_files = [f for f in csv_files if f not in train_files[class_name]]

        if len(remaining_files) == 0:
            print(f"    WARNING: {class_name}: No remaining files for test set!")
            continue

        n_test = min(test_per_class, len(remaining_files))
        sampled_test = np.random.choice(remaining_files, size=n_test, replace=False).tolist()
        test_files[class_name] = sampled_test

        print(f"    {class_name:<25}: {len(sampled_test):3d} samples")

    # Summary
    total_train = sum(len(files) for files in train_files.values())
    total_test = sum(len(files) for files in test_files.values())

    print("\n" + "=" * 70)
    print("  SAMPLING SUMMARY")
    print("=" * 70)
    print(f"  Training samples: {total_train} ({total_train // len(config.CLASSES)} per class)")
    print(f"  Test samples:     {total_test} ({total_test // len(all_class_files)} per class)")
    print("=" * 70)

    return train_files, test_files


if __name__ == "__main__":
    train_files, test_files = sample_csv_files()

    print("\n  Sample files:")
    for cls in config.CLASSES[:3]:
        if train_files[cls]:
            print(f"    {cls}: {train_files[cls][0].name}")
