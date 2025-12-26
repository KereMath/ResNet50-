"""
Plot Generator - Generate time series plots from CSV files
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import config


def generate_plot_from_csv(csv_path, output_path):
    """
    Generate a time series plot from a CSV file

    Args:
        csv_path: Path to CSV file
        output_path: Path to save plot image

    Returns:
        True if successful, False otherwise
    """
    try:
        # Load CSV
        df = pd.read_csv(csv_path)

        # Get time series data (first numerical column)
        if 'data' in df.columns:
            ts_data = df['data'].values
        elif 'value' in df.columns:
            ts_data = df['value'].values
        else:
            ts_data = df.iloc[:, 0].values

        # Ensure numeric and remove NaN
        ts_data = pd.to_numeric(ts_data, errors='coerce')
        ts_data = ts_data[~np.isnan(ts_data)]

        if len(ts_data) == 0:
            return False

        # Create plot
        fig, ax = plt.subplots(figsize=config.PLOT_SIZE, dpi=config.PLOT_DPI)
        ax.plot(ts_data, linewidth=1.5, color='#1f77b4')
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.grid(True, alpha=0.3)

        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(output_path, dpi=config.PLOT_DPI, bbox_inches='tight')
        plt.close(fig)

        return True

    except Exception as e:
        print(f"    ERROR generating plot for {csv_path.name}: {e}")
        return False


def generate_plots_for_split(files_dict, output_base_dir, split_name):
    """
    Generate plots for a train/test split

    Args:
        files_dict: Dictionary {class_name: [csv_paths]}
        output_base_dir: Base directory for plots
        split_name: "train" or "test"

    Returns:
        plot_paths: Dictionary {class_name: [plot_paths]}
        labels: Dictionary {class_name: [label_indices]}
    """
    print("\n" + "=" * 70)
    print(f"  GENERATING {split_name.upper()} PLOTS")
    print("=" * 70)

    plot_paths = {cls: [] for cls in config.CLASSES}
    labels = {cls: [] for cls in config.CLASSES}

    total_plots = sum(len(files) for files in files_dict.values())
    print(f"  Total plots to generate: {total_plots}")

    pbar = tqdm(total=total_plots, desc=f"  Generating {split_name} plots")

    for class_name, csv_files in files_dict.items():
        if not csv_files:
            continue

        # Create class directory
        class_dir = output_base_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        label_idx = config.CLASS_TO_IDX[class_name]

        for csv_path in csv_files:
            # Generate plot filename
            plot_filename = csv_path.stem + '.png'
            plot_path = class_dir / plot_filename

            # Generate plot
            success = generate_plot_from_csv(csv_path, plot_path)

            if success:
                plot_paths[class_name].append(str(plot_path))
                labels[class_name].append(label_idx)

            pbar.update(1)

    pbar.close()

    # Summary
    total_generated = sum(len(paths) for paths in plot_paths.values())
    print(f"\n  Successfully generated: {total_generated}/{total_plots} plots")

    for class_name in config.CLASSES:
        if plot_paths[class_name]:
            print(f"    {class_name:<25}: {len(plot_paths[class_name]):3d} plots")

    return plot_paths, labels


def generate_all_plots(train_files, test_files):
    """
    Generate all plots for training and testing

    Args:
        train_files: Dictionary {class_name: [csv_paths]}
        test_files: Dictionary {class_name: [csv_paths]}

    Returns:
        train_plot_paths: Dictionary {class_name: [plot_paths]}
        train_labels: Dictionary {class_name: [label_indices]}
        test_plot_paths: Dictionary {class_name: [plot_paths]}
        test_labels: Dictionary {class_name: [label_indices]}
    """
    # Create output directories
    config.TRAIN_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    config.TEST_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Generate training plots
    train_plot_paths, train_labels = generate_plots_for_split(
        train_files, config.TRAIN_PLOTS_DIR, "train"
    )

    # Generate test plots
    test_plot_paths, test_labels = generate_plots_for_split(
        test_files, config.TEST_PLOTS_DIR, "test"
    )

    print("\n" + "=" * 70)
    print("  PLOT GENERATION COMPLETE")
    print("=" * 70)
    print(f"  Training plots: {config.TRAIN_PLOTS_DIR}")
    print(f"  Test plots:     {config.TEST_PLOTS_DIR}")
    print("=" * 70)

    return train_plot_paths, train_labels, test_plot_paths, test_labels


if __name__ == "__main__":
    from data_sampler import sample_csv_files

    print("Sampling CSV files...")
    train_files, test_files = sample_csv_files()

    print("\nGenerating plots...")
    train_plots, train_labels, test_plots, test_labels = generate_all_plots(
        train_files, test_files
    )

    print(f"\nDone! Check {config.PLOTS_DIR}")
