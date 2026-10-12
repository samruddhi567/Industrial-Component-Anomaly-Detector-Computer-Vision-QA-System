"""
CLI script to generate synthetic test dataset of industrial components.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.utils.dataset_gen import generate_offline_dataset

if __name__ == "__main__":
    out_dir = str(ROOT_DIR / "data" / "sample_dataset")
    print(f"Generating synthetic industrial component dataset into: {out_dir}...")
    summary = generate_offline_dataset(output_dir=out_dir, num_samples_per_class=15)
    print(f"Dataset generated successfully!")
    print(f"Total Images: {summary['total_images']}")
    print(f"Categories: {', '.join(summary['categories'])}")
    print(f"Manifest: {summary['manifest_path']}")
