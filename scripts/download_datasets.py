#!/usr/bin/env python3
"""Download all datasets required for Pi0.6 training.

This script downloads:
1. Open X-Embodiment datasets (Bridge, RT-1, Fractal, etc.) via TFDS
2. ALOHA sim datasets via LeRobot
3. LIBERO dataset via LeRobot

DROID must be downloaded separately from: https://droid-dataset.github.io/

Usage:
    python scripts/download_datasets.py --data_dir /data/rlds_datasets
    python scripts/download_datasets.py --data_dir /data/rlds_datasets --skip_oxe  # Skip large OXE datasets
"""

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Open X-Embodiment datasets to download
OXE_DATASETS = [
    "bridge_dataset",           # WidowX manipulation (~60k demos)
    "fractal20220817_data",     # Google robot manipulation
    "rt1_robot_action",         # RT-1 language-conditioned (~130k episodes)
    "kuka",                     # Kuka industrial manipulation
    "taco_play",                # TACO benchmark
    "berkeley_cable_routing",   # Deformable objects
    "berkeley_autolab_ur5",     # UR5 manipulation
]

# LeRobot datasets to download
LEROBOT_DATASETS = [
    "lerobot/aloha_sim_transfer_cube_human",
    "lerobot/aloha_sim_insertion_human",
    "lerobot/aloha_sim_transfer_cube_scripted",
    "lerobot/aloha_sim_insertion_scripted",
    "physical-intelligence/libero",
    "physical-intelligence/aloha_pen_uncap_diverse",
]


def download_oxe_datasets(data_dir: str, datasets: list[str]) -> None:
    """Download Open X-Embodiment datasets using TensorFlow Datasets."""
    try:
        import tensorflow_datasets as tfds
    except ImportError:
        logger.error("tensorflow_datasets not installed. Run: pip install tensorflow_datasets")
        return

    logger.info(f"Downloading {len(datasets)} Open X-Embodiment datasets to {data_dir}")

    for i, dataset_name in enumerate(datasets, 1):
        logger.info(f"[{i}/{len(datasets)}] Downloading {dataset_name}...")
        try:
            # Just load to trigger download
            builder = tfds.builder(dataset_name, data_dir=data_dir)
            builder.download_and_prepare()
            logger.info(f"  ✓ {dataset_name} downloaded successfully")
        except Exception as e:
            logger.error(f"  ✗ Failed to download {dataset_name}: {e}")
            continue


def download_lerobot_datasets(datasets: list[str]) -> None:
    """Download LeRobot datasets."""
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        logger.error("lerobot not installed. Run: pip install lerobot")
        return

    logger.info(f"Downloading {len(datasets)} LeRobot datasets")

    for i, repo_id in enumerate(datasets, 1):
        logger.info(f"[{i}/{len(datasets)}] Downloading {repo_id}...")
        try:
            # Loading triggers download
            ds = LeRobotDataset(repo_id)
            logger.info(f"  ✓ {repo_id}: {ds.num_episodes} episodes, {ds.num_frames} frames")
        except Exception as e:
            logger.error(f"  ✗ Failed to download {repo_id}: {e}")
            continue


def check_droid(data_dir: str) -> bool:
    """Check if DROID dataset is available."""
    droid_path = Path(data_dir) / "droid"
    if droid_path.exists():
        logger.info(f"✓ DROID dataset found at {droid_path}")
        return True
    else:
        logger.warning(f"✗ DROID dataset not found at {droid_path}")
        logger.warning("  Download DROID from: https://droid-dataset.github.io/")
        logger.warning("  DROID is required for pi06_comprehensive training")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download datasets for Pi0.6 training")
    parser.add_argument("--data_dir", type=str, default="/data/rlds_datasets",
                        help="Directory to store RLDS datasets")
    parser.add_argument("--skip_oxe", action="store_true",
                        help="Skip Open X-Embodiment downloads (large)")
    parser.add_argument("--skip_lerobot", action="store_true",
                        help="Skip LeRobot downloads")
    parser.add_argument("--only", type=str, default=None,
                        help="Only download specific dataset (comma-separated)")
    args = parser.parse_args()

    # Create data directory
    os.makedirs(args.data_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Pi0.6 Dataset Downloader")
    logger.info("=" * 60)

    # Check DROID
    check_droid(args.data_dir)

    # Download OXE datasets
    if not args.skip_oxe:
        if args.only:
            datasets = [d.strip() for d in args.only.split(",") if d.strip() in OXE_DATASETS]
        else:
            datasets = OXE_DATASETS

        if datasets:
            download_oxe_datasets(args.data_dir, datasets)
    else:
        logger.info("Skipping Open X-Embodiment downloads (--skip_oxe)")

    # Download LeRobot datasets
    if not args.skip_lerobot:
        if args.only:
            datasets = [d.strip() for d in args.only.split(",") if d.strip() in LEROBOT_DATASETS]
        else:
            datasets = LEROBOT_DATASETS

        if datasets:
            download_lerobot_datasets(datasets)
    else:
        logger.info("Skipping LeRobot downloads (--skip_lerobot)")

    logger.info("=" * 60)
    logger.info("Download complete!")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Download DROID manually if not present: https://droid-dataset.github.io/")
    logger.info("2. Update data paths in training configs")
    logger.info("3. Run training: python scripts/train.py pi06_comprehensive --exp_name pi06_v1")


if __name__ == "__main__":
    main()
