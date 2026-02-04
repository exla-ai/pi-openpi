#!/usr/bin/env python3
"""Convert Isaac Lab numpy episodes to LeRobot format.

This script expects the numpy output produced by `scripts/isaaclab_data_collection.py` when
run with `--format numpy` or `--format both`.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def _discover_episode_dirs(input_dir: Path) -> list[Path]:
    episode_dirs = sorted([p for p in input_dir.glob("episode_*") if p.is_dir()])
    if not episode_dirs:
        raise FileNotFoundError(f"No episode_* directories found in {input_dir}")
    return episode_dirs


def _infer_keys(episode_dir: Path) -> tuple[str, list[str]]:
    files = [p for p in episode_dir.glob("*.npy") if p.name not in {"actions.npy", "rewards.npy", "metadata.npy"}]
    if not files:
        raise FileNotFoundError(f"No observation files found in {episode_dir}")

    keys = [p.stem for p in files]
    if "state" not in keys:
        raise ValueError(f"Expected a state.npy file in {episode_dir}, found {keys}")

    image_keys = [k for k in keys if k != "state"]
    if not image_keys:
        raise ValueError(f"No image observation files found in {episode_dir}")

    # Default to the first image key as the top camera.
    return image_keys[0], image_keys


def _build_features(image_shape: tuple[int, ...], state_dim: int, action_dim: int) -> dict:
    return {
        "observation.images.top": {
            "dtype": "image",
            "shape": image_shape,
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": ["state"],
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": ["action"],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Isaac Lab numpy episodes to LeRobot")
    parser.add_argument("--input-dir", type=Path, required=True, help="Path to Isaac Lab numpy dataset directory")
    parser.add_argument("--repo-id", type=str, required=True, help="LeRobot repo ID (e.g., my_org/isaac_cabinet)")
    parser.add_argument("--task", type=str, required=True, help="Task prompt to assign to all episodes")
    parser.add_argument("--robot-type", type=str, default="franka", help="Robot type metadata")
    parser.add_argument("--fps", type=int, default=30, help="Frame rate metadata")
    parser.add_argument("--image-key", type=str, default=None, help="Observation key to use as top camera")
    parser.add_argument("--push-to-hub", action="store_true", help="Push dataset to Hugging Face Hub")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()

    episode_dirs = _discover_episode_dirs(input_dir)
    sample_dir = episode_dirs[0]
    default_image_key, image_keys = _infer_keys(sample_dir)
    image_key = args.image_key or default_image_key
    if image_key not in image_keys:
        raise ValueError(f"image_key '{image_key}' not found in {sample_dir}. Available: {image_keys}")

    sample_images = np.load(sample_dir / f"{image_key}.npy")
    sample_state = np.load(sample_dir / "state.npy")
    sample_actions = np.load(sample_dir / "actions.npy")

    if sample_images.ndim != 4:
        raise ValueError(f"Expected image array shape (T,H,W,C), got {sample_images.shape}")

    image_shape = sample_images.shape[1:]
    state_dim = sample_state.shape[-1]
    action_dim = sample_actions.shape[-1]

    output_path = HF_LEROBOT_HOME / args.repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type=args.robot_type,
        fps=args.fps,
        features=_build_features(image_shape, state_dim, action_dim),
        image_writer_threads=4,
        image_writer_processes=2,
    )

    for ep_dir in episode_dirs:
        images = np.load(ep_dir / f"{image_key}.npy")
        states = np.load(ep_dir / "state.npy")
        actions = np.load(ep_dir / "actions.npy")

        num_frames = min(len(images), len(states), len(actions))
        for idx in range(num_frames):
            dataset.add_frame(
                {
                    "observation.images.top": images[idx],
                    "observation.state": states[idx],
                    "action": actions[idx],
                    "task": args.task,
                }
            )
        dataset.save_episode()

    if args.push_to_hub:
        dataset.push_to_hub()

    print(f"Converted {len(episode_dirs)} episodes to LeRobot dataset at {output_path}")


if __name__ == "__main__":
    main()
