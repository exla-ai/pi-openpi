"""Isaac Lab data adapter for RECAP training.

This module provides utilities to:
1. Load episodes collected from Isaac Lab
2. Convert them to the format expected by RECAP training
3. Compute advantages using the value function
"""

from __future__ import annotations

import os
import dataclasses
from typing import Dict, List, Optional, Iterator, Any
import logging

import h5py
import numpy as np
import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class RECAPEpisode:
    """Episode data formatted for RECAP training."""
    # Observations at each timestep
    images: Dict[str, np.ndarray]  # (T, H, W, C) for each camera
    state: np.ndarray  # (T, state_dim)

    # Actions
    actions: np.ndarray  # (T, action_dim)

    # For value function training
    time_to_completion: np.ndarray  # (T,) remaining steps at each timestep

    # For advantage computation
    rewards: np.ndarray  # (T,)
    success: bool

    # Episode info
    length: int

    def __post_init__(self):
        assert len(self.actions) == self.length
        assert len(self.time_to_completion) == self.length


class IsaacLabDataset:
    """Dataset of episodes collected from Isaac Lab."""

    def __init__(
        self,
        data_path: str,
        action_chunk_size: int = 50,
        cameras: List[str] = ["base_0_rgb"],
        image_size: tuple[int, int] = (224, 224),
    ):
        """Initialize dataset.

        Args:
            data_path: Path to HDF5 file or numpy directory
            action_chunk_size: Number of future actions to include (for pi0 training)
            cameras: List of camera names to include
            image_size: Expected image size (H, W)
        """
        self.data_path = data_path
        self.action_chunk_size = action_chunk_size
        self.cameras = cameras
        self.image_size = image_size

        self._episodes: List[RECAPEpisode] = []
        self._load_data()

        # Build index for random access
        self._build_index()

    def _load_data(self):
        """Load episodes from disk."""
        if self.data_path.endswith(".hdf5"):
            self._load_hdf5()
        else:
            self._load_numpy()

    def _load_hdf5(self):
        """Load from HDF5 format."""
        with h5py.File(self.data_path, "r") as f:
            num_episodes = f.attrs["num_episodes"]
            logger.info(f"Loading {num_episodes} episodes from {self.data_path}")

            for ep_idx in range(num_episodes):
                ep_group = f[f"episode_{ep_idx:06d}"]

                # Load observations
                obs_group = ep_group["observations"]
                images = {}
                for cam in self.cameras:
                    if cam in obs_group:
                        images[cam] = obs_group[cam][:]
                    else:
                        # Try alternate names
                        for key in obs_group.keys():
                            if "rgb" in key.lower() or "image" in key.lower():
                                images[cam] = obs_group[key][:]
                                break

                # Handle state - might be under different names
                state = None
                for key in ["state", "policy", "observation"]:
                    if key in obs_group:
                        state = obs_group[key][:]
                        break
                if state is None:
                    # Create dummy state if not present
                    length = ep_group.attrs["length"]
                    state = np.zeros((length, 1), dtype=np.float32)

                # Load other data
                actions = ep_group["actions"][:]
                rewards = ep_group["rewards"][:] if "rewards" in ep_group else np.zeros(len(actions))

                if "time_to_completion" in ep_group:
                    ttc = ep_group["time_to_completion"][:]
                else:
                    # Compute from episode length
                    length = len(actions)
                    ttc = np.array([length - t for t in range(length)], dtype=np.int32)

                episode = RECAPEpisode(
                    images=images,
                    state=state,
                    actions=actions,
                    time_to_completion=ttc,
                    rewards=rewards,
                    success=bool(ep_group.attrs.get("success", False)),
                    length=len(actions),
                )
                self._episodes.append(episode)

        logger.info(f"Loaded {len(self._episodes)} episodes")

    def _load_numpy(self):
        """Load from numpy directory format."""
        if os.path.isfile(self.data_path):
            raise ValueError(f"Expected directory, got file: {self.data_path}")

        # Find all episode directories
        episode_dirs = sorted([
            d for d in os.listdir(self.data_path)
            if d.startswith("episode_") and os.path.isdir(os.path.join(self.data_path, d))
        ])

        logger.info(f"Loading {len(episode_dirs)} episodes from {self.data_path}")

        for ep_dir_name in episode_dirs:
            ep_dir = os.path.join(self.data_path, ep_dir_name)

            # Load observations
            images = {}
            for cam in self.cameras:
                cam_path = os.path.join(ep_dir, f"{cam}.npy")
                if os.path.exists(cam_path):
                    images[cam] = np.load(cam_path)

            # Load state
            state_path = os.path.join(ep_dir, "state.npy")
            if os.path.exists(state_path):
                state = np.load(state_path)
            else:
                # Try policy observations
                for name in ["policy.npy", "observation.npy"]:
                    alt_path = os.path.join(ep_dir, name)
                    if os.path.exists(alt_path):
                        state = np.load(alt_path)
                        break
                else:
                    state = np.zeros((1, 1), dtype=np.float32)

            # Load actions
            actions = np.load(os.path.join(ep_dir, "actions.npy"))

            # Load rewards
            rewards_path = os.path.join(ep_dir, "rewards.npy")
            if os.path.exists(rewards_path):
                rewards = np.load(rewards_path)
            else:
                rewards = np.zeros(len(actions), dtype=np.float32)

            # Load metadata
            meta_path = os.path.join(ep_dir, "metadata.npy")
            if os.path.exists(meta_path):
                metadata = np.load(meta_path, allow_pickle=True).item()
                success = metadata.get("success", False)
                ttc = metadata.get("time_to_completion", None)
            else:
                success = False
                ttc = None

            if ttc is None:
                length = len(actions)
                ttc = np.array([length - t for t in range(length)], dtype=np.int32)
            else:
                ttc = np.array(ttc, dtype=np.int32)

            episode = RECAPEpisode(
                images=images,
                state=state,
                actions=actions,
                time_to_completion=ttc,
                rewards=rewards,
                success=success,
                length=len(actions),
            )
            self._episodes.append(episode)

        logger.info(f"Loaded {len(self._episodes)} episodes")

    def _build_index(self):
        """Build index for random access to timesteps."""
        self._timestep_index = []  # (episode_idx, timestep_idx)

        for ep_idx, episode in enumerate(self._episodes):
            # Only include timesteps where we have enough future actions
            max_t = episode.length - self.action_chunk_size
            for t in range(max(0, max_t)):
                self._timestep_index.append((ep_idx, t))

        logger.info(f"Built index with {len(self._timestep_index)} timesteps")

    def __len__(self) -> int:
        return len(self._timestep_index)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get a single training sample.

        Returns dict with:
            - images: Dict of camera images (H, W, C)
            - state: Robot state (state_dim,)
            - actions: Future action chunk (action_chunk_size, action_dim)
            - time_to_completion: Steps remaining (scalar)
            - improvement_indicator: Whether this is a "good" trajectory (bool)
        """
        ep_idx, t = self._timestep_index[index]
        episode = self._episodes[ep_idx]

        # Get observation at timestep t
        images = {cam: episode.images[cam][t] for cam in episode.images}
        state = episode.state[t]

        # Get action chunk starting at t
        action_end = min(t + self.action_chunk_size, episode.length)
        actions = episode.actions[t:action_end]

        # Pad if necessary
        if len(actions) < self.action_chunk_size:
            pad_length = self.action_chunk_size - len(actions)
            actions = np.concatenate([
                actions,
                np.tile(actions[-1:], (pad_length, 1))
            ], axis=0)

        # Time to completion
        ttc = episode.time_to_completion[t]

        # Use computed advantages if available, otherwise use episode success
        if hasattr(self, '_has_computed_advantages') and self._has_computed_advantages:
            improvement_indicator = bool(self._improvement_indicators[index])
        elif hasattr(episode, '_improvement_indicators'):
            improvement_indicator = bool(episode._improvement_indicators[t])
        else:
            improvement_indicator = episode.success

        return {
            "images": images,
            "state": state.astype(np.float32),
            "actions": actions.astype(np.float32),
            "time_to_completion": np.array(ttc, dtype=np.int32),
            "improvement_indicator": improvement_indicator,
        }

    @property
    def episodes(self) -> List[RECAPEpisode]:
        return self._episodes

    def get_episode(self, idx: int) -> RECAPEpisode:
        return self._episodes[idx]


class RECAPDataLoader:
    """Data loader for RECAP training."""

    def __init__(
        self,
        dataset: IsaacLabDataset,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

    def __iter__(self) -> Iterator[Dict[str, jnp.ndarray]]:
        """Iterate over batches."""
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            self.rng.shuffle(indices)

        for batch_start in range(0, len(indices), self.batch_size):
            batch_indices = indices[batch_start:batch_start + self.batch_size]

            if len(batch_indices) < self.batch_size:
                continue  # Skip incomplete batches

            samples = [self.dataset[i] for i in batch_indices]

            # Collate into batch
            batch = {}
            for key in samples[0]:
                if key == "images":
                    # Handle nested dict
                    batch["images"] = {}
                    for cam in samples[0]["images"]:
                        batch["images"][cam] = jnp.stack([s["images"][cam] for s in samples])
                else:
                    batch[key] = jnp.stack([s[key] for s in samples])

            yield batch

    def __len__(self) -> int:
        return len(self.dataset) // self.batch_size


def compute_advantages_with_value_function(
    dataset: IsaacLabDataset,
    value_function,
    batch_size: int = 32,
    tokenizer=None,
    prompt: str = "complete the task",
) -> IsaacLabDataset:
    """Compute advantages for each timestep using the trained value function.

    A(o_t) = V(o_t) - actual_time_to_completion[t]

    where V(o_t) is the value function's predicted expected time-to-completion.
    Positive advantage means this trajectory is doing BETTER than the policy average
    (will finish faster than expected).

    Args:
        dataset: Dataset with episodes
        value_function: Trained value function model
        batch_size: Batch size for value function inference
        tokenizer: Tokenizer for prompts (optional)
        prompt: Task prompt for conditioning

    Returns:
        Dataset with computed advantages and improvement indicators
    """
    from openpi.models.model import Observation

    logger.info("Computing advantages with value function...")
    logger.info(f"  Processing {len(dataset)} timesteps in batches of {batch_size}")

    all_advantages = []
    all_indicators = []

    # Process in batches for efficiency
    for batch_start in range(0, len(dataset), batch_size):
        batch_end = min(batch_start + batch_size, len(dataset))
        batch_indices = list(range(batch_start, batch_end))

        # Collect batch samples
        samples = [dataset[i] for i in batch_indices]

        # Build observation for value function
        images = {}
        for cam in samples[0]["images"]:
            images[cam] = jnp.stack([s["images"][cam] for s in samples])

        image_masks = {cam: jnp.ones(len(samples), dtype=jnp.bool_) for cam in images}
        state = jnp.stack([s["state"] for s in samples])
        actual_ttc = jnp.stack([s["time_to_completion"] for s in samples])

        observation = Observation(
            images=images,
            image_masks=image_masks,
            state=state,
            tokenized_prompt=None,
            tokenized_prompt_mask=None,
        )

        try:
            # Get value function predictions
            advantages = value_function.compute_advantage(observation, actual_ttc)
            advantages = np.array(advantages)
        except Exception as e:
            # Fallback: use heuristic based on episode success
            logger.debug(f"Falling back to heuristic advantages: {e}")
            advantages = np.array([
                1.0 if s.get("improvement_indicator", False) else -1.0
                for s in samples
            ])

        all_advantages.extend(advantages.tolist())

        if batch_start % (batch_size * 10) == 0:
            logger.info(f"  Processed {batch_start}/{len(dataset)} timesteps")

    # Store computed advantages in dataset
    dataset._advantages = np.array(all_advantages)
    dataset._improvement_indicators = dataset._advantages > 0
    dataset._has_computed_advantages = True

    # Update episodes with advantages
    idx = 0
    for episode in dataset.episodes:
        ep_len = episode.length - dataset.action_chunk_size
        if ep_len > 0:
            episode_advantages = dataset._advantages[idx:idx + ep_len]
            episode._advantages = np.zeros(episode.length)
            episode._advantages[:len(episode_advantages)] = episode_advantages
            episode._improvement_indicators = episode._advantages > 0
            idx += ep_len

    num_good = int(np.sum(dataset._improvement_indicators))
    num_bad = len(dataset) - num_good

    logger.info(f"Advantage computation complete:")
    logger.info(f"  Mean advantage: {np.mean(dataset._advantages):.4f}")
    logger.info(f"  Std advantage: {np.std(dataset._advantages):.4f}")
    logger.info(f"  Good samples (I=1): {num_good} ({100*num_good/len(dataset):.1f}%)")
    logger.info(f"  Bad samples (I=0): {num_bad} ({100*num_bad/len(dataset):.1f}%)")

    return dataset


def create_recap_dataset(
    data_path: str,
    action_chunk_size: int = 50,
    cameras: List[str] = ["base_0_rgb"],
    image_size: tuple[int, int] = (224, 224),
) -> IsaacLabDataset:
    """Create a RECAP dataset from collected Isaac Lab data.

    Args:
        data_path: Path to HDF5 file or numpy directory
        action_chunk_size: Number of future actions to include
        cameras: List of camera names to include
        image_size: Expected image size (H, W)

    Returns:
        Dataset ready for RECAP training
    """
    return IsaacLabDataset(
        data_path=data_path,
        action_chunk_size=action_chunk_size,
        cameras=cameras,
        image_size=image_size,
    )
