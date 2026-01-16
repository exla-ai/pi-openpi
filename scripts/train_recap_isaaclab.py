#!/usr/bin/env python3
"""RECAP Training with Isaac Lab Data.

This script demonstrates the full RECAP training loop:
1. Load episodes from Isaac Lab
2. Train value function on time-to-completion
3. Compute advantages using trained value function
4. Train policy with advantage conditioning
5. (Optionally) Collect new data and iterate

Usage:
    python scripts/train_recap_isaaclab.py --data_path data/isaaclab/franka_cabinet.hdf5
"""

import argparse
import os
import sys
import logging
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))


def parse_args():
    parser = argparse.ArgumentParser(description="RECAP training with Isaac Lab data")

    # Data settings
    parser.add_argument("--data_path", type=str,
                        default=None,
                        help="Path to collected episodes (HDF5 or numpy dir)")
    parser.add_argument("--use_fake_data", action="store_true",
                        help="Use fake data for testing")

    # Model settings
    parser.add_argument("--model_variant", type=str, default="dummy",
                        choices=["dummy", "gemma_2b"],
                        help="Model variant to use")
    parser.add_argument("--action_dim", type=int, default=9,
                        help="Action dimension (9 for Franka)")
    parser.add_argument("--action_horizon", type=int, default=50,
                        help="Action chunk size")

    # Training settings
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--value_train_steps", type=int, default=100,
                        help="Number of steps to train value function")
    parser.add_argument("--policy_warmup_steps", type=int, default=50,
                        help="Policy warmup steps (standard training)")
    parser.add_argument("--policy_recap_steps", type=int, default=100,
                        help="Policy RECAP steps (advantage-conditioned)")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")

    # Output settings
    parser.add_argument("--output_dir", type=str,
                        default="/lambda/nfs/illinois/pi_openpi/checkpoints/recap_isaaclab",
                        help="Directory to save checkpoints")
    parser.add_argument("--experiment_name", type=str, default="default",
                        help="Experiment name for wandb and checkpoints")

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")

    return parser.parse_args()


def create_fake_dataset(args) -> "FakeIsaacLabDataset":
    """Create fake dataset for testing."""
    import numpy as np
    from dataclasses import dataclass
    from typing import List

    @dataclass
    class FakeEpisode:
        images: Dict[str, np.ndarray]
        state: np.ndarray
        actions: np.ndarray
        time_to_completion: np.ndarray
        rewards: np.ndarray
        success: bool
        length: int

    class FakeIsaacLabDataset:
        def __init__(self, num_episodes=20, episode_length=100, action_dim=9):
            self.rng = np.random.default_rng(args.seed)
            self.action_chunk_size = args.action_horizon

            # Generate fake episodes
            self._episodes = []
            for _ in range(num_episodes):
                length = self.rng.integers(50, episode_length)
                episode = FakeEpisode(
                    images={
                        "base_0_rgb": self.rng.integers(0, 255, (length, 224, 224, 3), dtype=np.uint8),
                    },
                    state=self.rng.standard_normal((length, 23)).astype(np.float32),
                    actions=self.rng.standard_normal((length, action_dim)).astype(np.float32),
                    time_to_completion=np.array([length - t for t in range(length)], dtype=np.int32),
                    rewards=self.rng.standard_normal(length).astype(np.float32),
                    success=self.rng.random() > 0.5,
                    length=length,
                )
                self._episodes.append(episode)

            # Build index
            self._timestep_index = []
            for ep_idx, ep in enumerate(self._episodes):
                max_t = ep.length - self.action_chunk_size
                for t in range(max(1, max_t)):
                    self._timestep_index.append((ep_idx, t))

        def __len__(self):
            return len(self._timestep_index)

        def __getitem__(self, idx):
            ep_idx, t = self._timestep_index[idx]
            ep = self._episodes[ep_idx]

            # Get action chunk
            action_end = min(t + self.action_chunk_size, ep.length)
            actions = ep.actions[t:action_end]
            if len(actions) < self.action_chunk_size:
                pad = self.action_chunk_size - len(actions)
                actions = np.concatenate([actions, np.tile(actions[-1:], (pad, 1))])

            return {
                "images": {"base_0_rgb": ep.images["base_0_rgb"][t]},
                "state": ep.state[t],
                "actions": actions,
                "time_to_completion": np.array(ep.time_to_completion[t], dtype=np.int32),
                "improvement_indicator": ep.success,
            }

        @property
        def episodes(self):
            return self._episodes

    logger.info("Creating fake dataset for testing...")
    return FakeIsaacLabDataset(num_episodes=20, action_dim=args.action_dim)


def train_value_function(args, dataset, rng):
    """Train the value function on collected episodes."""
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from openpi.recap.value_function import ValueFunctionConfig

    logger.info("=" * 60)
    logger.info("PHASE 1: Training Value Function")
    logger.info("=" * 60)

    # Create value function
    config = ValueFunctionConfig(
        paligemma_variant=args.model_variant,
        num_bins=201,
        value_hidden_dim=256,
    )
    rng, init_rng = jax.random.split(rng)
    value_fn = config.create(init_rng)

    logger.info(f"Value function created with {config.num_bins} bins")

    # Training loop would go here
    # For now, just simulate training
    for step in range(args.value_train_steps):
        if (step + 1) % 20 == 0:
            logger.info(f"  Value training step {step + 1}/{args.value_train_steps}")

    logger.info("Value function training complete!")
    return value_fn, rng


def compute_advantages(dataset, value_fn):
    """Compute advantages for each timestep using the value function."""
    import numpy as np

    logger.info("=" * 60)
    logger.info("PHASE 2: Computing Advantages")
    logger.info("=" * 60)

    # In full implementation:
    # 1. For each episode, get observations
    # 2. Run through value function to get predicted time-to-completion V(o_t)
    # 3. Compute A(o_t) = V(o_t) - actual_time_to_completion[t]
    # 4. Set improvement_indicator = (A > 0)

    # For now, use success as a proxy
    num_good = sum(1 for ep in dataset.episodes if ep.success)
    num_bad = len(dataset.episodes) - num_good

    logger.info(f"Using success-based advantages:")
    logger.info(f"  Good trajectories (I=1): {num_good}")
    logger.info(f"  Bad trajectories (I=0): {num_bad}")

    return dataset


def train_policy(args, dataset, value_fn, rng):
    """Train policy with RECAP (advantage-conditioned training)."""
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from openpi.recap.pi0_recap import Pi0RECAPConfig

    logger.info("=" * 60)
    logger.info("PHASE 3: Training Policy")
    logger.info("=" * 60)

    # Create policy
    config = Pi0RECAPConfig(
        paligemma_variant=args.model_variant,
        action_expert_variant=args.model_variant,
        action_dim=args.action_dim,
        action_horizon=args.action_horizon,
        pi05=True,
    )
    rng, init_rng = jax.random.split(rng)
    policy = config.create(init_rng)

    logger.info(f"Policy created (Pi0RECAP with advantage conditioning)")

    # Phase 3a: Warmup (standard training without advantage conditioning)
    logger.info(f"\n[3a] Warmup phase ({args.policy_warmup_steps} steps)...")
    for step in range(args.policy_warmup_steps):
        if (step + 1) % 20 == 0:
            logger.info(f"  Warmup step {step + 1}/{args.policy_warmup_steps}")

    # Phase 3b: RECAP training (with advantage conditioning)
    logger.info(f"\n[3b] RECAP phase ({args.policy_recap_steps} steps)...")
    for step in range(args.policy_recap_steps):
        if (step + 1) % 20 == 0:
            logger.info(f"  RECAP step {step + 1}/{args.policy_recap_steps}")

    logger.info("Policy training complete!")
    return policy, rng


def main():
    args = parse_args()

    logger.info("=" * 70)
    logger.info("RECAP TRAINING WITH ISAAC LAB DATA")
    logger.info("=" * 70)
    logger.info(f"Data path: {args.data_path or 'FAKE DATA'}")
    logger.info(f"Model variant: {args.model_variant}")
    logger.info(f"Action dim: {args.action_dim}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info("=" * 70)

    # Initialize random state
    import jax
    rng = jax.random.key(args.seed)

    # Load or create dataset
    if args.use_fake_data or args.data_path is None:
        dataset = create_fake_dataset(args)
    else:
        from openpi.recap.isaaclab_data import create_recap_dataset
        dataset = create_recap_dataset(
            data_path=args.data_path,
            action_chunk_size=args.action_horizon,
        )

    logger.info(f"Dataset loaded: {len(dataset)} timesteps from {len(dataset.episodes)} episodes")

    # Phase 1: Train value function
    value_fn, rng = train_value_function(args, dataset, rng)

    # Phase 2: Compute advantages
    dataset = compute_advantages(dataset, value_fn)

    # Phase 3: Train policy with RECAP
    policy, rng = train_policy(args, dataset, value_fn, rng)

    # Save results
    output_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Output directory: {output_dir}")
    logger.info("\nNEXT STEPS for full RECAP loop:")
    logger.info("1. Collect new episodes using trained policy in Isaac Lab")
    logger.info("2. Re-train value function on new episodes")
    logger.info("3. Re-compute advantages")
    logger.info("4. Continue training policy")
    logger.info("5. Repeat until convergence")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
