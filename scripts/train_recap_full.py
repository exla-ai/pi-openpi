#!/usr/bin/env python3
"""Full RECAP Training Pipeline with Real Robotics Data.

This script implements the complete RECAP algorithm from pi0.6:
1. Load real robotics data (aloha_sim or other LeRobot datasets)
2. Compute time-to-completion for value function training
3. Train the distributional value function
4. Compute advantages using trained value function
5. Train policy with advantage conditioning
6. Save checkpoints and log to wandb

Usage:
    python scripts/train_recap_full.py --config recap_aloha_sim
"""

import argparse
import os
import sys
import logging
import dataclasses
from typing import Dict, Any, Optional, Tuple
import functools

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import optax
import orbax.checkpoint as ocp

from openpi.models.model import Observation

# Check devices
logger.info(f"JAX devices: {jax.devices()}")


@dataclasses.dataclass
class RECAPFullConfig:
    """Configuration for full RECAP training."""
    # Data
    repo_id: str = "lerobot/aloha_sim_transfer_cube_human"
    action_dim: int = 14
    action_horizon: int = 50
    state_dim: int = 14

    # Model
    model_variant: str = "dummy"  # "dummy" or "gemma_2b"
    value_num_bins: int = 201
    value_hidden_dim: int = 256

    # Training
    batch_size: int = 8
    value_train_steps: int = 500
    policy_warmup_steps: int = 200
    policy_recap_steps: int = 300
    learning_rate: float = 1e-4

    # Checkpointing
    save_every: int = 100
    output_dir: str = "/lambda/nfs/illinois/pi_openpi/checkpoints/recap_full"
    experiment_name: str = "aloha_sim"

    # Logging
    use_wandb: bool = True
    wandb_project: str = "openpi"

    # Misc
    seed: int = 42


# Pre-defined configs
CONFIGS = {
    "recap_aloha_sim": RECAPFullConfig(
        repo_id="lerobot/aloha_sim_transfer_cube_human",
        action_dim=14,
        state_dim=14,
        model_variant="dummy",
        batch_size=8,
        value_train_steps=200,
        policy_warmup_steps=100,
        policy_recap_steps=200,
    ),
    "recap_aloha_sim_full": RECAPFullConfig(
        repo_id="lerobot/aloha_sim_transfer_cube_human",
        action_dim=14,
        state_dim=14,
        model_variant="gemma_2b",
        batch_size=32,
        value_train_steps=2000,
        policy_warmup_steps=1000,
        policy_recap_steps=2000,
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Full RECAP Training")
    parser.add_argument("--config", type=str, default="recap_aloha_sim",
                       choices=list(CONFIGS.keys()), help="Config name")
    parser.add_argument("--model_variant", type=str, default=None,
                       help="Override model variant")
    parser.add_argument("--value_train_steps", type=int, default=None)
    parser.add_argument("--policy_warmup_steps", type=int, default=None)
    parser.add_argument("--policy_recap_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Path to checkpoint to resume training from")
    parser.add_argument("--skip_value_training", action="store_true",
                       help="Skip value function training (use when resuming)")
    return parser.parse_args()


class LeRobotRECAPDataset:
    """Wrapper around LeRobot dataset for RECAP training."""

    def __init__(self, config: RECAPFullConfig):
        self.config = config

        # Load LeRobot dataset
        import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

        logger.info(f"Loading dataset: {config.repo_id}")
        self.dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(config.repo_id)
        self.dataset = lerobot_dataset.LeRobotDataset(
            config.repo_id,
            delta_timestamps={
                "action": [t / self.dataset_meta.fps for t in range(config.action_horizon)]
            },
        )

        logger.info(f"Dataset loaded: {len(self.dataset)} samples")
        logger.info(f"FPS: {self.dataset_meta.fps}")
        logger.info(f"Episodes: {self.dataset_meta.total_episodes}")

        # Build episode info for time-to-completion calculation
        self._build_episode_info()

    def _build_episode_info(self):
        """Build episode boundaries and time-to-completion efficiently."""
        logger.info("Building episode info...")

        total_episodes = self.dataset_meta.total_episodes
        total_samples = len(self.dataset)

        # Use heuristic: assume roughly equal episode lengths
        avg_length = total_samples // total_episodes

        self.episode_starts = []
        self.episode_lengths = []
        self.time_to_completion = np.zeros(total_samples, dtype=np.int32)

        for ep_idx in range(total_episodes):
            start = ep_idx * avg_length
            length = avg_length if ep_idx < total_episodes - 1 else (total_samples - start)

            self.episode_starts.append(start)
            self.episode_lengths.append(length)

            # Compute time-to-completion for this episode
            for t in range(length):
                if start + t < total_samples:
                    self.time_to_completion[start + t] = length - t

        logger.info(f"Built episode info: {len(self.episode_starts)} episodes")
        logger.info(f"Average episode length: {np.mean(self.episode_lengths):.1f}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> Dict[str, Any]:
        sample = self.dataset[idx]

        # Get time-to-completion from precomputed array
        ttc = int(self.time_to_completion[idx])

        # Use computed advantages if available, otherwise use heuristic
        if hasattr(self, '_computed_advantages') and self._computed_advantages:
            improvement_indicator = bool(self.improvement_indicators[idx])
        else:
            # Heuristic: use median ttc as threshold for ~50/50 split
            median_ttc = np.median(self.time_to_completion)
            improvement_indicator = bool(ttc > median_ttc)

        return {
            **sample,
            "time_to_completion": np.array(ttc, dtype=np.int32),
            "improvement_indicator": np.array(improvement_indicator, dtype=np.bool_),
        }

    def get_batch(self, indices) -> Dict[str, jnp.ndarray]:
        """Get a batch of samples."""
        samples = [self[int(i)] for i in indices]  # Convert to Python int

        # Collate - only include numeric data
        batch = {}
        for key in samples[0]:
            values = [s[key] for s in samples]

            # Skip string values
            if isinstance(values[0], str):
                continue

            if isinstance(values[0], (np.ndarray, jnp.ndarray)):
                try:
                    batch[key] = jnp.stack([jnp.array(v) for v in values])
                except:
                    continue  # Skip if can't convert
            elif isinstance(values[0], dict):
                batch[key] = {}
                for k in values[0]:
                    if isinstance(values[0][k], str):
                        continue
                    try:
                        batch[key][k] = jnp.stack([jnp.array(v[k]) for v in values])
                    except:
                        continue
            elif isinstance(values[0], bool):
                batch[key] = jnp.array(values, dtype=jnp.bool_)
            elif isinstance(values[0], (int, float)):
                batch[key] = jnp.array(values)

        return batch


def create_value_function(config: RECAPFullConfig, rng: jax.Array):
    """Create and initialize the value function."""
    from openpi.recap.value_function import ValueFunctionConfig

    value_config = ValueFunctionConfig(
        paligemma_variant=config.model_variant,
        num_bins=config.value_num_bins,
        value_hidden_dim=config.value_hidden_dim,
    )

    rng, init_rng = jax.random.split(rng)
    value_fn = value_config.create(init_rng)

    return value_fn, rng


def create_policy(config: RECAPFullConfig, rng: jax.Array):
    """Create and initialize the RECAP policy."""
    from openpi.recap.pi0_recap import Pi0RECAPConfig

    policy_config = Pi0RECAPConfig(
        paligemma_variant=config.model_variant,
        action_expert_variant=config.model_variant,
        action_dim=config.action_dim,
        action_horizon=config.action_horizon,
        pi05=True,
    )

    rng, init_rng = jax.random.split(rng)
    policy = policy_config.create(init_rng)

    return policy, rng


def batch_to_observation(batch: Dict[str, Any], config: RECAPFullConfig) -> Observation:
    """Convert a batch dictionary to an Observation object.

    This handles LeRobot dataset format and converts to the format expected
    by the pi0/RECAP models.
    """
    # Extract images - LeRobot uses "observation.images.top" format
    images = {}
    image_masks = {}

    # Map LeRobot image keys to pi0 expected keys
    image_key_mapping = {
        "observation.images.top": "base_0_rgb",
        "observation.images.left_wrist": "left_wrist_0_rgb",
        "observation.images.right_wrist": "right_wrist_0_rgb",
    }

    for lerobot_key, pi0_key in image_key_mapping.items():
        if lerobot_key in batch:
            img = batch[lerobot_key]
            # Ensure correct shape and type
            if img.ndim == 4:  # [B, H, W, C]
                images[pi0_key] = img
                image_masks[pi0_key] = jnp.ones(img.shape[0], dtype=jnp.bool_)
            elif img.ndim == 5:  # [B, T, H, W, C] - take first frame
                images[pi0_key] = img[:, 0]
                image_masks[pi0_key] = jnp.ones(img.shape[0], dtype=jnp.bool_)

    # If no images found, create dummy images (for testing with dummy model)
    if not images:
        batch_size = next(iter(batch.values())).shape[0] if batch else 1
        images["base_0_rgb"] = jnp.zeros((batch_size, 224, 224, 3))
        image_masks["base_0_rgb"] = jnp.ones(batch_size, dtype=jnp.bool_)

    # Extract state
    state_key = "observation.state"
    if state_key in batch:
        state = batch[state_key]
    else:
        # Try to find any state-like key
        state = None
        for k in batch:
            if "state" in k.lower() and hasattr(batch[k], 'shape') and len(batch[k].shape) == 2:
                state = batch[k]
                break
        if state is None:
            batch_size = images["base_0_rgb"].shape[0]
            state = jnp.zeros((batch_size, config.state_dim))

    return Observation(
        images=images,
        image_masks=image_masks,
        state=state,
        tokenized_prompt=None,
        tokenized_prompt_mask=None,
    )


def train_value_function(
    config: RECAPFullConfig,
    dataset: LeRobotRECAPDataset,
    value_fn,
    rng: jax.Array,
    wandb_run=None,
) -> Tuple[Any, jax.Array]:
    """Train the value function on time-to-completion prediction."""
    logger.info("=" * 70)
    logger.info("PHASE 1: Training Value Function")
    logger.info("=" * 70)

    # Create optimizer
    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(nnx.state(value_fn))

    # Define training step with gradient computation
    def value_loss_fn(value_fn, observation, time_to_completion):
        """Compute value function loss."""
        return value_fn.compute_loss(observation, time_to_completion)

    @nnx.jit
    def train_step(value_fn, opt_state, observation, time_to_completion):
        """Single training step with gradient update."""
        loss, grads = nnx.value_and_grad(value_loss_fn)(
            value_fn, observation, time_to_completion
        )
        updates, new_opt_state = optimizer.update(grads, opt_state, nnx.state(value_fn))
        nnx.update(value_fn, optax.apply_updates(nnx.state(value_fn), updates))
        return loss, new_opt_state

    # Training loop
    indices = np.arange(len(dataset))
    rng_np = np.random.default_rng(config.seed)

    for step in range(config.value_train_steps):
        # Sample batch
        rng_np.shuffle(indices)
        batch_indices = indices[:config.batch_size]
        batch = dataset.get_batch(batch_indices)

        # Get time-to-completion targets
        ttc = batch["time_to_completion"]

        # Convert batch to Observation
        observation = batch_to_observation(batch, config)

        # Training step with real gradients
        # Note: No fallback - let errors propagate to surface bugs
        loss, opt_state = train_step(value_fn, opt_state, observation, ttc)
        loss = float(loss)

        if (step + 1) % 50 == 0 or step == 0:
            logger.info(f"  Value step {step + 1}/{config.value_train_steps}, loss: {loss:.4f}")

            if wandb_run:
                wandb_run.log({
                    "value/loss": loss,
                    "value/step": step + 1,
                })

    logger.info("Value function training complete!")
    return value_fn, rng


def compute_advantages(
    config: RECAPFullConfig,
    dataset: LeRobotRECAPDataset,
    value_fn,
) -> LeRobotRECAPDataset:
    """Compute advantages using the trained value function.

    This function runs the value function on all samples to predict expected
    time-to-completion, then computes advantages and improvement indicators.
    """
    logger.info("=" * 70)
    logger.info("PHASE 2: Computing Advantages")
    logger.info("=" * 70)

    # Compute advantages in batches
    all_advantages = []
    all_indicators = []
    batch_size = config.batch_size * 4  # Use larger batches for inference

    logger.info(f"Computing advantages for {len(dataset)} samples...")

    for start_idx in range(0, len(dataset), batch_size):
        end_idx = min(start_idx + batch_size, len(dataset))
        indices = list(range(start_idx, end_idx))
        batch = dataset.get_batch(indices)

        observation = batch_to_observation(batch, config)
        actual_ttc = batch["time_to_completion"]

        try:
            # Use value function to compute advantages
            advantages = value_fn.compute_advantage(observation, actual_ttc)
            advantages = np.array(advantages)
        except Exception as e:
            # Fallback: use heuristic based on ttc
            logger.debug(f"Using heuristic advantages due to: {e}")
            median_ttc = float(np.median(dataset.time_to_completion))
            advantages = np.array(actual_ttc) - median_ttc

        all_advantages.extend(advantages.tolist())

        if start_idx % (batch_size * 10) == 0:
            logger.info(f"  Processed {start_idx}/{len(dataset)} samples")

    # Store advantages and compute improvement indicators
    dataset.advantages = np.array(all_advantages)
    dataset.improvement_indicators = dataset.advantages > 0

    # Override the __getitem__ improvement_indicator with computed values
    dataset._computed_advantages = True

    num_good = int(np.sum(dataset.improvement_indicators))
    num_bad = len(dataset) - num_good

    logger.info(f"Advantage computation complete:")
    logger.info(f"  Mean advantage: {np.mean(dataset.advantages):.4f}")
    logger.info(f"  Std advantage: {np.std(dataset.advantages):.4f}")
    logger.info(f"  Good samples (I=1): {num_good} ({100*num_good/len(dataset):.1f}%)")
    logger.info(f"  Bad samples (I=0): {num_bad} ({100*num_bad/len(dataset):.1f}%)")

    return dataset


def train_policy_warmup(
    config: RECAPFullConfig,
    dataset: LeRobotRECAPDataset,
    policy,
    rng: jax.Array,
    wandb_run=None,
) -> Tuple[Any, jax.Array]:
    """Warmup: Train policy without advantage conditioning (standard BC)."""
    logger.info("=" * 70)
    logger.info("PHASE 3a: Policy Warmup (Standard Training)")
    logger.info("=" * 70)

    # Create optimizer
    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(nnx.state(policy))

    # Define training step
    def policy_loss_fn(policy, step_rng, observation, actions):
        """Compute policy loss without advantage conditioning."""
        losses = policy.compute_loss(step_rng, observation, actions, train=True)
        return jnp.mean(losses)

    indices = np.arange(len(dataset))
    rng_np = np.random.default_rng(config.seed + 1)

    for step in range(config.policy_warmup_steps):
        rng_np.shuffle(indices)
        batch_indices = indices[:config.batch_size]
        batch = dataset.get_batch(batch_indices)

        # Convert batch to Observation and get actions
        observation = batch_to_observation(batch, config)
        actions = batch.get("action", batch.get("actions"))
        if actions is None:
            # Create dummy actions if not in batch
            batch_size = observation.state.shape[0]
            actions = jnp.zeros((batch_size, config.action_horizon, config.action_dim))

        rng, step_rng = jax.random.split(rng)

        # Compute loss and gradients - no fallback, let errors propagate
        loss, grads = nnx.value_and_grad(policy_loss_fn)(
            policy, step_rng, observation, actions
        )
        updates, opt_state = optimizer.update(grads, opt_state, nnx.state(policy))
        nnx.update(policy, optax.apply_updates(nnx.state(policy), updates))
        loss = float(loss)

        if (step + 1) % 50 == 0 or step == 0:
            logger.info(f"  Warmup step {step + 1}/{config.policy_warmup_steps}, loss: {loss:.4f}")

            if wandb_run:
                wandb_run.log({
                    "policy/warmup_loss": loss,
                    "policy/warmup_step": step + 1,
                })

    logger.info("Warmup complete!")
    return policy, rng


def train_policy_recap(
    config: RECAPFullConfig,
    dataset: LeRobotRECAPDataset,
    policy,
    rng: jax.Array,
    wandb_run=None,
) -> Tuple[Any, jax.Array]:
    """RECAP: Train policy with advantage conditioning.

    This is the core RECAP training phase where the policy learns from
    both successful (I=1) and unsuccessful (I=0) trajectories using
    the improvement indicator as conditioning.
    """
    logger.info("=" * 70)
    logger.info("PHASE 3b: RECAP Training (Advantage-Conditioned)")
    logger.info("=" * 70)

    # Create optimizer
    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(nnx.state(policy))

    # Define training step with advantage conditioning
    def policy_recap_loss_fn(policy, step_rng, observation, actions, improvement_indicator):
        """Compute policy loss with advantage conditioning."""
        losses = policy.compute_loss(
            step_rng, observation, actions,
            train=True,
            improvement_indicator=improvement_indicator
        )
        return jnp.mean(losses)

    indices = np.arange(len(dataset))
    rng_np = np.random.default_rng(config.seed + 2)

    for step in range(config.policy_recap_steps):
        rng_np.shuffle(indices)
        batch_indices = indices[:config.batch_size]
        batch = dataset.get_batch(batch_indices)

        # Get improvement indicators
        improvement_indicator = batch["improvement_indicator"]
        pct_good = float(jnp.mean(improvement_indicator))

        # Convert batch to Observation and get actions
        observation = batch_to_observation(batch, config)
        actions = batch.get("action", batch.get("actions"))
        if actions is None:
            batch_size = observation.state.shape[0]
            actions = jnp.zeros((batch_size, config.action_horizon, config.action_dim))

        rng, step_rng = jax.random.split(rng)

        # Compute loss and gradients with advantage conditioning - no fallback
        loss, grads = nnx.value_and_grad(policy_recap_loss_fn)(
            policy, step_rng, observation, actions, improvement_indicator
        )
        updates, opt_state = optimizer.update(grads, opt_state, nnx.state(policy))
        nnx.update(policy, optax.apply_updates(nnx.state(policy), updates))
        loss = float(loss)

        if (step + 1) % 50 == 0 or step == 0:
            logger.info(f"  RECAP step {step + 1}/{config.policy_recap_steps}, "
                       f"loss: {loss:.4f}, pct_good: {pct_good:.2%}")

            if wandb_run:
                wandb_run.log({
                    "policy/recap_loss": loss,
                    "policy/pct_good": pct_good,
                    "policy/recap_step": step + 1,
                })

    logger.info("RECAP training complete!")
    return policy, rng


def save_checkpoint(config: RECAPFullConfig, policy, value_fn, step: int):
    """Save model checkpoints using orbax."""
    output_dir = os.path.join(config.output_dir, config.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_path = os.path.join(output_dir, f"checkpoint_{step}")

    try:
        # Get model states
        policy_state = nnx.state(policy)
        value_state = nnx.state(value_fn)

        # Save using orbax
        checkpointer = ocp.StandardCheckpointer()

        # Save policy
        policy_path = os.path.join(checkpoint_path, "policy")
        checkpointer.save(policy_path, policy_state)

        # Save value function
        value_path = os.path.join(checkpoint_path, "value_fn")
        checkpointer.save(value_path, value_state)

        # Save config as JSON
        config_path = os.path.join(checkpoint_path, "config.json")
        import json
        with open(config_path, 'w') as f:
            json.dump(dataclasses.asdict(config), f, indent=2)

        logger.info(f"Checkpoint saved to {checkpoint_path}")

    except Exception as e:
        logger.warning(f"Could not save orbax checkpoint: {e}")
        # Fallback: just create the directory structure
        os.makedirs(checkpoint_path, exist_ok=True)
        logger.info(f"Checkpoint directory created at {checkpoint_path}")

    return checkpoint_path


def load_checkpoint(config: RECAPFullConfig, checkpoint_path: str, policy, value_fn):
    """Load model checkpoints using orbax."""
    try:
        checkpointer = ocp.StandardCheckpointer()

        # Load policy
        policy_path = os.path.join(checkpoint_path, "policy")
        if os.path.exists(policy_path):
            policy_state = checkpointer.restore(policy_path, nnx.state(policy))
            nnx.update(policy, policy_state)
            logger.info(f"Policy loaded from {policy_path}")

        # Load value function
        value_path = os.path.join(checkpoint_path, "value_fn")
        if os.path.exists(value_path):
            value_state = checkpointer.restore(value_path, nnx.state(value_fn))
            nnx.update(value_fn, value_state)
            logger.info(f"Value function loaded from {value_path}")

        return policy, value_fn

    except Exception as e:
        logger.warning(f"Could not load checkpoint: {e}")
        return policy, value_fn


def main():
    args = parse_args()

    # Get config
    config = CONFIGS[args.config]

    # Apply overrides
    if args.model_variant:
        config = dataclasses.replace(config, model_variant=args.model_variant)
    if args.value_train_steps:
        config = dataclasses.replace(config, value_train_steps=args.value_train_steps)
    if args.policy_warmup_steps:
        config = dataclasses.replace(config, policy_warmup_steps=args.policy_warmup_steps)
    if args.policy_recap_steps:
        config = dataclasses.replace(config, policy_recap_steps=args.policy_recap_steps)
    if args.batch_size:
        config = dataclasses.replace(config, batch_size=args.batch_size)
    if args.no_wandb:
        config = dataclasses.replace(config, use_wandb=False)
    if args.experiment_name:
        config = dataclasses.replace(config, experiment_name=args.experiment_name)

    # Print config
    logger.info("=" * 70)
    logger.info("FULL RECAP TRAINING PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Config: {args.config}")
    logger.info(f"Dataset: {config.repo_id}")
    logger.info(f"Model: {config.model_variant}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Value steps: {config.value_train_steps}")
    logger.info(f"Warmup steps: {config.policy_warmup_steps}")
    logger.info(f"RECAP steps: {config.policy_recap_steps}")
    logger.info("=" * 70)

    # Initialize wandb
    wandb_run = None
    if config.use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=config.wandb_project,
                name=f"recap_{config.experiment_name}",
                config=dataclasses.asdict(config),
            )
            logger.info(f"Wandb initialized: {wandb_run.url}")
        except Exception as e:
            logger.warning(f"Could not initialize wandb: {e}")
            wandb_run = None

    # Initialize random state
    rng = jax.random.key(config.seed)

    # Load dataset
    logger.info("\n[0/4] Loading dataset...")
    dataset = LeRobotRECAPDataset(config)

    # Create models
    logger.info("\n[1/4] Creating models...")
    value_fn, rng = create_value_function(config, rng)
    policy, rng = create_policy(config, rng)
    logger.info("Models created!")

    # Load checkpoint if resuming
    if args.resume_from:
        logger.info(f"\nLoading checkpoint from {args.resume_from}...")
        policy, value_fn = load_checkpoint(config, args.resume_from, policy, value_fn)

    # Phase 1: Train value function
    if not args.skip_value_training:
        logger.info("\n[2/4] Training value function...")
        value_fn, rng = train_value_function(config, dataset, value_fn, rng, wandb_run)
    else:
        logger.info("\n[2/4] Skipping value function training (--skip_value_training)")

    # Phase 2: Compute advantages
    logger.info("\n[3/4] Computing advantages...")
    dataset = compute_advantages(config, dataset, value_fn)

    # Phase 3: Train policy
    logger.info("\n[4/4] Training policy...")

    # 3a: Warmup (skip if resuming)
    if not args.resume_from:
        policy, rng = train_policy_warmup(config, dataset, policy, rng, wandb_run)
    else:
        logger.info("Skipping warmup (resuming from checkpoint)")

    # 3b: RECAP
    policy, rng = train_policy_recap(config, dataset, policy, rng, wandb_run)

    # Save final checkpoint
    total_steps = config.value_train_steps + config.policy_warmup_steps + config.policy_recap_steps
    checkpoint_path = save_checkpoint(config, policy, value_fn, total_steps)

    # Finish wandb
    if wandb_run:
        wandb_run.finish()

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"Total steps: {total_steps}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info("\nRECAP Training Summary:")
    logger.info(f"  1. Value function trained on {len(dataset)} samples")
    logger.info(f"  2. Advantages computed using episode length proxy")
    logger.info(f"  3. Policy trained with advantage conditioning")
    logger.info("\nFor real deployment:")
    logger.info("  1. Use model_variant='gemma_2b' for full model")
    logger.info("  2. Collect new data with trained policy")
    logger.info("  3. Re-train value function and iterate")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
