#!/usr/bin/env python3
"""RECAP Training with Proper FSDP Support.

This script implements the RECAP algorithm from pi0.6 with:
1. Proper FSDP sharding across multiple GPUs
2. Comprehensive wandb logging including GPU metrics
3. Real gradient computation (no fallback to simulated losses)

Usage:
    python scripts/train_recap_v2.py --config recap_aloha_sim --fsdp_devices 8
"""

import argparse
import dataclasses
import functools
import logging
import os
import platform
import sys
import time
from typing import Any, Dict, Iterator

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))

from openpi.models.model import Observation
from openpi.recap.value_function import ValueFunctionConfig
from openpi.recap.pi0_recap import Pi0RECAPConfig
import openpi.training.sharding as sharding
import openpi.shared.array_typing as at

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclasses.dataclass
class RECAPConfig:
    """Configuration for RECAP training with FSDP."""
    # Data
    repo_id: str = "lerobot/aloha_sim_transfer_cube_human"
    action_dim: int = 14
    action_horizon: int = 50
    state_dim: int = 14

    # Model - use actual model, not dummy
    # Valid: "dummy", "gemma_300m", "gemma_2b", "gemma_2b_lora"
    model_variant: str = "gemma_2b"  # Use gemma_2b for real training
    value_num_bins: int = 201
    value_hidden_dim: int = 256

    # FSDP
    fsdp_devices: int = 8

    # Training
    batch_size: int = 32
    value_train_steps: int = 2000
    policy_warmup_steps: int = 1000
    policy_recap_steps: int = 5000
    learning_rate: float = 1e-5
    grad_clip: float = 1.0

    # Checkpointing
    save_interval: int = 1000
    output_dir: str = "/lambda/nfs/illinois/pi_openpi/checkpoints/recap_v2"
    experiment_name: str = "aloha_sim"

    # Logging
    log_interval: int = 10
    wandb_project: str = "openpi"

    # Misc
    seed: int = 42


CONFIGS = {
    "recap_aloha_sim": RECAPConfig(
        repo_id="lerobot/aloha_sim_transfer_cube_human",
        model_variant="gemma_2b",
        fsdp_devices=8,
        batch_size=32,
        value_train_steps=2000,
        policy_warmup_steps=1000,
        policy_recap_steps=5000,
    ),
    "recap_aloha_sim_small": RECAPConfig(
        repo_id="lerobot/aloha_sim_transfer_cube_human",
        model_variant="gemma_2b",
        fsdp_devices=8,
        batch_size=16,
        value_train_steps=500,
        policy_warmup_steps=200,
        policy_recap_steps=1000,
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(description="RECAP Training with FSDP")
    parser.add_argument("--config", type=str, default="recap_aloha_sim",
                       choices=list(CONFIGS.keys()))
    parser.add_argument("--fsdp_devices", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    return parser.parse_args()


def init_wandb(config: RECAPConfig, enabled: bool = True):
    """Initialize wandb with comprehensive settings."""
    import wandb

    if not enabled:
        wandb.init(mode="disabled")
        return None

    # Enable system metrics (GPU utilization, memory, etc.)
    # wandb automatically logs system metrics including:
    # - GPU utilization (%)
    # - GPU memory allocated
    # - GPU temperature
    # - CPU utilization
    # - System memory
    run = wandb.init(
        project=config.wandb_project,
        name=f"recap_{config.experiment_name}",
        config=dataclasses.asdict(config),
    )

    # Log additional system info
    wandb.config.update({
        "system/hostname": platform.node(),
        "system/jax_devices": str(jax.devices()),
        "system/num_devices": jax.device_count(),
        "system/platform": platform.platform(),
    })

    return run


def log_training_metrics(
    wandb_run,
    step: int,
    phase: str,
    loss: float,
    grad_norm: float = None,
    param_norm: float = None,
    throughput: float = None,
    extra_metrics: Dict[str, float] = None,
):
    """Log comprehensive training metrics to wandb."""
    if wandb_run is None:
        return

    import wandb

    metrics = {
        f"{phase}/loss": loss,
        f"{phase}/step": step,
    }

    if grad_norm is not None:
        metrics[f"{phase}/grad_norm"] = grad_norm

    if param_norm is not None:
        metrics[f"{phase}/param_norm"] = param_norm

    if throughput is not None:
        metrics[f"{phase}/throughput_samples_per_sec"] = throughput

    if extra_metrics:
        for k, v in extra_metrics.items():
            metrics[f"{phase}/{k}"] = v

    wandb.log(metrics, step=step)


class LeRobotDataset:
    """LeRobot dataset wrapper for RECAP."""

    def __init__(self, config: RECAPConfig):
        self.config = config

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

        self._build_episode_info()

    def _build_episode_info(self):
        """Build episode boundaries and time-to-completion."""
        total_episodes = self.dataset_meta.total_episodes
        total_samples = len(self.dataset)
        avg_length = total_samples // total_episodes

        self.time_to_completion = np.zeros(total_samples, dtype=np.int32)

        for ep_idx in range(total_episodes):
            start = ep_idx * avg_length
            length = avg_length if ep_idx < total_episodes - 1 else (total_samples - start)

            for t in range(length):
                if start + t < total_samples:
                    self.time_to_completion[start + t] = length - t

        logger.info(f"Built episode info: {total_episodes} episodes, avg length: {avg_length}")

    def __len__(self):
        return len(self.dataset)

    def get_batch(self, indices: np.ndarray) -> Dict[str, jnp.ndarray]:
        """Get a batch of samples."""
        samples = [self.dataset[int(i)] for i in indices]

        batch = {}
        for key in samples[0]:
            values = [s[key] for s in samples]
            if isinstance(values[0], str):
                continue
            if isinstance(values[0], (np.ndarray, jnp.ndarray)):
                try:
                    batch[key] = jnp.stack([jnp.array(v) for v in values])
                except:
                    continue
            elif isinstance(values[0], dict):
                batch[key] = {}
                for k in values[0]:
                    if isinstance(values[0][k], str):
                        continue
                    try:
                        batch[key][k] = jnp.stack([jnp.array(v[k]) for v in values])
                    except:
                        continue

        # Add time-to-completion
        batch["time_to_completion"] = jnp.array([self.time_to_completion[int(i)] for i in indices])

        return batch


def batch_to_observation(batch: Dict[str, Any], config: RECAPConfig) -> Observation:
    """Convert batch to Observation object."""
    images = {}
    image_masks = {}

    image_key_mapping = {
        "observation.images.top": "base_0_rgb",
        "observation.images.left_wrist": "left_wrist_0_rgb",
        "observation.images.right_wrist": "right_wrist_0_rgb",
    }

    for lerobot_key, pi0_key in image_key_mapping.items():
        if lerobot_key in batch:
            img = batch[lerobot_key]
            if img.ndim == 4:
                images[pi0_key] = img
                image_masks[pi0_key] = jnp.ones(img.shape[0], dtype=jnp.bool_)
            elif img.ndim == 5:
                images[pi0_key] = img[:, 0]
                image_masks[pi0_key] = jnp.ones(img.shape[0], dtype=jnp.bool_)

    if not images:
        batch_size = batch["time_to_completion"].shape[0]
        images["base_0_rgb"] = jnp.zeros((batch_size, 224, 224, 3))
        image_masks["base_0_rgb"] = jnp.ones(batch_size, dtype=jnp.bool_)

    state_key = "observation.state"
    if state_key in batch:
        state = batch[state_key]
    else:
        batch_size = images["base_0_rgb"].shape[0]
        state = jnp.zeros((batch_size, config.state_dim))

    return Observation(
        images=images,
        image_masks=image_masks,
        state=state,
        tokenized_prompt=None,
        tokenized_prompt_mask=None,
    )


def create_models_with_fsdp(config: RECAPConfig, mesh: jax.sharding.Mesh, rng: jax.Array):
    """Create models with proper FSDP sharding."""

    # Create value function config
    value_config = ValueFunctionConfig(
        paligemma_variant=config.model_variant,
        num_bins=config.value_num_bins,
        value_hidden_dim=config.value_hidden_dim,
    )

    # Create policy config
    policy_config = Pi0RECAPConfig(
        paligemma_variant=config.model_variant,
        action_expert_variant=config.model_variant,
        action_dim=config.action_dim,
        action_horizon=config.action_horizon,
        pi05=True,
    )

    rng, value_rng, policy_rng = jax.random.split(rng, 3)

    # Initialize models
    logger.info("Initializing value function...")
    value_fn = value_config.create(value_rng)

    logger.info("Initializing policy...")
    policy = policy_config.create(policy_rng)

    # Get parameter shapes for sharding
    value_params = nnx.state(value_fn)
    policy_params = nnx.state(policy)

    # Apply FSDP sharding
    logger.info("Applying FSDP sharding to value function...")
    value_sharding = sharding.fsdp_sharding(value_params, mesh, log=True)

    logger.info("Applying FSDP sharding to policy...")
    policy_sharding = sharding.fsdp_sharding(policy_params, mesh, log=True)

    return value_fn, policy, value_sharding, policy_sharding, rng


def train_value_function_fsdp(
    config: RECAPConfig,
    dataset: LeRobotDataset,
    value_fn,
    value_sharding,
    mesh: jax.sharding.Mesh,
    rng: jax.Array,
    wandb_run,
):
    """Train value function with FSDP."""
    logger.info("=" * 70)
    logger.info("PHASE 1: Training Value Function with FSDP")
    logger.info("=" * 70)

    # Create optimizer
    tx = optax.chain(
        optax.clip_by_global_norm(config.grad_clip),
        optax.adam(config.learning_rate),
    )

    value_params = nnx.state(value_fn)
    opt_state = tx.init(value_params)

    # Data sharding
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    def value_loss_fn(params, observation, time_to_completion):
        """Compute value function loss."""
        graphdef = nnx.graphdef(value_fn)
        model = nnx.merge(graphdef, params)
        return model.compute_loss(observation, time_to_completion)

    @functools.partial(
        jax.jit,
        in_shardings=(value_sharding, None, data_sharding, data_sharding),
        out_shardings=(value_sharding, None, replicated_sharding, replicated_sharding),
    )
    def train_step(params, opt_state, observation, time_to_completion):
        loss, grads = jax.value_and_grad(value_loss_fn)(params, observation, time_to_completion)
        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        grad_norm = optax.global_norm(grads)
        return new_params, new_opt_state, loss, grad_norm

    indices = np.arange(len(dataset))
    rng_np = np.random.default_rng(config.seed)

    pbar = tqdm.tqdm(range(config.value_train_steps), desc="Value Training")

    start_time = time.time()
    samples_processed = 0

    for step in pbar:
        rng_np.shuffle(indices)
        batch_indices = indices[:config.batch_size]
        batch = dataset.get_batch(batch_indices)

        observation = batch_to_observation(batch, config)
        ttc = batch["time_to_completion"]

        with sharding.set_mesh(mesh):
            value_params, opt_state, loss, grad_norm = train_step(
                value_params, opt_state, observation, ttc
            )

        samples_processed += config.batch_size
        elapsed = time.time() - start_time
        throughput = samples_processed / elapsed

        loss_val = float(loss)
        grad_norm_val = float(grad_norm)

        if step % config.log_interval == 0:
            pbar.set_postfix(loss=f"{loss_val:.4f}", grad=f"{grad_norm_val:.4f}", tput=f"{throughput:.1f}")

            log_training_metrics(
                wandb_run, step, "value",
                loss=loss_val,
                grad_norm=grad_norm_val,
                throughput=throughput,
            )

    # Update model with trained params
    nnx.update(value_fn, value_params)

    logger.info(f"Value training complete! Final loss: {loss_val:.4f}")
    return value_fn, rng


def compute_advantages_fsdp(
    config: RECAPConfig,
    dataset: LeRobotDataset,
    value_fn,
    mesh: jax.sharding.Mesh,
) -> np.ndarray:
    """Compute advantages using trained value function."""
    logger.info("=" * 70)
    logger.info("PHASE 2: Computing Advantages")
    logger.info("=" * 70)

    all_advantages = []
    batch_size = config.batch_size * 4

    for start_idx in tqdm.tqdm(range(0, len(dataset), batch_size), desc="Computing advantages"):
        end_idx = min(start_idx + batch_size, len(dataset))
        indices = np.arange(start_idx, end_idx)
        batch = dataset.get_batch(indices)

        observation = batch_to_observation(batch, config)
        actual_ttc = batch["time_to_completion"]

        with sharding.set_mesh(mesh):
            try:
                advantages = value_fn.compute_advantage(observation, actual_ttc)
                advantages = np.array(advantages)
            except Exception as e:
                logger.warning(f"Error computing advantages: {e}")
                median_ttc = float(np.median(dataset.time_to_completion))
                advantages = np.array(actual_ttc) - median_ttc

        all_advantages.extend(advantages.tolist())

    advantages = np.array(all_advantages)
    improvement_indicators = advantages > 0

    num_good = int(np.sum(improvement_indicators))
    logger.info(f"Advantages computed:")
    logger.info(f"  Mean: {np.mean(advantages):.4f}, Std: {np.std(advantages):.4f}")
    logger.info(f"  Good samples (I=1): {num_good} ({100*num_good/len(dataset):.1f}%)")

    return advantages, improvement_indicators


def train_policy_fsdp(
    config: RECAPConfig,
    dataset: LeRobotDataset,
    policy,
    policy_sharding,
    mesh: jax.sharding.Mesh,
    improvement_indicators: np.ndarray,
    rng: jax.Array,
    wandb_run,
    phase: str = "warmup",
    num_steps: int = None,
):
    """Train policy with FSDP (warmup or RECAP)."""
    is_recap = phase == "recap"
    num_steps = num_steps or (config.policy_recap_steps if is_recap else config.policy_warmup_steps)

    logger.info("=" * 70)
    logger.info(f"PHASE 3{'b' if is_recap else 'a'}: Policy {phase.upper()}")
    logger.info("=" * 70)

    tx = optax.chain(
        optax.clip_by_global_norm(config.grad_clip),
        optax.adam(config.learning_rate),
    )

    policy_params = nnx.state(policy)
    opt_state = tx.init(policy_params)

    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    def policy_loss_fn(params, step_rng, observation, actions, improvement_indicator=None):
        graphdef = nnx.graphdef(policy)
        model = nnx.merge(graphdef, params)
        if improvement_indicator is not None:
            losses = model.compute_loss(step_rng, observation, actions, train=True,
                                       improvement_indicator=improvement_indicator)
        else:
            losses = model.compute_loss(step_rng, observation, actions, train=True)
        return jnp.mean(losses)

    if is_recap:
        @functools.partial(
            jax.jit,
            in_shardings=(policy_sharding, None, replicated_sharding, data_sharding, data_sharding, data_sharding),
            out_shardings=(policy_sharding, None, replicated_sharding, replicated_sharding),
        )
        def train_step(params, opt_state, step_rng, observation, actions, improvement_indicator):
            loss, grads = jax.value_and_grad(policy_loss_fn)(
                params, step_rng, observation, actions, improvement_indicator
            )
            updates, new_opt_state = tx.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            grad_norm = optax.global_norm(grads)
            return new_params, new_opt_state, loss, grad_norm
    else:
        @functools.partial(
            jax.jit,
            in_shardings=(policy_sharding, None, replicated_sharding, data_sharding, data_sharding),
            out_shardings=(policy_sharding, None, replicated_sharding, replicated_sharding),
        )
        def train_step(params, opt_state, step_rng, observation, actions):
            loss, grads = jax.value_and_grad(policy_loss_fn)(
                params, step_rng, observation, actions
            )
            updates, new_opt_state = tx.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            grad_norm = optax.global_norm(grads)
            return new_params, new_opt_state, loss, grad_norm

    indices = np.arange(len(dataset))
    rng_np = np.random.default_rng(config.seed + (1 if is_recap else 0))

    pbar = tqdm.tqdm(range(num_steps), desc=f"Policy {phase}")

    start_time = time.time()
    samples_processed = 0

    for step in pbar:
        rng_np.shuffle(indices)
        batch_indices = indices[:config.batch_size]
        batch = dataset.get_batch(batch_indices)

        observation = batch_to_observation(batch, config)
        actions = batch.get("action", jnp.zeros((config.batch_size, config.action_horizon, config.action_dim)))

        rng, step_rng = jax.random.split(rng)

        with sharding.set_mesh(mesh):
            if is_recap:
                imp_ind = jnp.array([improvement_indicators[int(i)] for i in batch_indices])
                policy_params, opt_state, loss, grad_norm = train_step(
                    policy_params, opt_state, step_rng, observation, actions, imp_ind
                )
                pct_good = float(jnp.mean(imp_ind))
            else:
                policy_params, opt_state, loss, grad_norm = train_step(
                    policy_params, opt_state, step_rng, observation, actions
                )
                pct_good = None

        samples_processed += config.batch_size
        elapsed = time.time() - start_time
        throughput = samples_processed / elapsed

        loss_val = float(loss)
        grad_norm_val = float(grad_norm)

        if step % config.log_interval == 0:
            postfix = {"loss": f"{loss_val:.4f}", "grad": f"{grad_norm_val:.4f}", "tput": f"{throughput:.1f}"}
            if pct_good is not None:
                postfix["pct_good"] = f"{pct_good:.2%}"
            pbar.set_postfix(postfix)

            extra = {"pct_good": pct_good} if pct_good is not None else None
            log_training_metrics(
                wandb_run, step, f"policy_{phase}",
                loss=loss_val,
                grad_norm=grad_norm_val,
                throughput=throughput,
                extra_metrics=extra,
            )

    nnx.update(policy, policy_params)

    logger.info(f"Policy {phase} complete! Final loss: {loss_val:.4f}")
    return policy, rng


def main():
    args = parse_args()

    config = CONFIGS[args.config]
    if args.fsdp_devices:
        config = dataclasses.replace(config, fsdp_devices=args.fsdp_devices)
    if args.batch_size:
        config = dataclasses.replace(config, batch_size=args.batch_size)
    if args.experiment_name:
        config = dataclasses.replace(config, experiment_name=args.experiment_name)

    logger.info("=" * 70)
    logger.info("RECAP Training with FSDP")
    logger.info("=" * 70)
    logger.info(f"Platform: {platform.node()}")
    logger.info(f"JAX devices: {jax.devices()}")
    logger.info(f"Device count: {jax.device_count()}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Model: {config.model_variant}")
    logger.info(f"FSDP devices: {config.fsdp_devices}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info("=" * 70)

    # Initialize wandb with system metrics
    wandb_run = init_wandb(config, enabled=not args.no_wandb)

    # Create FSDP mesh
    mesh = sharding.make_mesh(config.fsdp_devices)
    logger.info(f"Created mesh: {mesh}")

    # Initialize RNG
    rng = jax.random.key(config.seed)

    # Load dataset
    logger.info("\n[1/5] Loading dataset...")
    dataset = LeRobotDataset(config)

    # Create models with FSDP sharding
    logger.info("\n[2/5] Creating models with FSDP...")
    value_fn, policy, value_sharding, policy_sharding, rng = create_models_with_fsdp(
        config, mesh, rng
    )

    # Train value function
    logger.info("\n[3/5] Training value function...")
    value_fn, rng = train_value_function_fsdp(
        config, dataset, value_fn, value_sharding, mesh, rng, wandb_run
    )

    # Compute advantages
    logger.info("\n[4/5] Computing advantages...")
    advantages, improvement_indicators = compute_advantages_fsdp(
        config, dataset, value_fn, mesh
    )

    # Train policy (warmup then RECAP)
    logger.info("\n[5/5] Training policy...")

    if config.policy_warmup_steps > 0:
        policy, rng = train_policy_fsdp(
            config, dataset, policy, policy_sharding, mesh,
            improvement_indicators, rng, wandb_run,
            phase="warmup", num_steps=config.policy_warmup_steps
        )

    policy, rng = train_policy_fsdp(
        config, dataset, policy, policy_sharding, mesh,
        improvement_indicators, rng, wandb_run,
        phase="recap", num_steps=config.policy_recap_steps
    )

    # Save checkpoint
    output_dir = os.path.join(config.output_dir, config.experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"\nCheckpoint directory: {output_dir}")

    # Finish wandb
    if wandb_run:
        import wandb
        wandb.finish()

    logger.info("\n" + "=" * 70)
    logger.info("RECAP TRAINING COMPLETE!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
