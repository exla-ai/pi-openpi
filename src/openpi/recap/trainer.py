"""RECAP Training Loop.

Implements the iterative training process from the pi0.6 paper:
1. Collect episodes with current policy
2. Train value function on (observation, time_to_completion) pairs
3. Compute advantages for all data
4. Train policy with advantage conditioning

The key insight is that this allows learning from BOTH successful and
unsuccessful trajectories by conditioning on whether each trajectory
is doing better (I_t=1) or worse (I_t=0) than average.
"""

import dataclasses
import functools
import logging
from typing import Iterator

import flax.nnx as nnx
from flax.training import common_utils
import jax
import jax.numpy as jnp
import optax

from openpi.models import model as _model
from openpi.recap.value_function import ValueFunction, ValueFunctionConfig, compute_improvement_indicator
from openpi.recap.pi0_recap import Pi0RECAP, Pi0RECAPConfig
from openpi.shared import array_typing as at
import openpi.training.sharding as sharding

logger = logging.getLogger("openpi")


@dataclasses.dataclass
class RECAPConfig:
    """Configuration for RECAP training."""

    # Policy configuration
    policy_config: Pi0RECAPConfig = dataclasses.field(default_factory=Pi0RECAPConfig)

    # Value function configuration
    value_config: ValueFunctionConfig = dataclasses.field(default_factory=ValueFunctionConfig)

    # Training parameters
    policy_lr: float = 1e-5
    value_lr: float = 1e-4
    batch_size: int = 32

    # Value function training
    value_train_steps: int = 1000
    value_grad_clip: float = 1.0

    # Policy training
    policy_train_steps: int = 1000
    policy_grad_clip: float = 1.0

    # RECAP iteration parameters
    num_iterations: int = 10
    episodes_per_iteration: int = 100

    # Advantage threshold for I_t computation
    advantage_threshold: float = 0.0

    # Random seed
    seed: int = 42


@dataclasses.dataclass
class Episode:
    """A single episode of interaction."""

    observations: list[dict]  # List of observation dicts
    actions: at.Float[at.Array, "T ah ad"]  # Actions taken
    success: bool  # Whether episode succeeded
    episode_length: int  # Total length of episode


@dataclasses.dataclass
class RECAPBatch:
    """A batch for RECAP training."""

    observations: _model.Observation
    actions: _model.Actions
    time_to_completion: at.Int[at.Array, " b"]  # Steps remaining until episode end
    improvement_indicator: at.Bool[at.Array, " b"]  # I_t indicator


class RECAPTrainer:
    """Trainer for RECAP algorithm.

    Implements the iterative training loop:
    1. Collect data with current policy
    2. Train value function
    3. Compute advantages
    4. Train policy with advantage conditioning
    """

    def __init__(
        self,
        config: RECAPConfig,
        mesh: jax.sharding.Mesh | None = None,
    ):
        self.config = config
        self.mesh = mesh or sharding.make_mesh(1)

        # Initialize RNG
        self.rng = jax.random.key(config.seed)

        # Initialize models
        self.rng, policy_rng, value_rng = jax.random.split(self.rng, 3)
        self.policy = config.policy_config.create(policy_rng)
        self.value_fn = config.value_config.create(value_rng)

        # Initialize optimizers
        self.policy_optimizer = optax.chain(
            optax.clip_by_global_norm(config.policy_grad_clip),
            optax.adam(config.policy_lr),
        )
        self.value_optimizer = optax.chain(
            optax.clip_by_global_norm(config.value_grad_clip),
            optax.adam(config.value_lr),
        )

        # Initialize optimizer states
        policy_params = nnx.state(self.policy, nnx.Param)
        value_params = nnx.state(self.value_fn, nnx.Param)
        self.policy_opt_state = self.policy_optimizer.init(policy_params)
        self.value_opt_state = self.value_optimizer.init(value_params)

        # Episode buffer for collected data
        self.episode_buffer: list[Episode] = []

    def add_episode(self, episode: Episode) -> None:
        """Add a collected episode to the buffer."""
        self.episode_buffer.append(episode)

    def clear_episodes(self) -> None:
        """Clear the episode buffer."""
        self.episode_buffer = []

    def create_value_batches(self, batch_size: int) -> Iterator[tuple[_model.Observation, at.Int[at.Array, " b"]]]:
        """Create batches for value function training from collected episodes.

        Yields (observation, time_to_completion) pairs for each timestep in collected episodes.
        """
        observations = []
        time_to_completions = []

        for episode in self.episode_buffer:
            for t, obs_dict in enumerate(episode.observations):
                # Time remaining = episode_length - current_step
                time_remaining = episode.episode_length - t
                observations.append(obs_dict)
                time_to_completions.append(time_remaining)

        # Shuffle and batch
        num_samples = len(observations)
        indices = jax.random.permutation(self.rng, num_samples)
        self.rng, _ = jax.random.split(self.rng)

        for i in range(0, num_samples - batch_size + 1, batch_size):
            batch_indices = indices[i : i + batch_size]
            batch_obs = [observations[int(j)] for j in batch_indices]
            batch_ttc = jnp.array([time_to_completions[int(j)] for j in batch_indices])

            # Convert observation list to Observation object
            obs = self._collate_observations(batch_obs)

            yield obs, batch_ttc

    def create_policy_batches(
        self, batch_size: int
    ) -> Iterator[RECAPBatch]:
        """Create batches for policy training with advantage conditioning.

        Computes advantages using the trained value function and creates
        improvement indicators for each sample.
        """
        all_data = []

        for episode in self.episode_buffer:
            for t in range(len(episode.observations) - 1):  # -1 because we need action at t
                obs_dict = episode.observations[t]
                action = episode.actions[t]
                time_remaining = episode.episode_length - t

                # Compute advantage using value function
                obs = self._collate_observations([obs_dict])
                advantage = self.value_fn.compute_advantage(
                    obs, jnp.array([time_remaining])
                )[0]

                # Compute improvement indicator
                improvement = float(advantage) > self.config.advantage_threshold

                all_data.append({
                    "obs": obs_dict,
                    "action": action,
                    "time_remaining": time_remaining,
                    "improvement": improvement,
                })

        # Shuffle and batch
        num_samples = len(all_data)
        indices = jax.random.permutation(self.rng, num_samples)
        self.rng, _ = jax.random.split(self.rng)

        for i in range(0, num_samples - batch_size + 1, batch_size):
            batch_indices = indices[i : i + batch_size]
            batch_data = [all_data[int(j)] for j in batch_indices]

            obs = self._collate_observations([d["obs"] for d in batch_data])
            actions = jnp.stack([d["action"] for d in batch_data])
            time_to_completion = jnp.array([d["time_remaining"] for d in batch_data])
            improvement_indicator = jnp.array([d["improvement"] for d in batch_data])

            yield RECAPBatch(
                observations=obs,
                actions=actions,
                time_to_completion=time_to_completion,
                improvement_indicator=improvement_indicator,
            )

    def _collate_observations(self, obs_list: list[dict]) -> _model.Observation:
        """Collate a list of observation dicts into a batched Observation."""
        # Stack each field
        images = {}
        image_masks = {}

        first_obs = obs_list[0]
        if "image" in first_obs:
            for key in first_obs["image"]:
                images[key] = jnp.stack([obs["image"][key] for obs in obs_list])
                if "image_mask" in first_obs and key in first_obs["image_mask"]:
                    image_masks[key] = jnp.stack([obs["image_mask"][key] for obs in obs_list])
                else:
                    image_masks[key] = jnp.ones(len(obs_list), dtype=jnp.bool_)

        state = jnp.stack([obs["state"] for obs in obs_list])

        tokenized_prompt = None
        tokenized_prompt_mask = None
        if "tokenized_prompt" in first_obs:
            tokenized_prompt = jnp.stack([obs["tokenized_prompt"] for obs in obs_list])
            tokenized_prompt_mask = jnp.stack([obs["tokenized_prompt_mask"] for obs in obs_list])

        return _model.Observation(
            images=images,
            image_masks=image_masks,
            state=state,
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask,
        )

    def train_value_function(self, num_steps: int | None = None) -> dict[str, float]:
        """Train the value function on collected episodes.

        Args:
            num_steps: Number of training steps (default from config)

        Returns:
            Training metrics
        """
        num_steps = num_steps or self.config.value_train_steps

        @jax.jit
        def value_train_step(params, opt_state, obs, time_to_completion):
            def loss_fn(p):
                # Temporarily update model with params
                nnx.update(self.value_fn, p)
                return self.value_fn.compute_loss(obs, time_to_completion)

            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, new_opt_state = self.value_optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss

        losses = []
        value_params = nnx.state(self.value_fn, nnx.Param)

        step = 0
        while step < num_steps:
            for obs, time_to_completion in self.create_value_batches(self.config.batch_size):
                if step >= num_steps:
                    break

                value_params, self.value_opt_state, loss = value_train_step(
                    value_params, self.value_opt_state, obs, time_to_completion
                )
                losses.append(float(loss))
                step += 1

                if step % 100 == 0:
                    logger.info(f"Value function step {step}/{num_steps}, loss: {loss:.4f}")

        # Update model with trained params
        nnx.update(self.value_fn, value_params)

        return {
            "value_loss": float(jnp.mean(jnp.array(losses))),
            "value_steps": step,
        }

    def train_policy(self, num_steps: int | None = None) -> dict[str, float]:
        """Train the policy with advantage conditioning.

        Args:
            num_steps: Number of training steps (default from config)

        Returns:
            Training metrics
        """
        num_steps = num_steps or self.config.policy_train_steps

        @jax.jit
        def policy_train_step(params, opt_state, rng, batch: RECAPBatch):
            def loss_fn(p):
                nnx.update(self.policy, p)
                losses = self.policy.compute_loss(
                    rng,
                    batch.observations,
                    batch.actions,
                    train=True,
                    improvement_indicator=batch.improvement_indicator,
                )
                return jnp.mean(losses)

            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, new_opt_state = self.policy_optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss

        losses = []
        policy_params = nnx.state(self.policy, nnx.Param)

        step = 0
        while step < num_steps:
            for batch in self.create_policy_batches(self.config.batch_size):
                if step >= num_steps:
                    break

                self.rng, step_rng = jax.random.split(self.rng)
                policy_params, self.policy_opt_state, loss = policy_train_step(
                    policy_params, self.policy_opt_state, step_rng, batch
                )
                losses.append(float(loss))
                step += 1

                if step % 100 == 0:
                    logger.info(f"Policy step {step}/{num_steps}, loss: {loss:.4f}")

        # Update model with trained params
        nnx.update(self.policy, policy_params)

        return {
            "policy_loss": float(jnp.mean(jnp.array(losses))),
            "policy_steps": step,
        }

    def run_iteration(self) -> dict[str, float]:
        """Run one RECAP iteration.

        1. Train value function on collected data
        2. Compute advantages and train policy

        Returns:
            Metrics from this iteration
        """
        if not self.episode_buffer:
            raise ValueError("No episodes in buffer. Collect data first.")

        logger.info(f"Running RECAP iteration with {len(self.episode_buffer)} episodes")

        # Train value function
        value_metrics = self.train_value_function()
        logger.info(f"Value function training complete: {value_metrics}")

        # Train policy with advantage conditioning
        policy_metrics = self.train_policy()
        logger.info(f"Policy training complete: {policy_metrics}")

        return {**value_metrics, **policy_metrics}


def create_recap_trainer(
    config: RECAPConfig,
    policy_checkpoint: str | None = None,
    mesh: jax.sharding.Mesh | None = None,
) -> RECAPTrainer:
    """Create a RECAP trainer, optionally loading from checkpoint.

    Args:
        config: RECAP configuration
        policy_checkpoint: Optional path to policy checkpoint
        mesh: Optional JAX mesh for distributed training

    Returns:
        Initialized RECAP trainer
    """
    trainer = RECAPTrainer(config, mesh)

    if policy_checkpoint:
        # Load policy weights
        from openpi.models import model as model_utils
        params = model_utils.restore_params(policy_checkpoint)

        # Load into policy
        graphdef, state = nnx.split(trainer.policy)
        state.replace_by_pure_dict(params)
        trainer.policy = nnx.merge(graphdef, state)

        logger.info(f"Loaded policy from {policy_checkpoint}")

    return trainer
