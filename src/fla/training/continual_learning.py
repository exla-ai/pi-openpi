"""Continual learning methods to prevent catastrophic forgetting.

This module implements several continual learning techniques:
1. EWC (Elastic Weight Consolidation) - Penalizes changes to important weights
2. Experience Replay - Mixed training on old and new tasks (built into data loader)
3. L2 Regularization toward base weights

References:
- EWC: "Overcoming catastrophic forgetting in neural networks" (Kirkpatrick et al., 2017)
- Online EWC: "Progress & Compress" (Schwarz et al., 2018)
"""

import dataclasses
import logging
from typing import Callable

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import flax.traverse_util as traverse_util

import fla.shared.array_typing as at
import fla.models.model as _model


@dataclasses.dataclass
class EWCConfig:
    """Configuration for Elastic Weight Consolidation.

    Attributes:
        enabled: Whether to use EWC regularization
        lambda_ewc: Strength of EWC regularization (higher = less forgetting, but slower learning)
        fisher_samples: Number of samples to estimate Fisher information matrix
        online_ewc: Whether to use online EWC (accumulates Fisher across tasks)
        gamma: Decay factor for online EWC (0-1, lower = faster decay of old task importance)
    """
    enabled: bool = False
    lambda_ewc: float = 1000.0
    fisher_samples: int = 200
    online_ewc: bool = True
    gamma: float = 0.95


@dataclasses.dataclass
class L2RegConfig:
    """Configuration for L2 regularization toward base weights.

    This is a simpler alternative to EWC that penalizes deviation from
    the base model weights. Useful for fine-tuning.

    Attributes:
        enabled: Whether to use L2 regularization
        lambda_l2: Strength of L2 regularization
        base_params: Reference parameters to regularize toward (set during training)
    """
    enabled: bool = False
    lambda_l2: float = 0.01


@dataclasses.dataclass
class ContinualLearningConfig:
    """Main configuration for continual learning methods."""
    ewc: EWCConfig = dataclasses.field(default_factory=EWCConfig)
    l2_reg: L2RegConfig = dataclasses.field(default_factory=L2RegConfig)


@dataclasses.dataclass
class EWCState:
    """State for EWC regularization.

    Attributes:
        fisher: Diagonal Fisher information matrix (importance weights)
        optimal_params: Parameters at the end of previous task training
        task_count: Number of tasks trained so far
    """
    fisher: at.Params | None = None
    optimal_params: at.Params | None = None
    task_count: int = 0


def compute_fisher_information(
    model: _model.BaseModel,
    data_loader,
    trainable_filter: Callable,
    num_samples: int = 200,
    rng: at.KeyArrayLike = None,
) -> at.Params:
    """Compute diagonal Fisher information matrix.

    The Fisher information measures how sensitive the loss is to each parameter.
    Parameters with high Fisher values are important for the current task.

    Args:
        model: The trained model
        data_loader: Data loader for the current task
        trainable_filter: Filter for trainable parameters
        num_samples: Number of samples to estimate Fisher
        rng: Random key for sampling

    Returns:
        Diagonal Fisher information matrix as a parameter dict
    """
    if rng is None:
        rng = jax.random.key(0)

    model.eval()
    params = nnx.state(model)
    trainable_params = params.filter(trainable_filter)

    # Initialize Fisher as zeros with same structure as trainable params
    fisher = jax.tree.map(jnp.zeros_like, trainable_params.to_pure_dict())

    data_iter = iter(data_loader)

    for i in range(num_samples):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            batch = next(data_iter)

        observation, actions = batch
        sample_rng = jax.random.fold_in(rng, i)

        # Compute gradients of log-likelihood (loss) w.r.t. parameters
        def loss_fn(params_dict):
            # Temporarily update model with these params
            params.filter(trainable_filter).replace_by_pure_dict(params_dict)
            nnx.update(model, params)
            loss = model.compute_loss(sample_rng, observation, actions, train=False)
            return jnp.mean(loss)

        grads = jax.grad(loss_fn)(trainable_params.to_pure_dict())

        # Fisher is expectation of squared gradients
        fisher = jax.tree.map(lambda f, g: f + g ** 2, fisher, grads)

    # Average over samples
    fisher = jax.tree.map(lambda f: f / num_samples, fisher)

    logging.info(f"Computed Fisher information from {num_samples} samples")
    return fisher


def ewc_loss(
    current_params: at.Params,
    optimal_params: at.Params,
    fisher: at.Params,
    lambda_ewc: float,
) -> at.Array:
    """Compute EWC regularization loss.

    EWC loss = (lambda/2) * sum_i F_i * (theta_i - theta*_i)^2

    where F_i is Fisher information and theta*_i are optimal params for previous tasks.

    Args:
        current_params: Current model parameters (dict)
        optimal_params: Optimal parameters from previous task
        fisher: Fisher information matrix
        lambda_ewc: Regularization strength

    Returns:
        EWC regularization loss (scalar)
    """
    def param_ewc_loss(curr, opt, fish):
        return fish * (curr - opt) ** 2

    losses = jax.tree.map(param_ewc_loss, current_params, optimal_params, fisher)
    total_loss = sum(jnp.sum(l) for l in jax.tree.leaves(losses))

    return (lambda_ewc / 2) * total_loss


def l2_regularization_loss(
    current_params: at.Params,
    base_params: at.Params,
    lambda_l2: float,
) -> at.Array:
    """Compute L2 regularization loss toward base weights.

    L2 loss = (lambda/2) * sum_i (theta_i - theta_base_i)^2

    Args:
        current_params: Current model parameters
        base_params: Base/reference parameters
        lambda_l2: Regularization strength

    Returns:
        L2 regularization loss (scalar)
    """
    def param_l2_loss(curr, base):
        return (curr - base) ** 2

    losses = jax.tree.map(param_l2_loss, current_params, base_params)
    total_loss = sum(jnp.sum(l) for l in jax.tree.leaves(losses))

    return (lambda_l2 / 2) * total_loss


def update_ewc_state(
    ewc_state: EWCState,
    new_fisher: at.Params,
    current_params: at.Params,
    gamma: float = 0.95,
) -> EWCState:
    """Update EWC state after training on a new task (online EWC).

    For online EWC, we accumulate Fisher information across tasks:
    F_cumulative = gamma * F_old + F_new

    Args:
        ewc_state: Current EWC state
        new_fisher: Fisher information from current task
        current_params: Current model parameters
        gamma: Decay factor for old Fisher information

    Returns:
        Updated EWC state
    """
    if ewc_state.fisher is None:
        # First task - just store the Fisher and params
        return EWCState(
            fisher=new_fisher,
            optimal_params=current_params,
            task_count=1,
        )

    # Online EWC: accumulate Fisher information
    accumulated_fisher = jax.tree.map(
        lambda old, new: gamma * old + new,
        ewc_state.fisher,
        new_fisher,
    )

    return EWCState(
        fisher=accumulated_fisher,
        optimal_params=current_params,
        task_count=ewc_state.task_count + 1,
    )


def save_ewc_state(ewc_state: EWCState, path: str):
    """Save EWC state to disk for resuming continual learning."""
    import pickle
    with open(path, 'wb') as f:
        pickle.dump({
            'fisher': ewc_state.fisher,
            'optimal_params': ewc_state.optimal_params,
            'task_count': ewc_state.task_count,
        }, f)
    logging.info(f"Saved EWC state to {path}")


def load_ewc_state(path: str) -> EWCState:
    """Load EWC state from disk."""
    import pickle
    with open(path, 'rb') as f:
        data = pickle.load(f)
    ewc_state = EWCState(
        fisher=data['fisher'],
        optimal_params=data['optimal_params'],
        task_count=data['task_count'],
    )
    logging.info(f"Loaded EWC state from {path} (task_count={ewc_state.task_count})")
    return ewc_state


class ReplayBuffer:
    """Simple replay buffer for experience replay.

    Stores a subset of samples from each task to mix into training
    on new tasks. This is complementary to EWC.
    """

    def __init__(self, max_size: int = 10000, samples_per_task: int = 1000):
        self.max_size = max_size
        self.samples_per_task = samples_per_task
        self.buffer = []
        self.task_boundaries = []  # Track where each task's samples start

    def add_task_samples(self, data_loader, num_samples: int = None):
        """Add samples from a task to the replay buffer."""
        if num_samples is None:
            num_samples = self.samples_per_task

        samples = []
        data_iter = iter(data_loader)
        for _ in range(num_samples):
            try:
                batch = next(data_iter)
                # Store individual samples, not batches
                for i in range(len(batch[1])):
                    obs = jax.tree.map(lambda x: x[i:i+1], batch[0])
                    act = batch[1][i:i+1]
                    samples.append((obs, act))
                    if len(samples) >= num_samples:
                        break
            except StopIteration:
                break
            if len(samples) >= num_samples:
                break

        self.task_boundaries.append(len(self.buffer))
        self.buffer.extend(samples)

        # Trim if over max size (remove oldest samples)
        if len(self.buffer) > self.max_size:
            excess = len(self.buffer) - self.max_size
            self.buffer = self.buffer[excess:]
            self.task_boundaries = [max(0, b - excess) for b in self.task_boundaries]

        logging.info(f"Added {len(samples)} samples to replay buffer (total: {len(self.buffer)})")

    def sample(self, batch_size: int, rng: at.KeyArrayLike):
        """Sample a batch from the replay buffer."""
        if len(self.buffer) == 0:
            return None

        indices = jax.random.randint(rng, (batch_size,), 0, len(self.buffer))
        indices = [int(i) for i in indices]

        # Collate samples into a batch
        obs_list = [self.buffer[i][0] for i in indices]
        act_list = [self.buffer[i][1] for i in indices]

        # Stack observations and actions
        obs_batch = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *obs_list)
        act_batch = jnp.concatenate(act_list, axis=0)

        return (obs_batch, act_batch)
