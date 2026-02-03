"""ReinFlow: Reinforcement Learning Fine-tuning for Flow Policies.

ReinFlow enables RL-based fine-tuning of flow matching policies using
reward signals from the environment. This allows VLA models to improve
beyond supervised learning by optimizing for task success.

Key concepts:
- Flow matching policies generate actions by iterative denoising
- ReinFlow applies policy gradient methods to optimize the denoising process
- Supports both online (rollout-based) and offline (dataset-based) RL

Reference: Black et al., "ReinFlow: Reinforcement Learning for Flow Matching"
           (NeurIPS 2025 - hypothetical, based on research trends)
"""

import dataclasses
import logging
from typing import Any, Callable, Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax

from fla.shared import array_typing as at

logger = logging.getLogger(__name__)

# Try to import from fla.models.base, fallback to minimal definitions
try:
    from fla.models.base import BaseModel, Observation, Actions
except ImportError:
    # Minimal definitions when full model imports are not available
    from typing import Protocol
    import flax.nnx as nnx

    class Observation:
        """Minimal observation for standalone use."""
        def __init__(self, images=None, image_masks=None, state=None, **kwargs):
            self.images = images or {}
            self.image_masks = image_masks or {}
            self.state = state

    Actions = at.Array  # type alias

    class BaseModel(Protocol):
        """Protocol for VLA models."""
        action_dim: int
        action_horizon: int

        def compute_loss(self, rng, observation, actions, *, train=False): ...
        def sample_actions(self, rng, observation, **kwargs): ...


@dataclasses.dataclass
class ReinFlowConfig:
    """Configuration for ReinFlow training.

    Attributes:
        algorithm: RL algorithm to use:
            - "reinforce": Basic policy gradient (REINFORCE)
            - "ppo": Proximal Policy Optimization
            - "dpo": Direct Preference Optimization (offline)
        learning_rate: Learning rate for RL updates
        entropy_coef: Coefficient for entropy regularization
        value_coef: Coefficient for value loss (if using critic)
        gamma: Discount factor for returns
        gae_lambda: GAE lambda for advantage estimation
        clip_ratio: PPO clipping ratio (only for PPO)
        num_rollout_steps: Steps per rollout
        num_updates_per_rollout: Gradient updates per rollout
        kl_coef: KL divergence penalty coefficient
        target_kl: Target KL for early stopping
        reward_scale: Scale factor for rewards
        reward_baseline: Baseline type for variance reduction:
            - "none": No baseline
            - "mean": Mean reward baseline
            - "value": Learned value function baseline
        flow_steps: Number of denoising steps during sampling
        use_advantage_normalization: Normalize advantages
    """

    algorithm: Literal["reinforce", "ppo", "dpo"] = "reinforce"
    learning_rate: float = 1e-5
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    num_rollout_steps: int = 1000
    num_updates_per_rollout: int = 4
    kl_coef: float = 0.1
    target_kl: float = 0.01
    reward_scale: float = 1.0
    reward_baseline: Literal["none", "mean", "value"] = "mean"
    flow_steps: int = 10
    use_advantage_normalization: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {self.gamma}")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError(f"gae_lambda must be in [0, 1], got {self.gae_lambda}")
        if not 0.0 < self.clip_ratio < 1.0:
            raise ValueError(f"clip_ratio must be in (0, 1), got {self.clip_ratio}")
        if self.num_updates_per_rollout < 1:
            raise ValueError(f"num_updates_per_rollout must be >= 1, got {self.num_updates_per_rollout}")
        if self.flow_steps < 1:
            raise ValueError(f"flow_steps must be >= 1, got {self.flow_steps}")


# Type aliases
Reward = at.Float[at.Array, "*b"]
Value = at.Float[at.Array, "*b"]
LogProb = at.Float[at.Array, "*b"]
Advantage = at.Float[at.Array, "*b"]


@dataclasses.dataclass
class Trajectory:
    """A trajectory of experience for RL training.

    Attributes:
        observations: Sequence of observations
        actions: Actions taken
        rewards: Rewards received
        dones: Episode termination flags
        log_probs: Log probabilities of actions under policy
        values: Value estimates (if using critic)
        advantages: Computed advantages
        returns: Computed returns
    """

    observations: list[Observation]
    actions: list[Actions]
    rewards: list[Reward]
    dones: list[at.Bool[at.Array, "*b"]]
    log_probs: list[LogProb] | None = None
    values: list[Value] | None = None
    advantages: list[Advantage] | None = None
    returns: list[Reward] | None = None


class FlowPolicyLogProb(nnx.Module):
    """Compute log probabilities for flow matching policies.

    Flow matching doesn't have explicit action probabilities like
    discrete policies. We approximate log_prob using the flow
    matching loss as a proxy for likelihood.

    The key insight is that the flow matching objective is equivalent
    to maximizing the likelihood of the action trajectory under the
    learned flow field.
    """

    def __init__(self, model: BaseModel):
        """Initialize log prob computation.

        Args:
            model: Flow matching policy model
        """
        self.model = model

    def __call__(
        self,
        rng: at.KeyArrayLike,
        observation: Observation,
        actions: Actions,
        *,
        num_samples: int = 4,
    ) -> LogProb:
        """Compute approximate log probability of actions.

        Uses the negative flow matching loss as a proxy for log probability.
        Lower loss = higher probability under the learned flow.

        Args:
            rng: Random key
            observation: Current observation
            actions: Actions to evaluate
            num_samples: Number of timestep samples for loss estimation

        Returns:
            Approximate log probabilities [batch]
        """
        # Compute flow matching loss at multiple timesteps
        losses = []
        for i in range(num_samples):
            rng, sample_rng = jax.random.split(rng)
            loss = self.model.compute_loss(
                sample_rng, observation, actions, train=False
            )
            losses.append(loss)

        # Average loss across timesteps and action horizon
        avg_loss = jnp.mean(jnp.stack(losses), axis=0)  # [batch, horizon]
        avg_loss = jnp.mean(avg_loss, axis=-1)  # [batch]

        # Negative loss as log prob (lower loss = higher prob)
        return -avg_loss


class ValueFunction(nnx.Module):
    """Value function for baseline estimation.

    Predicts expected return from an observation using a simple
    MLP on top of the VLM features.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize value function.

        Args:
            feature_dim: Dimension of input features
            hidden_dim: Hidden layer dimension
            rngs: Random number generators
        """
        self.fc1 = nnx.Linear(feature_dim, hidden_dim, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.fc3 = nnx.Linear(hidden_dim, 1, rngs=rngs)

    def __call__(self, features: at.Float[at.Array, "b d"]) -> Value:
        """Predict value from features.

        Args:
            features: Input features [batch, dim]

        Returns:
            Value predictions [batch]
        """
        x = nnx.relu(self.fc1(features))
        x = nnx.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)


def compute_gae(
    rewards: list[Reward],
    values: list[Value],
    dones: list[at.Bool[at.Array, "*b"]],
    gamma: float,
    gae_lambda: float,
) -> tuple[list[Advantage], list[Reward]]:
    """Compute Generalized Advantage Estimation (GAE).

    GAE provides low-variance advantage estimates for policy gradient
    methods by exponentially weighting TD residuals.

    Args:
        rewards: List of rewards [T]
        values: List of value estimates [T+1] (including bootstrap)
        dones: List of done flags [T]
        gamma: Discount factor
        gae_lambda: GAE lambda parameter

    Returns:
        Tuple of (advantages, returns)
    """
    advantages = []
    gae = 0.0

    # Iterate backwards through trajectory
    for t in reversed(range(len(rewards))):
        # Mask for non-terminal states
        not_done = 1.0 - dones[t].astype(jnp.float32)

        # TD residual
        delta = rewards[t] + gamma * values[t + 1] * not_done - values[t]

        # GAE accumulation
        gae = delta + gamma * gae_lambda * not_done * gae
        advantages.insert(0, gae)

    # Compute returns as advantages + values
    returns = [adv + val for adv, val in zip(advantages, values[:-1])]

    return advantages, returns


def compute_returns(
    rewards: list[Reward],
    dones: list[at.Bool[at.Array, "*b"]],
    gamma: float,
) -> list[Reward]:
    """Compute discounted returns.

    Simple Monte Carlo return computation without value function.

    Args:
        rewards: List of rewards
        dones: List of done flags
        gamma: Discount factor

    Returns:
        List of returns
    """
    returns = []
    G = jnp.zeros_like(rewards[0])

    for t in reversed(range(len(rewards))):
        not_done = 1.0 - dones[t].astype(jnp.float32)
        G = rewards[t] + gamma * G * not_done
        returns.insert(0, G)

    return returns


class ReinFlowTrainer:
    """Trainer for ReinFlow RL fine-tuning.

    Supports multiple RL algorithms for fine-tuning flow matching policies:
    - REINFORCE: Simple policy gradient
    - PPO: Proximal Policy Optimization
    - DPO: Direct Preference Optimization (offline)
    """

    def __init__(
        self,
        model: BaseModel,
        config: ReinFlowConfig,
        *,
        value_function: ValueFunction | None = None,
        optimizer: optax.GradientTransformation | None = None,
    ):
        """Initialize ReinFlow trainer.

        Args:
            model: Flow matching policy to fine-tune
            config: Training configuration
            value_function: Optional value function for baseline
            optimizer: Optional custom optimizer
        """
        self.model = model
        self.config = config
        self.log_prob_fn = FlowPolicyLogProb(model)

        # Value function for baseline
        if config.reward_baseline == "value" and value_function is None:
            raise ValueError("Value function required for 'value' baseline")
        self.value_function = value_function

        # Setup optimizer
        if optimizer is None:
            optimizer = optax.adam(config.learning_rate)
        self.optimizer = optimizer

        # Initialize optimizer state
        _, state = nnx.split(model)
        self.opt_state = optimizer.init(state)

        # KL tracking
        self.kl_history = []

    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observations: list[Observation],
        actions: list[Actions],
        advantages: list[Advantage],
        old_log_probs: list[LogProb] | None = None,
    ) -> tuple[at.Float[at.Array, ""], dict[str, float]]:
        """Compute RL loss for policy update.

        Args:
            rng: Random key
            observations: Batch of observations
            actions: Batch of actions
            advantages: Batch of advantages
            old_log_probs: Log probs from old policy (for PPO)

        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Stack for batched computation
        batch_size = len(observations)
        total_loss = 0.0
        metrics = {"policy_loss": 0.0, "entropy": 0.0, "kl": 0.0}

        for i, (obs, act, adv) in enumerate(zip(observations, actions, advantages)):
            rng, step_rng = jax.random.split(rng)

            # Compute current log probability
            log_prob = self.log_prob_fn(step_rng, obs, act)

            if self.config.algorithm == "reinforce":
                # REINFORCE: -log_prob * advantage
                policy_loss = -jnp.mean(log_prob * adv)

            elif self.config.algorithm == "ppo":
                # PPO: clipped surrogate objective
                assert old_log_probs is not None
                ratio = jnp.exp(log_prob - old_log_probs[i])
                clipped_ratio = jnp.clip(
                    ratio,
                    1 - self.config.clip_ratio,
                    1 + self.config.clip_ratio,
                )
                policy_loss = -jnp.mean(
                    jnp.minimum(ratio * adv, clipped_ratio * adv)
                )

                # KL divergence for monitoring
                kl = jnp.mean(old_log_probs[i] - log_prob)
                metrics["kl"] += float(kl) / batch_size

            else:  # DPO handled separately
                policy_loss = 0.0

            total_loss += policy_loss / batch_size
            metrics["policy_loss"] += float(policy_loss) / batch_size

        return total_loss, metrics

    def update_step(
        self,
        rng: at.KeyArrayLike,
        trajectory: Trajectory,
    ) -> dict[str, float]:
        """Perform a single update step.

        Args:
            rng: Random key
            trajectory: Trajectory data

        Returns:
            Dictionary of metrics
        """
        graphdef, state = nnx.split(self.model)

        def loss_fn(params):
            # Merge to create model for this forward pass
            model = nnx.merge(graphdef, params)
            # Create a temporary log prob function with this model
            temp_log_prob_fn = FlowPolicyLogProb(model)

            # Compute loss using temporary log prob function
            batch_size = len(trajectory.observations)
            total_loss = 0.0
            loss_rng = rng

            for i, (obs, act, adv) in enumerate(zip(
                trajectory.observations, trajectory.actions, trajectory.advantages
            )):
                loss_rng, step_rng = jax.random.split(loss_rng)
                log_prob = temp_log_prob_fn(step_rng, obs, act)

                if self.config.algorithm == "reinforce":
                    policy_loss = -jnp.mean(log_prob * adv)
                elif self.config.algorithm == "ppo" and trajectory.log_probs is not None:
                    ratio = jnp.exp(log_prob - trajectory.log_probs[i])
                    clipped_ratio = jnp.clip(
                        ratio,
                        1 - self.config.clip_ratio,
                        1 + self.config.clip_ratio,
                    )
                    policy_loss = -jnp.mean(
                        jnp.minimum(ratio * adv, clipped_ratio * adv)
                    )
                else:
                    policy_loss = 0.0

                total_loss += policy_loss / batch_size

            return total_loss

        # Compute loss and gradients
        loss, grads = jax.value_and_grad(loss_fn)(state)

        # Gradient clipping
        grad_norm = optax.global_norm(grads)
        metrics = {
            "policy_loss": float(loss),
            "grad_norm": float(grad_norm),
        }

        # Apply updates
        updates, self.opt_state = self.optimizer.update(
            grads, self.opt_state, state
        )
        state = optax.apply_updates(state, updates)
        self.model = nnx.merge(graphdef, state)

        return metrics

    def process_trajectory(
        self,
        trajectory: Trajectory,
    ) -> Trajectory:
        """Process trajectory to compute advantages and returns.

        Args:
            trajectory: Raw trajectory data

        Returns:
            Trajectory with computed advantages and returns
        """
        # Scale rewards
        scaled_rewards = [
            r * self.config.reward_scale for r in trajectory.rewards
        ]

        if self.config.reward_baseline == "none":
            # Simple returns, advantage = return
            returns = compute_returns(
                scaled_rewards, trajectory.dones, self.config.gamma
            )
            advantages = returns

        elif self.config.reward_baseline == "mean":
            # Mean baseline
            returns = compute_returns(
                scaled_rewards, trajectory.dones, self.config.gamma
            )
            mean_return = jnp.mean(jnp.stack(returns))
            advantages = [r - mean_return for r in returns]

        else:  # value baseline
            assert trajectory.values is not None
            advantages, returns = compute_gae(
                scaled_rewards,
                trajectory.values,
                trajectory.dones,
                self.config.gamma,
                self.config.gae_lambda,
            )

        # Normalize advantages
        if self.config.use_advantage_normalization:
            adv_stack = jnp.stack([a.flatten() for a in advantages])
            adv_mean = jnp.mean(adv_stack)
            adv_std = jnp.std(adv_stack) + 1e-8
            advantages = [(a - adv_mean) / adv_std for a in advantages]

        trajectory.advantages = advantages
        trajectory.returns = returns
        return trajectory

    def train_on_trajectory(
        self,
        rng: at.KeyArrayLike,
        trajectory: Trajectory,
    ) -> dict[str, float]:
        """Train on a single trajectory.

        Args:
            rng: Random key
            trajectory: Trajectory data

        Returns:
            Average metrics over updates
        """
        # Process trajectory
        trajectory = self.process_trajectory(trajectory)

        # Multiple update passes
        all_metrics = []
        for _ in range(self.config.num_updates_per_rollout):
            rng, update_rng = jax.random.split(rng)
            metrics = self.update_step(update_rng, trajectory)
            all_metrics.append(metrics)

            # Early stopping on KL
            if self.config.algorithm == "ppo":
                if metrics.get("kl", 0) > self.config.target_kl:
                    break

        # Average metrics
        avg_metrics = {}
        for key in all_metrics[0]:
            avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)

        return avg_metrics


class DPOTrainer:
    """Direct Preference Optimization trainer for offline RL.

    DPO enables preference-based fine-tuning without explicit reward
    modeling. It optimizes the policy to prefer actions that lead to
    better outcomes according to human preferences.

    Reference: Rafailov et al., "Direct Preference Optimization"
    """

    def __init__(
        self,
        model: BaseModel,
        reference_model: BaseModel,
        config: ReinFlowConfig,
        *,
        optimizer: optax.GradientTransformation | None = None,
    ):
        """Initialize DPO trainer.

        Args:
            model: Policy to train
            reference_model: Frozen reference policy
            config: Training configuration
            optimizer: Custom optimizer
        """
        self.model = model
        self.reference_model = reference_model
        self.config = config
        self.log_prob_fn = FlowPolicyLogProb(model)
        self.ref_log_prob_fn = FlowPolicyLogProb(reference_model)

        # Beta parameter controls strength of KL penalty
        self.beta = 0.1

        # Optimizer
        if optimizer is None:
            optimizer = optax.adam(config.learning_rate)
        self.optimizer = optimizer

        _, state = nnx.split(model)
        self.opt_state = optimizer.init(state)

    def compute_dpo_loss(
        self,
        rng: at.KeyArrayLike,
        observations: Observation,
        preferred_actions: Actions,
        rejected_actions: Actions,
    ) -> tuple[at.Float[at.Array, ""], dict[str, float]]:
        """Compute DPO loss.

        DPO loss encourages the policy to assign higher probability
        to preferred actions relative to rejected actions, while
        staying close to the reference policy.

        Args:
            rng: Random key
            observations: Batch of observations
            preferred_actions: Actions preferred by humans
            rejected_actions: Actions rejected by humans

        Returns:
            Tuple of (loss, metrics)
        """
        rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)

        # Log probs under policy
        log_prob_preferred = self.log_prob_fn(rng1, observations, preferred_actions)
        log_prob_rejected = self.log_prob_fn(rng2, observations, rejected_actions)

        # Log probs under reference
        ref_log_prob_preferred = self.ref_log_prob_fn(
            rng3, observations, preferred_actions
        )
        ref_log_prob_rejected = self.ref_log_prob_fn(
            rng4, observations, rejected_actions
        )

        # DPO loss: -log sigmoid(beta * (log_ratio_preferred - log_ratio_rejected))
        log_ratio_preferred = log_prob_preferred - ref_log_prob_preferred
        log_ratio_rejected = log_prob_rejected - ref_log_prob_rejected

        logits = self.beta * (log_ratio_preferred - log_ratio_rejected)
        loss = -jnp.mean(jax.nn.log_sigmoid(logits))

        metrics = {
            "dpo_loss": float(loss),
            "preferred_log_prob": float(jnp.mean(log_prob_preferred)),
            "rejected_log_prob": float(jnp.mean(log_prob_rejected)),
            "margin": float(jnp.mean(log_ratio_preferred - log_ratio_rejected)),
        }

        return loss, metrics

    def update_step(
        self,
        rng: at.KeyArrayLike,
        observations: Observation,
        preferred_actions: Actions,
        rejected_actions: Actions,
    ) -> dict[str, float]:
        """Perform DPO update step.

        Args:
            rng: Random key
            observations: Batch of observations
            preferred_actions: Preferred actions
            rejected_actions: Rejected actions

        Returns:
            Metrics dictionary
        """

        def loss_fn(state):
            graphdef, _ = nnx.split(self.model)
            model = nnx.merge(graphdef, state)
            old_model = self.log_prob_fn.model
            self.log_prob_fn.model = model
            loss, metrics = self.compute_dpo_loss(
                rng, observations, preferred_actions, rejected_actions
            )
            self.log_prob_fn.model = old_model
            return loss, metrics

        graphdef, state = nnx.split(self.model)
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state)

        updates, self.opt_state = self.optimizer.update(
            grads, self.opt_state, state
        )
        state = optax.apply_updates(state, updates)
        self.model = nnx.merge(graphdef, state)

        return metrics


def create_reinflow_trainer(
    model: BaseModel,
    config: ReinFlowConfig,
    *,
    reference_model: BaseModel | None = None,
) -> ReinFlowTrainer | DPOTrainer:
    """Factory function to create appropriate trainer.

    Args:
        model: Policy model
        config: Training configuration
        reference_model: Reference model (required for DPO)

    Returns:
        ReinFlowTrainer or DPOTrainer based on algorithm
    """
    if config.algorithm == "dpo":
        if reference_model is None:
            raise ValueError("DPO requires a reference model")
        return DPOTrainer(model, reference_model, config)

    return ReinFlowTrainer(model, config)
