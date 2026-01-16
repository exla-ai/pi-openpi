"""Distributional Value Function for RECAP.

The value function V^π(o_t, ℓ) predicts the distribution over time-to-completion
(number of steps remaining until task success) given an observation and task.

Key design decisions from pi0.6 paper:
- Uses 201 bins for discrete distribution (predicting 0-200+ steps remaining)
- Shares vision encoder with policy but has separate value head
- Trained with cross-entropy loss on (observation, time_remaining) pairs
"""

import dataclasses
import logging

import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")

# Number of bins for distributional value function
# Bin i represents "task will complete in i steps"
# Last bin (200) represents "200 or more steps remaining"
NUM_VALUE_BINS = 201


@dataclasses.dataclass(frozen=True)
class ValueFunctionConfig:
    """Configuration for the distributional value function."""

    # Number of bins for time-to-completion distribution
    num_bins: int = NUM_VALUE_BINS

    # Model architecture (should match policy for shared representations)
    paligemma_variant: str = "gemma_2b"

    # Hidden dimension for value head MLP
    value_hidden_dim: int = 1024

    # Dtype for computation (as string, e.g., "bfloat16")
    dtype: str = "bfloat16"

    # Action dimension (needed for model compatibility)
    action_dim: int = 14

    # Maximum token length
    max_token_len: int = 256

    def create(self, rng: at.KeyArrayLike) -> "ValueFunction":
        """Create and initialize the value function model."""
        return ValueFunction(self, nnx.Rngs(rng))


class ValueFunction(nnx.Module):
    """Distributional value function for RECAP.

    Predicts P(time_to_completion = k | observation, task) for k in {0, 1, ..., num_bins-1}.

    Architecture:
    - Shares SigLIP vision encoder with policy
    - Shares PaliGemma language model for task understanding
    - Adds MLP value head on top of pooled representations
    """

    def __init__(self, config: ValueFunctionConfig, rngs: nnx.Rngs):
        self.config = config
        self.num_bins = config.num_bins

        # Vision and language backbone (same as pi0)
        paligemma_config = _gemma.get_config(config.paligemma_variant)

        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config],
                embed_dtype=config.dtype,
                adarms=False,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False])

        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        # Initialize with dummy image
        dummy_images = {"base_0_rgb": jnp.zeros((1, 224, 224, 3))}
        img.lazy_init(dummy_images["base_0_rgb"], train=False, rngs=rngs)

        self.PaliGemma = nnx.Dict(llm=llm, img=img)

        # Value head MLP: pooled_features -> hidden -> num_bins logits
        self.value_proj1 = nnx.Linear(paligemma_config.width, config.value_hidden_dim, rngs=rngs)
        self.value_proj2 = nnx.Linear(config.value_hidden_dim, config.value_hidden_dim, rngs=rngs)
        self.value_head = nnx.Linear(config.value_hidden_dim, config.num_bins, rngs=rngs)

        self.deterministic = True

    @at.typecheck
    def embed_observation(
        self, obs: _model.Observation
    ) -> at.Float[at.Array, "b emb"]:
        """Encode observation into a single vector representation.

        Pools image and language tokens into a single embedding.
        """
        tokens = []

        # Embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            # Mean pool over spatial dimension
            pooled = jnp.mean(image_tokens, axis=1)  # [b, emb]
            tokens.append(pooled)

        # Embed language prompt
        if obs.tokenized_prompt is not None:
            lang_tokens = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            # Mean pool over sequence (masked)
            mask = obs.tokenized_prompt_mask.astype(jnp.float32)[..., None]
            pooled_lang = jnp.sum(lang_tokens * mask, axis=1) / (jnp.sum(mask, axis=1) + 1e-8)
            tokens.append(pooled_lang)

        # Concatenate and mean pool all representations
        all_tokens = jnp.stack(tokens, axis=1)  # [b, num_sources, emb]
        pooled = jnp.mean(all_tokens, axis=1)  # [b, emb]

        return pooled

    @at.typecheck
    def forward(
        self, observation: _model.Observation
    ) -> at.Float[at.Array, "b num_bins"]:
        """Compute value distribution logits.

        Args:
            observation: Current observation including images and task prompt

        Returns:
            Logits for time-to-completion distribution (not softmaxed)
        """
        # Get pooled observation representation
        obs_embedding = self.embed_observation(observation)

        # MLP value head
        x = self.value_proj1(obs_embedding)
        x = nnx.gelu(x)
        x = self.value_proj2(x)
        x = nnx.gelu(x)
        logits = self.value_head(x)

        return logits

    @at.typecheck
    def compute_loss(
        self,
        observation: _model.Observation,
        time_to_completion: at.Int[at.Array, " b"],
    ) -> at.Float[at.Array, ""]:
        """Compute cross-entropy loss for value function training.

        Args:
            observation: Current observations
            time_to_completion: Ground truth number of steps until episode end (capped at num_bins-1)

        Returns:
            Scalar cross-entropy loss
        """
        logits = self.forward(observation)

        # Clamp targets to valid range [0, num_bins-1]
        targets = jnp.clip(time_to_completion, 0, self.num_bins - 1)

        # Cross-entropy loss
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        loss = -jnp.mean(log_probs[jnp.arange(targets.shape[0]), targets])

        return loss

    @at.typecheck
    def predict_value(
        self, observation: _model.Observation
    ) -> tuple[at.Float[at.Array, " b"], at.Float[at.Array, "b num_bins"]]:
        """Predict expected time-to-completion and full distribution.

        Args:
            observation: Current observation

        Returns:
            expected_time: Expected time-to-completion (weighted mean of distribution)
            probs: Full probability distribution over bins
        """
        logits = self.forward(observation)
        probs = jax.nn.softmax(logits, axis=-1)

        # Compute expected value
        bin_values = jnp.arange(self.num_bins, dtype=jnp.float32)
        expected_time = jnp.sum(probs * bin_values[None, :], axis=-1)

        return expected_time, probs

    @at.typecheck
    def compute_advantage(
        self,
        observation: _model.Observation,
        actual_time_remaining: at.Int[at.Array, " b"],
    ) -> at.Float[at.Array, " b"]:
        """Compute advantage for advantage conditioning.

        Advantage = V(o_t) - (τ - t)

        Where:
        - V(o_t) is the predicted expected time-to-completion
        - (τ - t) is the actual time remaining in the episode

        Positive advantage means: the current trajectory is doing BETTER than average
        (will finish faster than the policy typically would from this state)

        Args:
            observation: Current observation
            actual_time_remaining: Actual number of steps remaining until episode end

        Returns:
            Advantage values (positive = better than average, negative = worse)
        """
        expected_time, _ = self.predict_value(observation)

        # Advantage = expected - actual
        # If expected > actual, we're doing better than average (positive advantage)
        advantage = expected_time - actual_time_remaining.astype(jnp.float32)

        return advantage


def compute_improvement_indicator(
    advantage: at.Float[at.Array, " b"],
    threshold: float = 0.0,
) -> at.Bool[at.Array, " b"]:
    """Compute binary improvement indicator I_t for advantage conditioning.

    I_t = 1 if advantage > threshold (trajectory is doing better than average)
    I_t = 0 otherwise (trajectory is doing worse than average)

    This indicator is used to condition the policy during training, allowing it
    to learn from both successful and unsuccessful trajectories.

    Args:
        advantage: Advantage values from value function
        threshold: Threshold for considering a trajectory "good" (default 0)

    Returns:
        Binary improvement indicators
    """
    return advantage > threshold
