"""Pi0 model with RECAP advantage conditioning.

This extends the base pi0 model to support advantage-conditioned policy learning.
The key modification is adding the improvement indicator I_t as an additional
input to the action generation process.

From the pi0.6 paper:
- I_t = 1 indicates the trajectory is doing better than average (should imitate)
- I_t = 0 indicates the trajectory is doing worse than average (should avoid)

During inference, we set I_t = 1 to generate actions that lead to successful outcomes.
"""

import dataclasses
import logging
from typing_extensions import override

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models import pi0
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


@dataclasses.dataclass(frozen=True)
class Pi0RECAPConfig(pi0_config.Pi0Config):
    """Configuration for Pi0 with RECAP advantage conditioning."""

    # Dimension of advantage conditioning embedding
    advantage_embedding_dim: int = 64

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI05  # Use PI05 for adaRMS support

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0RECAP":
        return Pi0RECAP(self, nnx.Rngs(rng))


class Pi0RECAP(pi0.Pi0):
    """Pi0 model extended with RECAP advantage conditioning.

    The key modification is adding the improvement indicator I_t as an input
    to condition the action generation. This allows the model to:
    - Learn to imitate good trajectories (I_t = 1)
    - Learn to avoid behaviors from bad trajectories (I_t = 0)
    """

    def __init__(self, config: Pi0RECAPConfig, rngs: nnx.Rngs):
        # Initialize base Pi0 model
        super().__init__(config, rngs)

        # Add advantage conditioning layers
        # The indicator I_t is embedded and added to the action tokens
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        self.advantage_embedding = nnx.Embed(
            num_embeddings=2,  # Binary: 0 or 1
            features=config.advantage_embedding_dim,
            rngs=rngs,
        )
        self.advantage_proj = nnx.Linear(
            config.advantage_embedding_dim,
            action_expert_config.width,
            rngs=rngs,
        )

    @at.typecheck
    def embed_suffix_with_advantage(
        self,
        obs: _model.Observation,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, " b"],
        improvement_indicator: at.Bool[at.Array, " b"],
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        """Embed suffix tokens with advantage conditioning.

        Extends the base embed_suffix to include the improvement indicator I_t.
        The indicator is embedded and added to the action token representations.
        """
        input_mask = []
        ar_mask = []
        tokens = []

        if not self.pi05:
            # Add state token (same as base)
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            ar_mask += [True]

        # Embed noisy actions
        action_tokens = self.action_in_proj(noisy_actions)

        # Compute time embedding
        time_emb = pi0.posemb_sincos(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0
        )

        # Embed improvement indicator and add to action tokens
        # I_t is broadcast across all action tokens in the sequence
        advantage_emb = self.advantage_embedding(improvement_indicator.astype(jnp.int32))
        advantage_emb = self.advantage_proj(advantage_emb)
        advantage_emb = einops.repeat(advantage_emb, "b emb -> b s emb", s=self.action_horizon)

        if self.pi05:
            # For pi0.5: use adaRMS conditioning
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)

            # Add advantage embedding to action tokens
            action_expert_tokens = action_tokens + advantage_emb
            adarms_cond = time_emb
        else:
            # For pi0: mix through MLP
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)

            # Add advantage embedding
            action_expert_tokens = action_time_tokens + advantage_emb
            adarms_cond = None

        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        ar_mask += [True] + ([False] * (self.action_horizon - 1))

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)

        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
        improvement_indicator: at.Bool[at.Array, " b"] | None = None,
    ) -> at.Float[at.Array, "*b ah"]:
        """Compute flow matching loss with optional advantage conditioning.

        If improvement_indicator is provided, uses advantage-conditioned loss.
        Otherwise, falls back to standard pi0 loss (for compatibility).

        Args:
            rng: Random key for noise sampling
            observation: Current observations
            actions: Target actions to match
            train: Whether in training mode (enables augmentation)
            improvement_indicator: Binary indicator I_t (1 = good trajectory, 0 = bad)

        Returns:
            Per-action loss values
        """
        if improvement_indicator is None:
            # Fall back to base pi0 loss for compatibility
            return super().compute_loss(rng, observation, actions, train=train)

        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # Forward pass with advantage conditioning
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix_with_advantage(
            observation, x_t, time, improvement_indicator
        )

        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = pi0.make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        improvement_indicator: at.Bool[at.Array, " b"] | None = None,
    ) -> _model.Actions:
        """Sample actions with advantage conditioning.

        For inference, we typically set improvement_indicator=True for all samples
        to generate actions that lead to successful outcomes.

        Args:
            rng: Random key for noise sampling
            observation: Current observation
            num_steps: Number of denoising steps
            noise: Optional initial noise (for reproducibility)
            improvement_indicator: Improvement indicator (default: all True for inference)

        Returns:
            Sampled action sequence
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        batch_size = observation.state.shape[0]

        # Default to I_t = 1 for inference (we want good trajectories)
        if improvement_indicator is None:
            improvement_indicator = jnp.ones(batch_size, dtype=jnp.bool_)

        dt = -1.0 / num_steps
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # Cache prefix tokens
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix_with_advantage(
                observation, x_t, jnp.broadcast_to(time, batch_size), improvement_indicator
            )

            suffix_attn_mask = pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)

            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
