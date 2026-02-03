"""LoRA (Low-Rank Adaptation) for Parameter-Efficient Fine-tuning.

Provides a clean interface for LoRA fine-tuning of VLA models, wrapping
the openpi LoRA implementation with FLA-specific utilities.

LoRA reduces the number of trainable parameters by representing weight
updates as low-rank decompositions: W' = W + BA, where B and A are
low-rank matrices.

Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models"
"""

import dataclasses
import logging
from typing import Any, Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from fla.shared import array_typing as at
from fla.shared.nnx_utils import PathRegex

logger = logging.getLogger(__name__)

# Conditionally import openpi LoRA implementation
# Falls back to standalone implementation if openpi is not available
try:
    import sys
    import os
    _OPENPI_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "src"))
    if _OPENPI_PATH not in sys.path:
        sys.path.insert(0, _OPENPI_PATH)
    from openpi.models import lora as _lora
    from openpi.models import gemma as _gemma
    _HAS_OPENPI = True
except (ImportError, AttributeError):
    _lora = None
    _gemma = None
    _HAS_OPENPI = False


@dataclasses.dataclass(frozen=True)
class LoRAConfig:
    """Configuration for LoRA fine-tuning.

    Attributes:
        rank: LoRA rank (r). Higher = more capacity but more parameters.
            Common values: 4, 8, 16, 32, 64
        alpha: LoRA scaling factor. Typically set equal to rank.
        dropout: Dropout probability for LoRA layers (0.0 = no dropout)
        target_modules: Which modules to apply LoRA to:
            - "attention": Query, Key, Value projections
            - "ffn": Feed-forward network layers
            - "all": Both attention and FFN
        apply_to_vlm: If True, apply LoRA to VLM backbone
        apply_to_action_expert: If True, apply LoRA to action expert
        rslora: Use rank-stabilized LoRA (rsLoRA) for better training
        init_scale: Initialization scale for LoRA-A matrix

    Raises:
        ValueError: If configuration values are invalid.
    """

    rank: int = 16
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: Literal["attention", "ffn", "all"] = "all"
    apply_to_vlm: bool = False
    apply_to_action_expert: bool = True
    rslora: bool = True
    init_scale: float = 0.01

    def __post_init__(self):
        """Validate configuration."""
        if self.rank < 1:
            raise ValueError(f"rank must be >= 1, got {self.rank}")
        if self.alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {self.alpha}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.init_scale <= 0:
            raise ValueError(f"init_scale must be > 0, got {self.init_scale}")
        if not self.apply_to_vlm and not self.apply_to_action_expert:
            logger.warning("Neither apply_to_vlm nor apply_to_action_expert is True. No LoRA will be applied.")


@dataclasses.dataclass
class LoRALayerConfig:
    """Configuration for a single LoRA layer (standalone version)."""
    rank: int
    alpha: float
    init_scale: float
    rslora: bool

    @property
    def scaling_value(self) -> float:
        """Compute LoRA scaling factor."""
        if self.rslora:
            return self.alpha / (self.rank ** 0.5)
        return self.alpha / self.rank


def create_lora_config(config: LoRAConfig) -> dict[str, Any]:
    """Convert FLA LoRAConfig to layer configs.

    Args:
        config: FLA LoRA configuration

    Returns:
        Dictionary mapping module type to LoRALayerConfig
    """
    if _HAS_OPENPI and _lora is not None:
        import flax.linen as nn
        base_config = _lora.LoRAConfig(
            rank=config.rank,
            alpha=config.alpha,
            init_fn=nn.initializers.normal(stddev=config.init_scale),
            rslora=config.rslora,
        )
    else:
        # Standalone config - use our own dataclass
        base_config = LoRALayerConfig(
            rank=config.rank,
            alpha=config.alpha,
            init_scale=config.init_scale,
            rslora=config.rslora,
        )

    lora_configs = {}
    if config.target_modules in ("attention", "all"):
        lora_configs["attn"] = base_config
    if config.target_modules in ("ffn", "all"):
        lora_configs["ffn"] = base_config

    return lora_configs


def get_lora_gemma_variant(
    base_variant: str,
    config: LoRAConfig,
) -> tuple[str, str]:
    """Get Gemma variant names with LoRA suffix.

    Args:
        base_variant: Base Gemma variant (e.g., "gemma_2b")
        config: LoRA configuration

    Returns:
        Tuple of (vlm_variant, action_expert_variant)
    """
    vlm_variant = base_variant
    action_expert_variant = "gemma_300m"

    if config.apply_to_vlm:
        vlm_variant = f"{base_variant}_lora"
    if config.apply_to_action_expert:
        action_expert_variant = "gemma_300m_lora"

    return vlm_variant, action_expert_variant


def get_lora_params_filter(config: LoRAConfig) -> nnx.filterlib.Filter:
    """Get filter for LoRA parameters.

    Returns a filter that matches only LoRA parameters (lora_a, lora_b).
    Used to extract only LoRA params for saving/loading adapters.

    Args:
        config: LoRA configuration

    Returns:
        NNX filter matching LoRA parameters
    """
    return PathRegex(".*lora.*")


def get_frozen_params_filter(config: LoRAConfig) -> nnx.filterlib.Filter:
    """Get filter for parameters to freeze during LoRA training.

    Returns a filter that matches parameters that should be frozen
    (i.e., NOT receive gradient updates).

    Args:
        config: LoRA configuration

    Returns:
        NNX filter matching frozen parameters
    """
    # If no LoRA is applied anywhere, nothing should be frozen by this filter
    if not config.apply_to_vlm and not config.apply_to_action_expert:
        return nnx.Nothing

    filters = []

    # Freeze base weights (non-LoRA) in modules where LoRA is applied
    if config.apply_to_vlm:
        # Freeze VLM base weights, but not LoRA weights
        filters.append(nnx.All(
            PathRegex(".*llm.*"),
            nnx.Not(PathRegex(".*llm.*_1.*")),  # Exclude action expert
            nnx.Not(PathRegex(".*lora.*")),  # Don't freeze LoRA
        ))

    if config.apply_to_action_expert:
        # Freeze action expert base weights, but not LoRA weights
        filters.append(nnx.All(
            PathRegex(".*llm.*_1.*"),  # Action expert
            nnx.Not(PathRegex(".*lora.*")),  # Don't freeze LoRA
        ))

    if not filters:
        return nnx.Nothing

    if len(filters) == 1:
        return filters[0]

    return nnx.Any(*filters)


def count_lora_params(state: Any) -> tuple[int, int]:
    """Count LoRA and total parameters.

    Args:
        state: Model state (NNX state, pytree, or module)

    Returns:
        Tuple of (lora_params, total_params)
    """
    # Handle NNX modules by splitting first
    if isinstance(state, nnx.Module):
        _, state = nnx.split(state)

    def get_size(x):
        """Get size of array, handling NNX Param wrappers."""
        if hasattr(x, 'value'):
            return x.value.size
        elif hasattr(x, 'size'):
            return x.size
        return 0

    def count_params(tree):
        leaves = jax.tree_util.tree_leaves(tree)
        return sum(get_size(x) for x in leaves)

    total = count_params(state)

    # Count LoRA params by filtering paths
    lora_count = 0
    flat_state = jax.tree_util.tree_leaves_with_path(state)
    for path, leaf in flat_state:
        path_str = "/".join(str(k) for k in path)
        if "lora" in path_str.lower():
            lora_count += get_size(leaf)

    return lora_count, total


class LoRALinear(nnx.Module):
    """Linear layer with LoRA adaptation.

    Implements W' = W + BA where:
    - W: Original frozen weights [in_features, out_features]
    - B: Low-rank matrix [in_features, rank]
    - A: Low-rank matrix [rank, out_features]

    The scaling factor is alpha/rank (or alpha/sqrt(rank) for rsLoRA).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: LoRAConfig,
        *,
        rngs: nnx.Rngs,
        use_bias: bool = False,
    ):
        """Initialize LoRA linear layer.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            config: LoRA configuration
            rngs: Random number generators
            use_bias: Whether to include bias term
        """
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        self.use_bias = use_bias

        # Original weight (frozen during LoRA training)
        self.weight = nnx.Param(
            jax.random.normal(rngs.params(), (in_features, out_features))
            * (1.0 / jnp.sqrt(in_features))
        )

        if use_bias:
            self.bias = nnx.Param(jnp.zeros(out_features))
        else:
            self.bias = None

        # LoRA matrices
        # A is initialized with normal distribution
        self.lora_a = nnx.Param(
            jax.random.normal(rngs.params(), (in_features, config.rank))
            * config.init_scale
        )
        # B is initialized to zero so initial output is unchanged
        self.lora_b = nnx.Param(jnp.zeros((config.rank, out_features)))

        # Compute scaling factor (use Python float to avoid being counted as param)
        if config.rslora:
            self.scaling = config.alpha / (config.rank ** 0.5)
        else:
            self.scaling = config.alpha / config.rank

    def __call__(self, x: at.Float[at.Array, "... d"]) -> at.Float[at.Array, "... o"]:
        """Forward pass with LoRA adaptation.

        Args:
            x: Input tensor [..., in_features]

        Returns:
            Output tensor [..., out_features]
        """
        # Original linear transformation
        result = jnp.dot(x, self.weight.value)

        # LoRA adaptation: x @ A @ B * scaling
        lora_out = jnp.dot(jnp.dot(x, self.lora_a.value), self.lora_b.value)
        result = result + lora_out * self.scaling

        if self.bias is not None:
            result = result + self.bias.value

        return result

    def merge_lora(self) -> None:
        """Merge LoRA weights into base weights.

        After merging, the layer behaves as a standard linear layer
        with the adapted weights. This is useful for inference.
        """
        delta = jnp.dot(self.lora_a.value, self.lora_b.value) * self.scaling
        self.weight.value = self.weight.value + delta
        # Reset LoRA to zero
        self.lora_a.value = jnp.zeros_like(self.lora_a.value)
        self.lora_b.value = jnp.zeros_like(self.lora_b.value)


def save_lora_adapter(
    model: nnx.Module,
    path: str,
    config: LoRAConfig,
) -> None:
    """Save only LoRA adapter weights.

    Saves a lightweight checkpoint containing only the LoRA parameters,
    not the full model weights. This enables efficient storage of
    multiple task-specific adapters.

    Args:
        model: Model with LoRA layers
        path: Path to save adapter
        config: LoRA configuration (saved for loading)
    """
    import pickle
    import os

    # Extract LoRA parameters directly from the model
    lora_state = {}

    def extract_lora_params(obj, prefix=""):
        """Recursively extract LoRA parameters."""
        if isinstance(obj, nnx.Param):
            if "lora" in prefix.lower():
                # Get the actual array value
                lora_state[prefix] = jnp.array(obj.value)
        elif isinstance(obj, nnx.Module):
            for name, child in vars(obj).items():
                if not name.startswith('_'):
                    extract_lora_params(child, f"{prefix}/{name}" if prefix else name)
        elif isinstance(obj, dict):
            for name, child in obj.items():
                extract_lora_params(child, f"{prefix}/{name}" if prefix else str(name))

    extract_lora_params(model)

    checkpoint = {
        "lora_state": lora_state,
        "config": dataclasses.asdict(config),
    }

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(checkpoint, f)


def load_lora_adapter(
    model: nnx.Module,
    path: str,
) -> LoRAConfig:
    """Load LoRA adapter weights into model.

    Args:
        model: Model with LoRA layers (must have matching architecture)
        path: Path to adapter checkpoint

    Returns:
        LoRA configuration from checkpoint
    """
    import pickle

    with open(path, "rb") as f:
        checkpoint = pickle.load(f)

    config = LoRAConfig(**checkpoint["config"])
    lora_state = checkpoint["lora_state"]

    def load_lora_params(obj, prefix=""):
        """Recursively load LoRA parameters."""
        if isinstance(obj, nnx.Param):
            full_path = prefix
            if full_path in lora_state:
                obj.value = lora_state[full_path]
        elif isinstance(obj, nnx.Module):
            for name, child in vars(obj).items():
                if not name.startswith('_'):
                    load_lora_params(child, f"{prefix}/{name}" if prefix else name)
        elif isinstance(obj, dict):
            for name, child in obj.items():
                load_lora_params(child, f"{prefix}/{name}" if prefix else str(name))

    load_lora_params(model)

    return config


def apply_lora_to_model(
    model: nnx.Module,
    config: LoRAConfig,
) -> nnx.Module:
    """Apply LoRA adaptation to an existing model.

    This function wraps the model's linear layers with LoRA adapters
    based on the configuration.

    Note: For Pi0.5 models, it's more efficient to use the built-in
    LoRA support via gemma variant names (e.g., "gemma_2b_lora").

    Args:
        model: Model to adapt
        config: LoRA configuration

    Returns:
        Model with LoRA layers (same object, modified in place)
    """
    # For Pi05Model, LoRA is built into the Gemma architecture
    # This function is provided for custom models
    raise NotImplementedError(
        "For Pi0.5 models, use Pi05Config with paligemma_variant='gemma_2b_lora' "
        "or action_expert_variant='gemma_300m_lora' instead."
    )


def get_trainable_params(
    model: nnx.Module,
    config: LoRAConfig,
) -> dict[str, Any]:
    """Get only trainable parameters for LoRA training.

    Args:
        model: Model with LoRA layers
        config: LoRA configuration

    Returns:
        Dictionary of trainable parameters (LoRA params only)
    """
    graphdef, state = nnx.split(model)

    # Filter to only LoRA params
    trainable = {}
    flat_state = jax.tree_util.tree_leaves_with_path(state)
    for path, leaf in flat_state:
        path_str = "/".join(str(k) for k in path)
        if "lora" in path_str.lower():
            trainable[path_str] = leaf

    return trainable
