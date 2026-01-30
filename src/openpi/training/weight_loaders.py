import dataclasses
import glob
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)

# Optional imports for Gemma 3 weight loading
try:
    from huggingface_hub import hf_hub_download, snapshot_download
    from safetensors import safe_open
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# Gemma 3 HuggingFace model ID
GEMMA3_4B_HF_ID = "google/gemma-3-4b-pt"


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


@dataclasses.dataclass(frozen=True)
class Gemma3WeightLoader(WeightLoader):
    """Loads Gemma 3 weights from HuggingFace and PaliGemma's SigLIP vision encoder.

    This loader:
    1. Loads PaliGemma's SigLIP vision encoder weights (required for image processing)
    2. Downloads Gemma 3 4B weights from HuggingFace
    3. Maps Gemma 3 weights to the OpenPI LLM format
    4. Leaves action expert weights randomly initialized (trained from scratch)

    The Gemma 3 4B model has:
    - 26 layers, 2304 width, 9216 mlp_dim
    - 8 attention heads, 4 kv heads (GQA), 256 head_dim
    """

    hf_model_id: str = GEMMA3_4B_HF_ID

    def load(self, params: at.Params) -> at.Params:
        # Step 1: Load PaliGemma's SigLIP vision encoder
        logger.info("Loading PaliGemma SigLIP vision encoder...")
        paligemma_path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with paligemma_path.open("rb") as f:
            paligemma_params = dict(np.load(f, allow_pickle=False))

        # Extract only the vision encoder (img) weights from PaliGemma
        paligemma_flat = flax.traverse_util.unflatten_dict(paligemma_params, sep="/")["params"]
        img_params = {"PaliGemma": {"img": paligemma_flat.get("img", {})}}

        # Step 2: Load Gemma 3 weights from HuggingFace
        logger.info(f"Loading Gemma 3 weights from HuggingFace: {self.hf_model_id}")
        gemma3_params = self._load_gemma3_from_hf()

        # Step 3: Merge vision encoder + Gemma 3 LLM weights
        merged_params = _deep_merge(img_params, gemma3_params)

        # Step 4: Merge with reference params, keeping action expert and other missing weights
        return _merge_params(merged_params, params, missing_regex=".*")

    def _load_gemma3_from_hf(self) -> dict:
        """Load and convert Gemma 3 weights from HuggingFace format to OpenPI format."""
        if not HF_AVAILABLE:
            raise ImportError(
                "Please install huggingface_hub and safetensors: "
                "pip install huggingface_hub safetensors"
            )

        # Download the safetensors files
        logger.info(f"Downloading Gemma 3 model files from {self.hf_model_id}...")
        model_files = []

        # Try single file first
        try:
            model_path = hf_hub_download(
                repo_id=self.hf_model_id,
                filename="model.safetensors",
            )
            model_files = [model_path]
            logger.info("Found single model.safetensors file")
        except Exception as e:
            logger.info(f"No single model file found ({e}), trying sharded files...")

            # Use snapshot_download to get all safetensors files (handles sharding automatically)
            try:
                cache_dir = snapshot_download(
                    repo_id=self.hf_model_id,
                    allow_patterns="*.safetensors"
                )
                model_files = sorted(glob.glob(f"{cache_dir}/*.safetensors"))
                logger.info(f"Found {len(model_files)} safetensors files")
            except Exception as download_error:
                raise RuntimeError(
                    f"Failed to download model weights from {self.hf_model_id}: {download_error}"
                ) from download_error

        if not model_files:
            raise RuntimeError(f"No safetensors files found for {self.hf_model_id}")

        # Load all weights
        hf_weights = {}
        for model_path in model_files:
            logger.info(f"Loading weights from {model_path}")
            with safe_open(model_path, framework="numpy") as f:
                for key in f.keys():
                    hf_weights[key] = f.get_tensor(key)

        logger.info(f"Loaded {len(hf_weights)} weight tensors")

        # Convert HuggingFace format to OpenPI format
        return self._convert_hf_to_openpi(hf_weights)

    def _convert_hf_to_openpi(self, hf_weights: dict) -> dict:
        """Convert HuggingFace Gemma 3 weights to OpenPI format.

        HuggingFace naming: model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
        OpenPI naming: PaliGemma/llm/layers/layer_{i}/attn/{q,kv,attn_vec}_einsum
        """
        openpi_weights = {}

        # Embedding table
        if "model.embed_tokens.weight" in hf_weights:
            emb = hf_weights["model.embed_tokens.weight"]
            openpi_weights["PaliGemma/llm/embedder/input_embedding"] = emb

        # Process each layer
        num_layers = 26  # Gemma 3 4B has 26 layers
        for i in range(num_layers):
            hf_prefix = f"model.layers.{i}"
            # OpenPI uses scan, so layers are stacked along axis 0
            layer_idx = i

            # Attention weights
            # HF: q_proj (num_heads * head_dim, hidden), k_proj, v_proj, o_proj
            # OpenPI with GQA: q_einsum (num_heads, hidden, head_dim), kv_einsum (2, num_kv_heads, hidden, head_dim)
            if f"{hf_prefix}.self_attn.q_proj.weight" in hf_weights:
                q_weight = hf_weights[f"{hf_prefix}.self_attn.q_proj.weight"]  # (num_heads*head_dim, hidden)
                k_weight = hf_weights[f"{hf_prefix}.self_attn.k_proj.weight"]  # (num_kv_heads*head_dim, hidden)
                v_weight = hf_weights[f"{hf_prefix}.self_attn.v_proj.weight"]  # (num_kv_heads*head_dim, hidden)
                o_weight = hf_weights[f"{hf_prefix}.self_attn.o_proj.weight"]  # (hidden, num_heads*head_dim)

                # Gemma 3 4B: 8 heads, 4 kv_heads, 256 head_dim, 2304 hidden
                num_heads, num_kv_heads, head_dim = 8, 4, 256
                hidden = q_weight.shape[1]

                # Reshape Q: (num_heads*head_dim, hidden) -> (num_heads, hidden, head_dim)
                q_reshaped = q_weight.reshape(num_heads, head_dim, hidden).transpose(0, 2, 1)

                # Reshape K, V and stack: -> (2, num_kv_heads, hidden, head_dim)
                k_reshaped = k_weight.reshape(num_kv_heads, head_dim, hidden).transpose(0, 2, 1)
                v_reshaped = v_weight.reshape(num_kv_heads, head_dim, hidden).transpose(0, 2, 1)
                kv_stacked = np.stack([k_reshaped, v_reshaped], axis=0)

                # Reshape O: (hidden, num_heads*head_dim) -> (num_heads, head_dim, hidden)
                o_reshaped = o_weight.T.reshape(num_heads, head_dim, hidden)

                openpi_weights[f"PaliGemma/llm/layers/layer/attn/q_einsum/w/{layer_idx}"] = q_reshaped
                openpi_weights[f"PaliGemma/llm/layers/layer/attn/kv_einsum/w/{layer_idx}"] = kv_stacked
                openpi_weights[f"PaliGemma/llm/layers/layer/attn/attn_vec_einsum/w/{layer_idx}"] = o_reshaped

            # MLP weights
            # HF: gate_proj, up_proj (mlp_dim, hidden), down_proj (hidden, mlp_dim)
            # OpenPI: gating_einsum (2, hidden, mlp_dim), linear (mlp_dim, hidden)
            if f"{hf_prefix}.mlp.gate_proj.weight" in hf_weights:
                gate = hf_weights[f"{hf_prefix}.mlp.gate_proj.weight"]  # (mlp_dim, hidden)
                up = hf_weights[f"{hf_prefix}.mlp.up_proj.weight"]  # (mlp_dim, hidden)
                down = hf_weights[f"{hf_prefix}.mlp.down_proj.weight"]  # (hidden, mlp_dim)

                # Stack gate and up: (2, hidden, mlp_dim)
                gating = np.stack([gate.T, up.T], axis=0)
                linear = down.T  # (mlp_dim, hidden)

                openpi_weights[f"PaliGemma/llm/layers/layer/mlp/gating_einsum/{layer_idx}"] = gating
                openpi_weights[f"PaliGemma/llm/layers/layer/mlp/linear/{layer_idx}"] = linear

            # Layer norms (RMSNorm)
            # HF: input_layernorm.weight, post_attention_layernorm.weight
            # OpenPI: pre_attention_norm/scale, pre_ffw_norm/scale
            if f"{hf_prefix}.input_layernorm.weight" in hf_weights:
                pre_attn_scale = hf_weights[f"{hf_prefix}.input_layernorm.weight"]
                openpi_weights[f"PaliGemma/llm/layers/layer/pre_attention_norm/scale/{layer_idx}"] = pre_attn_scale

            if f"{hf_prefix}.post_attention_layernorm.weight" in hf_weights:
                pre_ffw_scale = hf_weights[f"{hf_prefix}.post_attention_layernorm.weight"]
                openpi_weights[f"PaliGemma/llm/layers/layer/pre_ffw_norm/scale/{layer_idx}"] = pre_ffw_scale

        # Final layer norm
        if "model.norm.weight" in hf_weights:
            openpi_weights["PaliGemma/llm/final_norm/scale"] = hf_weights["model.norm.weight"]

        # Convert flat dict to nested dict
        return flax.traverse_util.unflatten_dict(
            {k: v for k, v in openpi_weights.items()},
            sep="/"
        )


def _deep_merge(dict1: dict, dict2: dict) -> dict:
    """Deep merge two nested dictionaries."""
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")
