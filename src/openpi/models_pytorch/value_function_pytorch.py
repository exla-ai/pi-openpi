# Copyright 2024 Physical Intelligence.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyTorch Value Function for RECAP (pi0.6).

The value function V^π(o_t, ℓ) predicts the distribution over time-to-completion
(number of steps remaining until task success) given an observation and task.

Key design decisions from pi0.6 paper:
- Uses 201 bins for discrete distribution (predicting 0-200+ steps remaining)
- Uses a separate 670M VLM backbone (smaller than the policy)
- Trained with cross-entropy loss on (observation, time_remaining) pairs
"""

import dataclasses
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SiglipVisionModel
from transformers.models.auto import CONFIG_MAPPING

import openpi.models.gemma as _gemma

# Number of bins for distributional value function
NUM_VALUE_BINS = 201


@dataclasses.dataclass
class ValueFunctionPytorchConfig:
    """Configuration for the PyTorch value function."""

    # VLM backbone variant (default: 670M for pi0.6)
    vlm_variant: str = "gemma_670m"

    # Number of bins for time-to-completion distribution
    num_bins: int = NUM_VALUE_BINS

    # Hidden dimension for value head MLP
    value_hidden_dim: int = 1024

    # Precision for computation
    dtype: Literal["bfloat16", "float32"] = "bfloat16"


class ValueFunctionPytorch(nn.Module):
    """Distributional value function for RECAP (PyTorch).

    Predicts P(time_to_completion = k | observation, task) for k in {0, 1, ..., num_bins-1}.

    Architecture:
    - SigLIP vision encoder for image embedding
    - Gemma-based language model for text/task understanding
    - MLP value head on top of pooled representations
    """

    def __init__(self, config: ValueFunctionPytorchConfig):
        super().__init__()
        self.config = config
        self.num_bins = config.num_bins

        # Get VLM config
        vlm_config = _gemma.get_config(config.vlm_variant)

        # Vision encoder (SigLIP)
        self.vision_encoder = SiglipVisionModel.from_pretrained(
            "google/siglip-so400m-patch14-384",
            torch_dtype=torch.bfloat16 if config.dtype == "bfloat16" else torch.float32,
        )

        # Vision projection to match VLM width
        self.vision_proj = nn.Linear(1152, vlm_config.width)  # SigLIP hidden size -> VLM width

        # Language model backbone
        gemma_config_hf = CONFIG_MAPPING["gemma"](
            hidden_size=vlm_config.width,
            intermediate_size=vlm_config.mlp_dim,
            num_hidden_layers=vlm_config.depth,
            num_attention_heads=vlm_config.num_heads,
            num_key_value_heads=vlm_config.num_kv_heads,
            head_dim=vlm_config.head_dim,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
        )

        from transformers import GemmaModel

        self.language_model = GemmaModel(config=gemma_config_hf)

        # Value head MLP: pooled_features -> hidden -> num_bins logits
        self.value_head = nn.Sequential(
            nn.Linear(vlm_config.width, config.value_hidden_dim),
            nn.GELU(),
            nn.Linear(config.value_hidden_dim, config.value_hidden_dim),
            nn.GELU(),
            nn.Linear(config.value_hidden_dim, config.num_bins),
        )

        # Set dtype
        if config.dtype == "bfloat16":
            self.to(torch.bfloat16)

    def embed_images(self, images: torch.Tensor) -> torch.Tensor:
        """Embed images using SigLIP vision encoder.

        Args:
            images: [B, C, H, W] or [B, N, C, H, W] for multiple images

        Returns:
            Pooled image embeddings [B, D]
        """
        # SigLIP expects 384x384 images
        target_size = (384, 384)

        if images.ndim == 5:
            # Multiple images per sample: [B, N, C, H, W]
            b, n, c, h, w = images.shape
            images = images.view(b * n, c, h, w)
            # Resize to SigLIP expected size
            if h != target_size[0] or w != target_size[1]:
                images = F.interpolate(images, size=target_size, mode='bilinear', align_corners=False)
            vision_outputs = self.vision_encoder(images)
            # Mean pool over patches
            pooled = vision_outputs.last_hidden_state.mean(dim=1)  # [B*N, D]
            pooled = pooled.view(b, n, -1).mean(dim=1)  # [B, D]
        else:
            # Single image: [B, C, H, W]
            # Resize to SigLIP expected size
            if images.shape[-2] != target_size[0] or images.shape[-1] != target_size[1]:
                images = F.interpolate(images, size=target_size, mode='bilinear', align_corners=False)
            vision_outputs = self.vision_encoder(images)
            pooled = vision_outputs.last_hidden_state.mean(dim=1)  # [B, D]

        # Project to VLM width
        return self.vision_proj(pooled)

    def forward(self, image_embedding: torch.Tensor) -> torch.Tensor:
        """Compute value distribution logits.

        Args:
            image_embedding: Pooled image embedding [B, D]

        Returns:
            Logits for time-to-completion distribution [B, num_bins]
        """
        return self.value_head(image_embedding)

    def compute_loss(
        self,
        image_embedding: torch.Tensor,
        time_to_completion: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cross-entropy loss for value function training.

        Args:
            image_embedding: Pooled image embedding [B, D]
            time_to_completion: Ground truth steps until episode end [B]

        Returns:
            Scalar cross-entropy loss
        """
        logits = self.forward(image_embedding)

        # Clamp targets to valid range [0, num_bins-1]
        targets = time_to_completion.clamp(0, self.num_bins - 1).long()

        return F.cross_entropy(logits, targets)

    def predict_value(
        self, image_embedding: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict expected time-to-completion and full distribution.

        Args:
            image_embedding: Pooled image embedding [B, D]

        Returns:
            expected_time: Expected time-to-completion [B]
            probs: Full probability distribution [B, num_bins]
        """
        logits = self.forward(image_embedding)
        probs = F.softmax(logits, dim=-1)

        # Compute expected value
        bin_values = torch.arange(self.num_bins, dtype=probs.dtype, device=probs.device)
        expected_time = (probs * bin_values.unsqueeze(0)).sum(dim=-1)

        return expected_time, probs

    def compute_advantage(
        self,
        image_embedding: torch.Tensor,
        actual_time_remaining: torch.Tensor,
    ) -> torch.Tensor:
        """Compute advantage for advantage conditioning.

        Advantage = V(o_t) - (τ - t)

        Where:
        - V(o_t) is the predicted expected time-to-completion
        - (τ - t) is the actual time remaining in the episode

        Positive advantage means: the current trajectory is doing BETTER than average
        (will finish faster than the policy typically would from this state)

        Args:
            image_embedding: Pooled image embedding [B, D]
            actual_time_remaining: Actual steps remaining until episode end [B]

        Returns:
            Advantage values [B] (positive = better than average)
        """
        expected_time, _ = self.predict_value(image_embedding)
        return expected_time - actual_time_remaining.float()


def compute_improvement_indicator(
    advantage: torch.Tensor,
    threshold: float = 0.0,
) -> torch.Tensor:
    """Compute binary improvement indicator I_t for advantage conditioning.

    I_t = 1 if advantage > threshold (trajectory is doing better than average)
    I_t = 0 otherwise (trajectory is doing worse than average)

    Args:
        advantage: Advantage values [B]
        threshold: Threshold for considering a trajectory "good" (default 0)

    Returns:
        Binary improvement indicators [B]
    """
    return advantage > threshold
