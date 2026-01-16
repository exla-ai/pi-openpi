"""RECAP: RL with Experience and Corrections via Advantage-conditioned Policies.

This module implements the RECAP algorithm from the pi0.6 paper for training
Vision-Language-Action models with reinforcement learning.

Key components:
- ValueFunction: Distributional value function predicting time-to-completion
- Pi0RECAP: Extended pi0 model with advantage conditioning
- RECAPTrainer: Training loop for iterative policy improvement
"""

from openpi.recap.value_function import ValueFunction, ValueFunctionConfig
from openpi.recap.pi0_recap import Pi0RECAP, Pi0RECAPConfig
from openpi.recap.trainer import RECAPTrainer, RECAPConfig
from openpi.recap.isaaclab_data import (
    IsaacLabDataset,
    RECAPDataLoader,
    RECAPEpisode,
    create_recap_dataset,
)

__all__ = [
    "ValueFunction",
    "ValueFunctionConfig",
    "Pi0RECAP",
    "Pi0RECAPConfig",
    "RECAPTrainer",
    "RECAPConfig",
    "IsaacLabDataset",
    "RECAPDataLoader",
    "RECAPEpisode",
    "create_recap_dataset",
]
