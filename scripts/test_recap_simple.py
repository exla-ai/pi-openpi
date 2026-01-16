#!/usr/bin/env python3
"""Simple smoke test for RECAP implementation.

Just verifies the modules can be imported and basic structures work.
"""

import sys
sys.path.insert(0, "src")

print("Testing imports...")
from openpi.recap.value_function import ValueFunction, ValueFunctionConfig, compute_improvement_indicator
from openpi.recap.pi0_recap import Pi0RECAP, Pi0RECAPConfig
from openpi.recap.trainer import RECAPTrainer, RECAPConfig

print("All imports successful!")

# Test config creation
print("\nTesting config creation...")
value_config = ValueFunctionConfig(paligemma_variant="dummy", num_bins=201)
print(f"ValueFunctionConfig: {value_config}")

policy_config = Pi0RECAPConfig(
    paligemma_variant="dummy",
    action_expert_variant="dummy",
    action_dim=14,
    action_horizon=50,
    pi05=True,
)
print(f"Pi0RECAPConfig: {policy_config}")

recap_config = RECAPConfig(
    policy_config=policy_config,
    value_config=value_config,
)
print(f"RECAPConfig: {recap_config}")

# Test improvement indicator
print("\nTesting improvement indicator...")
import jax.numpy as jnp
advantages = jnp.array([-1.0, 0.0, 1.0, 2.0])
indicators = compute_improvement_indicator(advantages)
print(f"Advantages: {advantages}")
print(f"Indicators: {indicators}")
assert indicators.tolist() == [False, False, True, True], "Indicator logic wrong!"

print("\n" + "=" * 60)
print("All smoke tests PASSED!")
print("=" * 60)
