#!/usr/bin/env python3
"""Test RECAP model initialization and forward pass with dummy models."""

import sys
sys.path.insert(0, "src")

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

import jax
import jax.numpy as jnp

logger.info(f"JAX devices: {jax.devices()}")

# Import RECAP modules
from openpi.recap.value_function import ValueFunction, ValueFunctionConfig, compute_improvement_indicator
from openpi.recap.pi0_recap import Pi0RECAP, Pi0RECAPConfig
from openpi.models import model as _model

def create_fake_observation(batch_size: int = 2) -> _model.Observation:
    """Create a fake observation for testing."""
    return _model.Observation(
        images={
            "base_0_rgb": jnp.zeros((batch_size, 224, 224, 3), dtype=jnp.float32),
            "left_wrist_0_rgb": jnp.zeros((batch_size, 224, 224, 3), dtype=jnp.float32),
            "right_wrist_0_rgb": jnp.zeros((batch_size, 224, 224, 3), dtype=jnp.float32),
        },
        image_masks={
            "base_0_rgb": jnp.ones(batch_size, dtype=jnp.bool_),
            "left_wrist_0_rgb": jnp.ones(batch_size, dtype=jnp.bool_),
            "right_wrist_0_rgb": jnp.ones(batch_size, dtype=jnp.bool_),
        },
        state=jnp.zeros((batch_size, 14), dtype=jnp.float32),
        tokenized_prompt=jnp.ones((batch_size, 64), dtype=jnp.int32),
        tokenized_prompt_mask=jnp.ones((batch_size, 64), dtype=jnp.bool_),
    )


def test_value_function():
    """Test value function with dummy model."""
    logger.info("=" * 60)
    logger.info("Testing Value Function (dummy model)")
    logger.info("=" * 60)

    config = ValueFunctionConfig(
        paligemma_variant="dummy",
        num_bins=201,
        value_hidden_dim=256,
    )

    rng = jax.random.key(42)
    logger.info("Initializing value function (this may take a moment)...")
    value_fn = config.create(rng)
    logger.info("Value function initialized!")

    # Test forward pass with JIT
    batch_size = 2
    obs = create_fake_observation(batch_size)

    logger.info("Testing forward pass (first call compiles)...")

    @jax.jit
    def forward_fn(obs):
        return value_fn.forward(obs)

    logits = forward_fn(obs)
    jax.block_until_ready(logits)
    logger.info(f"Forward pass OK! Logits shape: {logits.shape}")

    # Test predict value
    logger.info("Testing value prediction...")
    expected_time, probs = value_fn.predict_value(obs)
    logger.info(f"Expected time: {expected_time}, sum of probs: {jnp.sum(probs, axis=-1)}")

    # Test loss computation
    logger.info("Testing loss computation...")
    time_to_completion = jnp.array([50, 100], dtype=jnp.int32)
    loss = value_fn.compute_loss(obs, time_to_completion)
    logger.info(f"Loss: {float(loss):.4f}")

    logger.info("Value function test PASSED!")
    return True


def test_pi0_recap():
    """Test Pi0RECAP with dummy model."""
    logger.info("=" * 60)
    logger.info("Testing Pi0RECAP (dummy model)")
    logger.info("=" * 60)

    config = Pi0RECAPConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=14,
        action_horizon=50,
        max_token_len=256,
        pi05=True,
    )

    rng = jax.random.key(42)
    logger.info("Initializing Pi0RECAP (this may take a moment)...")
    policy = config.create(rng)
    logger.info("Pi0RECAP initialized!")

    batch_size = 2
    obs = create_fake_observation(batch_size)
    actions = jnp.zeros((batch_size, 50, 14), dtype=jnp.float32)
    improvement_indicator = jnp.array([True, False], dtype=jnp.bool_)

    # Test loss with advantage conditioning
    logger.info("Testing loss computation with advantage conditioning...")
    rng, loss_rng = jax.random.split(rng)

    @jax.jit
    def loss_fn(obs, actions, indicator):
        return policy.compute_loss(loss_rng, obs, actions, train=True, improvement_indicator=indicator)

    loss = loss_fn(obs, actions, improvement_indicator)
    jax.block_until_ready(loss)
    logger.info(f"Loss (with conditioning): mean={float(jnp.mean(loss)):.4f}, shape={loss.shape}")

    # Test action sampling (reduced num_steps for speed)
    logger.info("Testing action sampling...")
    rng, sample_rng = jax.random.split(rng)

    @jax.jit
    def sample_fn(obs, indicator):
        return policy.sample_actions(sample_rng, obs, num_steps=2, improvement_indicator=indicator)

    sampled_actions = sample_fn(obs, improvement_indicator)
    jax.block_until_ready(sampled_actions)
    logger.info(f"Sampled actions shape: {sampled_actions.shape}")

    logger.info("Pi0RECAP test PASSED!")
    return True


def main():
    """Run model tests."""
    logger.info("Starting RECAP model tests")

    results = {}

    try:
        results["Value Function"] = test_value_function()
    except Exception as e:
        logger.error(f"Value Function test FAILED: {e}")
        import traceback
        traceback.print_exc()
        results["Value Function"] = False

    try:
        results["Pi0RECAP"] = test_pi0_recap()
    except Exception as e:
        logger.error(f"Pi0RECAP test FAILED: {e}")
        import traceback
        traceback.print_exc()
        results["Pi0RECAP"] = False

    # Summary
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)

    all_passed = True
    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        logger.info(f"  {name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        logger.info("All model tests PASSED!")
        return 0
    else:
        logger.error("Some model tests FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
