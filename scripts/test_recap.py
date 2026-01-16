#!/usr/bin/env python3
"""Test script for RECAP implementation.

This script validates that the RECAP components work correctly:
1. Value function forward pass and loss computation
2. Pi0RECAP with advantage conditioning
3. Training loop with fake data

Run with:
    python scripts/test_recap.py
"""

import logging
import sys

import jax
import jax.numpy as jnp
import flax.nnx as nnx

# Add src to path
sys.path.insert(0, "src")

from openpi.recap.value_function import ValueFunction, ValueFunctionConfig, compute_improvement_indicator
from openpi.recap.pi0_recap import Pi0RECAP, Pi0RECAPConfig
from openpi.models import model as _model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


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
    """Test the distributional value function."""
    logger.info("=" * 60)
    logger.info("Testing Value Function")
    logger.info("=" * 60)

    # Use dummy model for faster testing
    config = ValueFunctionConfig(
        paligemma_variant="dummy",
        num_bins=201,
        value_hidden_dim=256,
    )

    rng = jax.random.key(42)
    logger.info("Initializing value function...")
    value_fn = config.create(rng)
    logger.info("Value function initialized!")

    # Test forward pass
    batch_size = 2
    obs = create_fake_observation(batch_size)

    logger.info("Testing forward pass...")
    logits = value_fn.forward(obs)
    assert logits.shape == (batch_size, 201), f"Expected shape (2, 201), got {logits.shape}"
    logger.info(f"Forward pass OK! Output shape: {logits.shape}")

    # Test loss computation
    logger.info("Testing loss computation...")
    time_to_completion = jnp.array([50, 100], dtype=jnp.int32)
    loss = value_fn.compute_loss(obs, time_to_completion)
    assert loss.shape == (), f"Expected scalar loss, got shape {loss.shape}"
    logger.info(f"Loss computation OK! Loss: {float(loss):.4f}")

    # Test value prediction
    logger.info("Testing value prediction...")
    expected_time, probs = value_fn.predict_value(obs)
    assert expected_time.shape == (batch_size,), f"Expected shape ({batch_size},), got {expected_time.shape}"
    assert probs.shape == (batch_size, 201), f"Expected shape ({batch_size}, 201), got {probs.shape}"
    assert jnp.allclose(jnp.sum(probs, axis=-1), 1.0), "Probabilities should sum to 1"
    logger.info(f"Value prediction OK! Expected times: {expected_time}")

    # Test advantage computation
    logger.info("Testing advantage computation...")
    advantage = value_fn.compute_advantage(obs, time_to_completion)
    assert advantage.shape == (batch_size,), f"Expected shape ({batch_size},), got {advantage.shape}"
    logger.info(f"Advantage computation OK! Advantages: {advantage}")

    # Test improvement indicator
    logger.info("Testing improvement indicator...")
    indicator = compute_improvement_indicator(advantage)
    assert indicator.shape == (batch_size,), f"Expected shape ({batch_size},), got {indicator.shape}"
    assert indicator.dtype == jnp.bool_, f"Expected bool dtype, got {indicator.dtype}"
    logger.info(f"Improvement indicator OK! Indicators: {indicator}")

    logger.info("Value function tests PASSED!")
    return True


def test_pi0_recap():
    """Test Pi0 with RECAP advantage conditioning."""
    logger.info("=" * 60)
    logger.info("Testing Pi0RECAP")
    logger.info("=" * 60)

    # Use dummy model for faster testing
    config = Pi0RECAPConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=14,
        action_horizon=50,
        max_token_len=256,
        pi05=True,  # Use pi0.5 for adaRMS
    )

    rng = jax.random.key(42)
    logger.info("Initializing Pi0RECAP...")
    policy = config.create(rng)
    logger.info("Pi0RECAP initialized!")

    batch_size = 2
    obs = create_fake_observation(batch_size)
    actions = jnp.zeros((batch_size, 50, 14), dtype=jnp.float32)

    # Test loss without advantage conditioning (backward compatible)
    logger.info("Testing loss computation (no advantage conditioning)...")
    rng, loss_rng = jax.random.split(rng)
    loss = policy.compute_loss(loss_rng, obs, actions, train=True)
    assert loss.shape == (batch_size, 50), f"Expected shape ({batch_size}, 50), got {loss.shape}"
    logger.info(f"Loss (no conditioning) OK! Mean loss: {float(jnp.mean(loss)):.4f}")

    # Test loss with advantage conditioning
    logger.info("Testing loss computation (with advantage conditioning)...")
    improvement_indicator = jnp.array([True, False], dtype=jnp.bool_)
    rng, loss_rng = jax.random.split(rng)
    loss_conditioned = policy.compute_loss(
        loss_rng, obs, actions, train=True, improvement_indicator=improvement_indicator
    )
    assert loss_conditioned.shape == (batch_size, 50), f"Expected shape ({batch_size}, 50), got {loss_conditioned.shape}"
    logger.info(f"Loss (with conditioning) OK! Mean loss: {float(jnp.mean(loss_conditioned)):.4f}")

    # Test action sampling without conditioning
    logger.info("Testing action sampling (no conditioning)...")
    rng, sample_rng = jax.random.split(rng)
    sampled_actions = policy.sample_actions(sample_rng, obs, num_steps=2)
    assert sampled_actions.shape == (batch_size, 50, 14), f"Expected shape ({batch_size}, 50, 14), got {sampled_actions.shape}"
    logger.info(f"Sampling (no conditioning) OK! Output shape: {sampled_actions.shape}")

    # Test action sampling with conditioning
    logger.info("Testing action sampling (with conditioning)...")
    rng, sample_rng = jax.random.split(rng)
    sampled_actions_conditioned = policy.sample_actions(
        sample_rng, obs, num_steps=2, improvement_indicator=improvement_indicator
    )
    assert sampled_actions_conditioned.shape == (batch_size, 50, 14)
    logger.info(f"Sampling (with conditioning) OK! Output shape: {sampled_actions_conditioned.shape}")

    logger.info("Pi0RECAP tests PASSED!")
    return True


def test_training_step():
    """Test a single training step with both value function and policy."""
    logger.info("=" * 60)
    logger.info("Testing Training Step")
    logger.info("=" * 60)

    import optax

    batch_size = 2
    rng = jax.random.key(42)

    # Initialize models
    value_config = ValueFunctionConfig(paligemma_variant="dummy", value_hidden_dim=256)
    policy_config = Pi0RECAPConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=14,
        action_horizon=50,
        pi05=True,
    )

    rng, v_rng, p_rng = jax.random.split(rng, 3)
    value_fn = value_config.create(v_rng)
    policy = policy_config.create(p_rng)

    # Create optimizers
    value_optimizer = optax.adam(1e-4)
    policy_optimizer = optax.adam(1e-5)

    value_params = nnx.state(value_fn, nnx.Param)
    policy_params = nnx.state(policy, nnx.Param)

    value_opt_state = value_optimizer.init(value_params)
    policy_opt_state = policy_optimizer.init(policy_params)

    # Create fake data
    obs = create_fake_observation(batch_size)
    actions = jnp.zeros((batch_size, 50, 14), dtype=jnp.float32)
    time_to_completion = jnp.array([50, 100], dtype=jnp.int32)

    # Value function training step
    logger.info("Testing value function training step...")

    def value_loss_fn(params):
        nnx.update(value_fn, params)
        return value_fn.compute_loss(obs, time_to_completion)

    value_loss, value_grads = jax.value_and_grad(value_loss_fn)(value_params)
    value_updates, value_opt_state = value_optimizer.update(value_grads, value_opt_state, value_params)
    value_params = optax.apply_updates(value_params, value_updates)
    logger.info(f"Value training step OK! Loss: {float(value_loss):.4f}")

    # Compute advantage for policy training
    nnx.update(value_fn, value_params)
    advantage = value_fn.compute_advantage(obs, time_to_completion)
    improvement_indicator = compute_improvement_indicator(advantage)
    logger.info(f"Computed advantages: {advantage}, indicators: {improvement_indicator}")

    # Policy training step
    logger.info("Testing policy training step...")
    rng, train_rng = jax.random.split(rng)

    def policy_loss_fn(params):
        nnx.update(policy, params)
        losses = policy.compute_loss(
            train_rng, obs, actions, train=True, improvement_indicator=improvement_indicator
        )
        return jnp.mean(losses)

    policy_loss, policy_grads = jax.value_and_grad(policy_loss_fn)(policy_params)
    policy_updates, policy_opt_state = policy_optimizer.update(policy_grads, policy_opt_state, policy_params)
    policy_params = optax.apply_updates(policy_params, policy_updates)
    logger.info(f"Policy training step OK! Loss: {float(policy_loss):.4f}")

    logger.info("Training step tests PASSED!")
    return True


def main():
    """Run all tests."""
    logger.info("Starting RECAP validation tests")
    logger.info(f"JAX devices: {jax.devices()}")

    tests = [
        ("Value Function", test_value_function),
        ("Pi0RECAP", test_pi0_recap),
        ("Training Step", test_training_step),
    ]

    results = {}
    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            logger.error(f"Test {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

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
        logger.info("All tests PASSED!")
        return 0
    else:
        logger.error("Some tests FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
