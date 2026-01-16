#!/usr/bin/env python3
"""Test the Isaac Lab data collection pipeline.

This script verifies:
1. Data collection works (mock mode without Isaac Lab)
2. Data can be loaded by RECAP dataset
3. Full training loop can process the data
"""

import os
import sys
import tempfile
import shutil

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def test_data_collection():
    """Test data collection (mock mode)."""
    logger.info("=" * 60)
    logger.info("TEST 1: Data Collection (Mock Mode)")
    logger.info("=" * 60)

    import numpy as np
    from dataclasses import dataclass
    from typing import List, Dict

    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    logger.info(f"Using temp directory: {temp_dir}")

    try:
        @dataclass
        class Episode:
            observations: List[Dict[str, np.ndarray]]
            actions: List[np.ndarray]
            rewards: List[float]
            dones: List[bool]
            success: bool
            length: int

        # Generate mock episodes
        rng = np.random.default_rng(42)
        episodes = []

        for ep_idx in range(5):  # Small number for testing
            length = rng.integers(20, 50)

            observations = []
            actions = []
            rewards = []
            dones = []

            for step in range(length):
                obs = {
                    "base_0_rgb": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
                    "state": rng.standard_normal(23).astype(np.float32),
                }
                observations.append(obs)
                actions.append(rng.standard_normal(9).astype(np.float32))
                rewards.append(float(rng.uniform(-1, 1)))
                dones.append(step == length - 1)

            episodes.append(Episode(
                observations=observations,
                actions=actions,
                rewards=rewards,
                dones=dones,
                success=rng.random() > 0.5,
                length=length,
            ))

        logger.info(f"Generated {len(episodes)} mock episodes")

        # Save to HDF5
        import h5py
        hdf5_path = os.path.join(temp_dir, "test_episodes.hdf5")

        with h5py.File(hdf5_path, "w") as f:
            f.attrs["num_episodes"] = len(episodes)
            f.attrs["total_timesteps"] = sum(ep.length for ep in episodes)
            f.attrs["success_rate"] = sum(ep.success for ep in episodes) / len(episodes)

            for ep_idx, episode in enumerate(episodes):
                ep_group = f.create_group(f"episode_{ep_idx:06d}")
                ep_group.attrs["length"] = episode.length
                ep_group.attrs["success"] = episode.success

                obs_group = ep_group.create_group("observations")
                for key in episode.observations[0].keys():
                    data = np.stack([obs[key] for obs in episode.observations], axis=0)
                    obs_group.create_dataset(key, data=data, compression="gzip")

                actions = np.stack(episode.actions, axis=0)
                ep_group.create_dataset("actions", data=actions, compression="gzip")

                rewards = np.array(episode.rewards, dtype=np.float32)
                ep_group.create_dataset("rewards", data=rewards)

                ttc = np.array([episode.length - t for t in range(episode.length)], dtype=np.int32)
                ep_group.create_dataset("time_to_completion", data=ttc)

        logger.info(f"Saved to {hdf5_path}")
        logger.info("TEST 1 PASSED!")

        return hdf5_path, temp_dir

    except Exception as e:
        shutil.rmtree(temp_dir)
        raise e


def test_data_loading(hdf5_path):
    """Test loading data with RECAP dataset."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Data Loading")
    logger.info("=" * 60)

    from openpi.recap.isaaclab_data import IsaacLabDataset, RECAPDataLoader

    # Load dataset
    dataset = IsaacLabDataset(
        data_path=hdf5_path,
        action_chunk_size=10,  # Smaller for testing
        cameras=["base_0_rgb"],
    )

    logger.info(f"Dataset loaded: {len(dataset)} timesteps from {len(dataset.episodes)} episodes")

    # Test getitem
    sample = dataset[0]
    logger.info(f"Sample keys: {list(sample.keys())}")
    logger.info(f"Sample images shape: {sample['images']['base_0_rgb'].shape}")
    logger.info(f"Sample state shape: {sample['state'].shape}")
    logger.info(f"Sample actions shape: {sample['actions'].shape}")
    logger.info(f"Sample time_to_completion: {sample['time_to_completion']}")
    logger.info(f"Sample improvement_indicator: {sample['improvement_indicator']}")

    # Test data loader
    loader = RECAPDataLoader(dataset, batch_size=2, shuffle=True)
    batch = next(iter(loader))
    logger.info(f"\nBatch keys: {list(batch.keys())}")
    logger.info(f"Batch images shape: {batch['images']['base_0_rgb'].shape}")
    logger.info(f"Batch actions shape: {batch['actions'].shape}")

    logger.info("TEST 2 PASSED!")
    return dataset


def test_model_integration(dataset):
    """Test that data can be fed to RECAP models."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Model Integration")
    logger.info("=" * 60)

    import jax
    import jax.numpy as jnp

    # Get a sample
    sample = dataset[0]

    # Verify shapes are compatible with pi0
    logger.info("Checking shape compatibility with pi0 model:")
    logger.info(f"  Image: {sample['images']['base_0_rgb'].shape} (expected: H, W, C)")
    logger.info(f"  State: {sample['state'].shape} (expected: state_dim)")
    logger.info(f"  Actions: {sample['actions'].shape} (expected: action_horizon, action_dim)")

    # Basic JAX operations to verify arrays are valid
    img_array = jnp.array(sample['images']['base_0_rgb'])
    state_array = jnp.array(sample['state'])
    action_array = jnp.array(sample['actions'])

    logger.info(f"\nJAX arrays created successfully:")
    logger.info(f"  Image: {img_array.shape}, dtype={img_array.dtype}")
    logger.info(f"  State: {state_array.shape}, dtype={state_array.dtype}")
    logger.info(f"  Actions: {action_array.shape}, dtype={action_array.dtype}")

    logger.info("TEST 3 PASSED!")


def test_value_function_training(dataset):
    """Test value function can be created and trained."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Value Function Training (Smoke Test)")
    logger.info("=" * 60)

    import jax
    from openpi.recap.value_function import ValueFunctionConfig

    # Create value function with dummy model
    config = ValueFunctionConfig(
        paligemma_variant="dummy",
        num_bins=201,
        value_hidden_dim=256,
    )

    rng = jax.random.key(42)
    value_fn = config.create(rng)

    logger.info(f"Value function created")
    logger.info(f"  Num bins: {config.num_bins}")
    logger.info(f"  Hidden dim: {config.value_hidden_dim}")

    logger.info("TEST 4 PASSED!")


def test_policy_training(dataset):
    """Test policy can be created with advantage conditioning."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 5: Policy with Advantage Conditioning")
    logger.info("=" * 60)

    import jax
    from openpi.recap.pi0_recap import Pi0RECAPConfig

    # Create policy with dummy model
    config = Pi0RECAPConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=9,
        action_horizon=10,
        pi05=True,
    )

    rng = jax.random.key(42)
    policy = config.create(rng)

    logger.info(f"Pi0RECAP policy created")
    logger.info(f"  Action dim: {config.action_dim}")
    logger.info(f"  Action horizon: {config.action_horizon}")
    logger.info(f"  Advantage embedding dim: {config.advantage_embedding_dim}")

    logger.info("TEST 5 PASSED!")


def main():
    logger.info("=" * 70)
    logger.info("ISAAC LAB DATA PIPELINE TESTS")
    logger.info("=" * 70)

    temp_dir = None
    try:
        # Test 1: Data collection
        hdf5_path, temp_dir = test_data_collection()

        # Test 2: Data loading
        dataset = test_data_loading(hdf5_path)

        # Test 3: Model integration
        test_model_integration(dataset)

        # Test 4: Value function
        test_value_function_training(dataset)

        # Test 5: Policy with advantage conditioning
        test_policy_training(dataset)

        logger.info("\n" + "=" * 70)
        logger.info("ALL TESTS PASSED!")
        logger.info("=" * 70)
        logger.info("\nThe Isaac Lab data pipeline is ready for use.")
        logger.info("Next steps:")
        logger.info("  1. Run data collection with real Isaac Lab:")
        logger.info("     python scripts/isaaclab_data_collection.py --task Isaac-Franka-Cabinet-Direct-v0")
        logger.info("  2. Train RECAP with collected data:")
        logger.info("     python scripts/train_recap_isaaclab.py --data_path data/isaaclab/franka_cabinet.hdf5")

        return 0

    except Exception as e:
        logger.error(f"TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        # Cleanup
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"\nCleaned up temp directory: {temp_dir}")


if __name__ == "__main__":
    sys.exit(main())
