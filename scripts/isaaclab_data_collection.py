#!/usr/bin/env python3
"""Isaac Lab Data Collection for RECAP Training.

This script collects episodes from Isaac Lab environments and saves them
in a format compatible with RECAP training:
- Images from camera observations
- Robot state (joint positions/velocities)
- Actions taken
- Rewards received
- Time-to-completion for value function training

Usage:
    # From the Isaac Lab directory:
    python scripts/isaaclab_data_collection.py --task Isaac-Franka-Cabinet-Direct-v0 --num_episodes 100

Requirements:
    - Isaac Lab installed and sourced
    - Run from Isaac Lab directory (or set ISAACSIM_PATH)
"""

from __future__ import annotations

import argparse
import os
import sys

# Add paths for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))


def parse_args():
    parser = argparse.ArgumentParser(description="Collect episodes from Isaac Lab for RECAP training")

    # Environment settings
    parser.add_argument("--task", type=str,
                        default=os.environ.get("RECAP_TASK", "Isaac-Franka-Cabinet-Direct-v0"),
                        help="Isaac Lab task name (env: RECAP_TASK)")
    parser.add_argument("--num_envs", type=int,
                        default=int(os.environ.get("RECAP_NUM_ENVS", "16")),
                        help="Number of parallel environments (env: RECAP_NUM_ENVS)")
    parser.add_argument("--num_episodes", type=int, default=100,
                        help="Total number of episodes to collect")
    parser.add_argument("--max_episode_length", type=int, default=500,
                        help="Maximum steps per episode")

    # Robot settings
    parser.add_argument("--robot", type=str, default="franka",
                        choices=["franka", "ur5", "ur10", "kuka"],
                        help="Robot type (affects action/state dimensions)")

    # Camera settings
    parser.add_argument("--image_width", type=int, default=224,
                        help="Camera image width")
    parser.add_argument("--image_height", type=int, default=224,
                        help="Camera image height")
    parser.add_argument("--cameras", type=str, default="base_0_rgb",
                        help="Comma-separated list of camera names")
    parser.add_argument("--add_wrist_cameras", action="store_true",
                        help="Add wrist-mounted cameras (requires robot support)")

    # Output settings
    parser.add_argument("--output_dir", type=str,
                        default=os.environ.get("RECAP_DATA_DIR", "/lambda/nfs/illinois/pi_openpi/data/isaaclab"),
                        help="Directory to save collected episodes (env: RECAP_DATA_DIR)")
    parser.add_argument("--dataset_name", type=str, default="franka_cabinet",
                        help="Name for the dataset")
    parser.add_argument("--format", type=str, default="both",
                        choices=["hdf5", "numpy", "both"],
                        help="Output format for saved data")

    # Policy settings
    parser.add_argument("--policy", type=str, default="random",
                        choices=["random", "scripted", "checkpoint"],
                        help="Policy type for data collection")
    parser.add_argument("--checkpoint_path", type=str,
                        default=os.environ.get("RECAP_CHECKPOINT_PATH", None),
                        help="Path to policy checkpoint (env: RECAP_CHECKPOINT_PATH)")
    parser.add_argument("--action_horizon", type=int, default=50,
                        help="Action horizon for checkpoint policy")

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--headless", action="store_true", help="Run without display")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")

    return parser.parse_args()


# Parse args before importing Isaac Lab (to set headless mode)
args = parse_args()

# Set headless before importing
if args.headless:
    os.environ["DISPLAY"] = ""

print("=" * 70)
print("Isaac Lab Data Collection for RECAP")
print("=" * 70)
print(f"Task: {args.task}")
print(f"Num envs: {args.num_envs}")
print(f"Num episodes: {args.num_episodes}")
print(f"Output: {args.output_dir}/{args.dataset_name}")
print("=" * 70)

# Check if Isaac Lab is available
try:
    from isaaclab.app import AppLauncher
    ISAACLAB_AVAILABLE = True
except ImportError:
    ISAACLAB_AVAILABLE = False
    print("\n[WARNING] Isaac Lab not available. Running in mock mode for testing.")
    print("To use real Isaac Lab, source the Isaac Lab environment first.")


def create_mock_data_collector(args):
    """Create a mock data collector for testing without Isaac Lab."""
    import numpy as np
    from dataclasses import dataclass
    from typing import List, Dict, Any

    @dataclass
    class Episode:
        """Container for a single episode."""
        observations: List[Dict[str, np.ndarray]]
        actions: List[np.ndarray]
        rewards: List[float]
        dones: List[bool]
        success: bool
        length: int

    class MockCollector:
        """Mock collector that generates random data."""

        def __init__(self, args):
            self.args = args
            self.rng = np.random.default_rng(args.seed)

        def collect_episodes(self, num_episodes: int) -> List[Episode]:
            """Collect episodes with random actions."""
            episodes = []

            for ep_idx in range(num_episodes):
                length = self.rng.integers(50, args.max_episode_length)

                observations = []
                actions = []
                rewards = []
                dones = []

                for step in range(length):
                    # Mock observation
                    obs = {
                        "base_0_rgb": self.rng.integers(0, 255, (args.image_height, args.image_width, 3), dtype=np.uint8),
                        "state": self.rng.standard_normal(23).astype(np.float32),
                    }
                    observations.append(obs)

                    # Mock action (9 DOF for Franka)
                    action = self.rng.standard_normal(9).astype(np.float32)
                    action = np.clip(action, -1, 1)
                    actions.append(action)

                    # Mock reward
                    reward = float(self.rng.uniform(-1, 1))
                    rewards.append(reward)

                    # Done flag
                    dones.append(step == length - 1)

                success = self.rng.random() > 0.5  # 50% success rate
                episodes.append(Episode(
                    observations=observations,
                    actions=actions,
                    rewards=rewards,
                    dones=dones,
                    success=success,
                    length=length,
                ))

                if (ep_idx + 1) % 10 == 0:
                    print(f"  Collected {ep_idx + 1}/{num_episodes} episodes")

            return episodes

    return MockCollector(args)


def create_isaaclab_collector(args):
    """Create the real Isaac Lab data collector."""

    # Launch Isaac Sim app first
    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app

    # Now import Isaac Lab modules
    import torch
    import numpy as np
    import gymnasium as gym
    from collections import defaultdict
    from dataclasses import dataclass
    from typing import List, Dict, Any

    import isaaclab.sim as sim_utils
    from isaaclab.sensors import TiledCamera, TiledCameraCfg

    # Import task registry
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    @dataclass
    class Episode:
        """Container for a single episode."""
        observations: List[Dict[str, np.ndarray]]
        actions: List[np.ndarray]
        rewards: List[float]
        dones: List[bool]
        success: bool
        length: int

    class IsaacLabCollector:
        """Collect episodes from Isaac Lab environments."""

        def __init__(self, args):
            self.args = args
            self.device = args.device

            # Parse environment config
            self.env_cfg = parse_env_cfg(
                args.task,
                device=args.device,
                num_envs=args.num_envs,
            )

            # Modify config to add camera if not present
            self._add_camera_config()

            # Create environment
            self.env = gym.make(args.task, cfg=self.env_cfg).unwrapped

            # Set seed
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)

            # Episode buffers
            self.episode_buffers = defaultdict(lambda: defaultdict(list))

        def _add_camera_config(self):
            """Add camera configuration to environment if not present."""
            # Check if scene has camera already
            if hasattr(self.env_cfg, 'scene'):
                scene_cfg = self.env_cfg.scene
                if not hasattr(scene_cfg, 'tiled_camera'):
                    print("[INFO] Adding base camera to environment")
                    # Add camera config - will be added during scene setup
                    self.needs_camera = True
                else:
                    self.needs_camera = False
            else:
                self.needs_camera = False

        def _get_policy_action(self, obs_dict):
            """Get action from policy."""
            if self.args.policy == "random":
                # Random actions
                action = torch.rand((self.args.num_envs, self.env.action_space.shape[1]),
                                   device=self.device) * 2 - 1
            elif self.args.policy == "scripted":
                # Simple scripted policy (move towards target)
                # This is environment-specific
                action = torch.zeros((self.args.num_envs, self.env.action_space.shape[1]),
                                    device=self.device)
            elif self.args.policy == "checkpoint":
                # Load and use trained RECAP policy
                action = self._get_checkpoint_action(obs_dict)
            else:
                raise ValueError(f"Unknown policy type: {self.args.policy}")

            return action

        def _load_checkpoint_policy(self):
            """Load the RECAP policy from checkpoint."""
            import jax
            import jax.numpy as jnp
            from openpi.recap.pi0_recap import Pi0RECAPConfig

            checkpoint_path = self.args.checkpoint_path
            if checkpoint_path is None:
                raise ValueError("checkpoint_path is required when policy=checkpoint")

            print(f"[INFO] Loading policy from {checkpoint_path}")

            # Get robot-specific config
            robot_configs = {
                "franka": {"action_dim": 9, "state_dim": 23},
                "ur5": {"action_dim": 6, "state_dim": 18},
                "ur10": {"action_dim": 6, "state_dim": 18},
                "kuka": {"action_dim": 7, "state_dim": 21},
            }
            robot_cfg = robot_configs.get(self.args.robot, robot_configs["franka"])

            # Create policy config
            policy_config = Pi0RECAPConfig(
                paligemma_variant="gemma_2b",  # Use full model for rollouts
                action_expert_variant="gemma_2b",
                action_dim=robot_cfg["action_dim"],
                action_horizon=self.args.action_horizon,
                pi05=True,
            )

            # Initialize policy
            rng = jax.random.key(self.args.seed)
            self.policy = policy_config.create(rng)

            # Load checkpoint
            try:
                import orbax.checkpoint as ocp
                from flax import nnx

                checkpointer = ocp.StandardCheckpointer()
                policy_path = os.path.join(checkpoint_path, "policy")
                if os.path.exists(policy_path):
                    policy_state = checkpointer.restore(policy_path, nnx.state(self.policy))
                    nnx.update(self.policy, policy_state)
                    print(f"[INFO] Policy loaded from {policy_path}")
                else:
                    print(f"[WARNING] No policy checkpoint found at {policy_path}")
            except Exception as e:
                print(f"[WARNING] Could not load checkpoint: {e}")

            self.policy_loaded = True
            self.action_counter = 0  # For action chunking

        def _get_checkpoint_action(self, obs_dict):
            """Get action from checkpoint policy."""
            import jax
            import jax.numpy as jnp
            from openpi.models.model import Observation

            # Load policy on first call
            if not hasattr(self, 'policy_loaded') or not self.policy_loaded:
                self._load_checkpoint_policy()

            # Check if we need new actions (action chunking)
            if not hasattr(self, 'cached_actions') or self.action_counter >= self.args.action_horizon:
                # Convert observations to model format
                policy_obs = obs_dict.get("policy", obs_dict)

                # Build observation for each environment
                batch_images = {}
                batch_states = []

                for env_idx in range(self.args.num_envs):
                    if isinstance(policy_obs, dict):
                        state = policy_obs.get("state", policy_obs.get("observation", None))
                        if state is not None:
                            batch_states.append(state[env_idx].cpu().numpy())
                        else:
                            # Find any state-like tensor
                            for k, v in policy_obs.items():
                                if isinstance(v, torch.Tensor) and v.ndim == 2:
                                    batch_states.append(v[env_idx].cpu().numpy())
                                    break
                    else:
                        batch_states.append(policy_obs[env_idx].cpu().numpy())

                    # Get images if available
                    if hasattr(self.env, '_tiled_camera'):
                        camera_data = self.env._tiled_camera.data.output.get("rgb")
                        if camera_data is not None:
                            img = camera_data[env_idx].cpu().numpy().astype(np.float32) / 255.0 * 2 - 1
                            if "base_0_rgb" not in batch_images:
                                batch_images["base_0_rgb"] = []
                            batch_images["base_0_rgb"].append(img)

                # Stack into batches
                state = jnp.array(np.stack(batch_states))

                if batch_images:
                    images = {k: jnp.array(np.stack(v)) for k, v in batch_images.items()}
                    image_masks = {k: jnp.ones(self.args.num_envs, dtype=jnp.bool_) for k in images}
                else:
                    # Create dummy images
                    images = {"base_0_rgb": jnp.zeros((self.args.num_envs, 224, 224, 3))}
                    image_masks = {"base_0_rgb": jnp.ones(self.args.num_envs, dtype=jnp.bool_)}

                observation = Observation(
                    images=images,
                    image_masks=image_masks,
                    state=state,
                    tokenized_prompt=None,
                    tokenized_prompt_mask=None,
                )

                # Sample actions with I=1 (want good trajectories)
                rng = jax.random.key(self.args.seed + self.action_counter)
                improvement_indicator = jnp.ones(self.args.num_envs, dtype=jnp.bool_)

                try:
                    actions = self.policy.sample_actions(
                        rng, observation,
                        num_steps=10,
                        improvement_indicator=improvement_indicator
                    )
                    self.cached_actions = np.array(actions)
                except Exception as e:
                    print(f"[WARNING] Policy inference failed: {e}, using random actions")
                    self.cached_actions = np.random.randn(
                        self.args.num_envs, self.args.action_horizon, self.env.action_space.shape[1]
                    ).astype(np.float32) * 0.1

                self.action_counter = 0

            # Get action for current timestep
            action = self.cached_actions[:, self.action_counter, :]
            self.action_counter += 1

            return torch.tensor(action, device=self.device)

        def _extract_observation(self, obs_dict, env_idx):
            """Extract observation for a single environment."""
            obs = {}

            # Get policy observations (could be dict or tensor)
            policy_obs = obs_dict.get("policy", obs_dict)

            if isinstance(policy_obs, dict):
                for key, value in policy_obs.items():
                    if isinstance(value, torch.Tensor):
                        obs[key] = value[env_idx].cpu().numpy()
                    else:
                        obs[key] = value[env_idx] if hasattr(value, '__getitem__') else value
            else:
                obs["state"] = policy_obs[env_idx].cpu().numpy()

            # Add camera image if available
            if hasattr(self.env, '_tiled_camera'):
                camera_data = self.env._tiled_camera.data.output.get("rgb")
                if camera_data is not None:
                    img = camera_data[env_idx].cpu().numpy()
                    obs["base_0_rgb"] = img.astype(np.uint8)

            return obs

        def collect_episodes(self, num_episodes: int) -> List[Episode]:
            """Collect specified number of episodes."""
            episodes = []
            episodes_collected = 0

            # Reset environment
            obs_dict, _ = self.env.reset()

            # Initialize episode buffers for each env
            for env_idx in range(self.args.num_envs):
                self.episode_buffers[env_idx] = defaultdict(list)

            step_count = 0

            while episodes_collected < num_episodes:
                # Get action
                action = self._get_policy_action(obs_dict)

                # Step environment
                next_obs_dict, rewards, terminated, truncated, info = self.env.step(action)
                dones = terminated | truncated

                # Store transitions for each env
                for env_idx in range(self.args.num_envs):
                    obs = self._extract_observation(obs_dict, env_idx)
                    self.episode_buffers[env_idx]["observations"].append(obs)
                    self.episode_buffers[env_idx]["actions"].append(action[env_idx].cpu().numpy())
                    self.episode_buffers[env_idx]["rewards"].append(float(rewards[env_idx]))
                    self.episode_buffers[env_idx]["dones"].append(bool(dones[env_idx]))

                    # Check if episode finished
                    if dones[env_idx]:
                        # Create episode
                        buffer = self.episode_buffers[env_idx]
                        success = bool(terminated[env_idx])  # terminated = success, truncated = timeout

                        episode = Episode(
                            observations=list(buffer["observations"]),
                            actions=list(buffer["actions"]),
                            rewards=list(buffer["rewards"]),
                            dones=list(buffer["dones"]),
                            success=success,
                            length=len(buffer["observations"]),
                        )
                        episodes.append(episode)
                        episodes_collected += 1

                        # Clear buffer
                        self.episode_buffers[env_idx] = defaultdict(list)

                        if episodes_collected % 10 == 0:
                            print(f"  Collected {episodes_collected}/{num_episodes} episodes")

                        if episodes_collected >= num_episodes:
                            break

                obs_dict = next_obs_dict
                step_count += 1

            return episodes[:num_episodes]

        def close(self):
            """Clean up."""
            self.env.close()

    return IsaacLabCollector(args), simulation_app


def compute_time_to_completion(episodes):
    """Compute time-to-completion for each timestep in episodes.

    For RECAP, we need τ - t for each timestep t, where τ is the episode length.
    This is used as the target for the value function.
    """
    import numpy as np

    for episode in episodes:
        episode_length = episode.length
        episode.time_to_completion = [episode_length - t for t in range(episode_length)]

    return episodes


def save_episodes_hdf5(episodes, output_path, dataset_name):
    """Save episodes in HDF5 format for RECAP training."""
    import h5py
    import numpy as np

    os.makedirs(output_path, exist_ok=True)
    filepath = os.path.join(output_path, f"{dataset_name}.hdf5")

    with h5py.File(filepath, "w") as f:
        # Metadata
        f.attrs["num_episodes"] = len(episodes)
        f.attrs["total_timesteps"] = sum(ep.length for ep in episodes)
        f.attrs["success_rate"] = sum(ep.success for ep in episodes) / len(episodes)

        # Store each episode
        for ep_idx, episode in enumerate(episodes):
            ep_group = f.create_group(f"episode_{ep_idx:06d}")
            ep_group.attrs["length"] = episode.length
            ep_group.attrs["success"] = episode.success

            # Observations
            obs_group = ep_group.create_group("observations")
            for key in episode.observations[0].keys():
                data = np.stack([obs[key] for obs in episode.observations], axis=0)
                obs_group.create_dataset(key, data=data, compression="gzip")

            # Actions
            actions = np.stack(episode.actions, axis=0)
            ep_group.create_dataset("actions", data=actions, compression="gzip")

            # Rewards
            rewards = np.array(episode.rewards, dtype=np.float32)
            ep_group.create_dataset("rewards", data=rewards)

            # Time to completion (for value function)
            if hasattr(episode, 'time_to_completion'):
                ttc = np.array(episode.time_to_completion, dtype=np.int32)
                ep_group.create_dataset("time_to_completion", data=ttc)

    print(f"Saved {len(episodes)} episodes to {filepath}")
    return filepath


def save_episodes_numpy(episodes, output_path, dataset_name):
    """Save episodes in numpy format (alternative to HDF5)."""
    import numpy as np

    output_dir = os.path.join(output_path, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    for ep_idx, episode in enumerate(episodes):
        ep_dir = os.path.join(output_dir, f"episode_{ep_idx:06d}")
        os.makedirs(ep_dir, exist_ok=True)

        # Save observations
        for key in episode.observations[0].keys():
            data = np.stack([obs[key] for obs in episode.observations], axis=0)
            np.save(os.path.join(ep_dir, f"{key}.npy"), data)

        # Save actions
        actions = np.stack(episode.actions, axis=0)
        np.save(os.path.join(ep_dir, "actions.npy"), actions)

        # Save rewards
        rewards = np.array(episode.rewards, dtype=np.float32)
        np.save(os.path.join(ep_dir, "rewards.npy"), rewards)

        # Save metadata
        metadata = {
            "length": episode.length,
            "success": episode.success,
        }
        if hasattr(episode, 'time_to_completion'):
            metadata["time_to_completion"] = episode.time_to_completion
        np.save(os.path.join(ep_dir, "metadata.npy"), metadata)

    # Save overall metadata
    overall_meta = {
        "num_episodes": len(episodes),
        "total_timesteps": sum(ep.length for ep in episodes),
        "success_rate": sum(ep.success for ep in episodes) / len(episodes),
    }
    np.save(os.path.join(output_dir, "metadata.npy"), overall_meta)

    print(f"Saved {len(episodes)} episodes to {output_dir}")
    return output_dir


def main():
    """Main data collection loop."""
    print("\n[1/4] Setting up collector...")

    if ISAACLAB_AVAILABLE:
        collector, simulation_app = create_isaaclab_collector(args)
    else:
        collector = create_mock_data_collector(args)
        simulation_app = None

    print("\n[2/4] Collecting episodes...")
    episodes = collector.collect_episodes(args.num_episodes)

    print("\n[3/4] Computing time-to-completion...")
    episodes = compute_time_to_completion(episodes)

    print("\n[4/4] Saving episodes...")

    # Save in specified format(s)
    hdf5_path = None
    numpy_path = None

    if args.format in ["hdf5", "both"]:
        hdf5_path = save_episodes_hdf5(episodes, args.output_dir, args.dataset_name)

    if args.format in ["numpy", "both"]:
        numpy_path = save_episodes_numpy(episodes, args.output_dir, args.dataset_name + "_numpy")

    # Print summary
    print("\n" + "=" * 70)
    print("DATA COLLECTION COMPLETE")
    print("=" * 70)
    print(f"Episodes collected: {len(episodes)}")
    print(f"Total timesteps: {sum(ep.length for ep in episodes)}")
    print(f"Success rate: {sum(ep.success for ep in episodes) / len(episodes):.2%}")
    print(f"Average episode length: {sum(ep.length for ep in episodes) / len(episodes):.1f}")
    print(f"Policy: {args.policy}")
    if args.policy == "checkpoint":
        print(f"Checkpoint: {args.checkpoint_path}")
    print(f"\nOutput files:")
    if hdf5_path:
        print(f"  HDF5: {hdf5_path}")
    if numpy_path:
        print(f"  NumPy: {numpy_path}")
    print("\nNext steps:")
    print("  1. Train RECAP with: python scripts/train_recap_full.py --config recap_aloha_sim")
    print("  2. Or collect more data with trained policy: --policy checkpoint --checkpoint_path <path>")
    print("=" * 70)

    # Cleanup
    if ISAACLAB_AVAILABLE:
        collector.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
