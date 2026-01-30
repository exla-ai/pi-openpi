#!/usr/bin/env python3
"""Evaluate a trained policy on ALOHA simulation tasks.

This script evaluates a policy checkpoint on ALOHA sim tasks and reports
success rates comparable to the Pi0.6 paper benchmarks.

Usage:
    # Evaluate on transfer cube task
    python scripts/evaluate_aloha_sim.py \
        --checkpoint_dir ./checkpoints/pi06_multi/pi06_multi_v1/20000 \
        --config pi06_multi \
        --task gym_aloha/AlohaTransferCube-v0 \
        --num_episodes 50

    # Evaluate on insertion task
    python scripts/evaluate_aloha_sim.py \
        --checkpoint_dir ./checkpoints/pi06_multi/pi06_multi_v1/20000 \
        --config pi06_multi \
        --task gym_aloha/AlohaInsertion-v0 \
        --num_episodes 50
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def create_policy(checkpoint_dir: str, config_name: str, default_prompt: str):
    """Create a policy from a checkpoint."""
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    # Get the training config
    train_config = _config.get_config(config_name)

    # Create policy config
    policy_cfg = _policy_config.PolicyConfig(
        model=train_config.model,
        norm_stats=None,  # Will be loaded from checkpoint
        default_prompt=default_prompt,
    )

    # Load checkpoint
    from openpi.shared import download
    checkpoint_path = download.maybe_download(checkpoint_dir)

    return policy_cfg.create_policy(checkpoint_path)


def create_environment(task: str, seed: int):
    """Create the ALOHA sim environment."""
    import gym_aloha  # noqa: F401
    import gymnasium

    env = gymnasium.make(task, obs_type="pixels_agent_pos")
    return env


def convert_observation(gym_obs: dict) -> dict:
    """Convert gym observation to policy input format."""
    from openpi_client import image_tools

    img = gym_obs["pixels"]["top"]
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
    # Convert axis order from [H, W, C] --> [C, H, W]
    img = np.transpose(img, (2, 0, 1))

    return {
        "state": gym_obs["agent_pos"],
        "images": {"cam_high": img},
    }


def run_episode(policy, env, max_steps: int = 400, action_horizon: int = 10) -> tuple[bool, float]:
    """Run a single episode and return (success, max_reward)."""
    obs, _ = env.reset()
    policy_obs = convert_observation(obs)

    max_reward = 0.0
    step = 0
    action_queue = []

    while step < max_steps:
        # Get action from policy if queue is empty
        if len(action_queue) == 0:
            # Add prompt to observation
            policy_input = {
                **policy_obs,
                "prompt": "Transfer the cube" if "TransferCube" in str(env.spec.id) else "Insert the peg",
            }

            # Get action chunk from policy
            result = policy.infer(policy_input)
            actions = result["actions"]  # Shape: [horizon, action_dim]
            action_queue = list(actions[:action_horizon])

        # Execute action
        action = action_queue.pop(0)
        obs, reward, terminated, truncated, info = env.step(action)
        policy_obs = convert_observation(obs)

        max_reward = max(max_reward, reward)
        step += 1

        if terminated or truncated:
            break

    # Success is typically reward >= 0.95 for ALOHA sim
    success = max_reward >= 0.95
    return success, max_reward


def evaluate(
    checkpoint_dir: str,
    config_name: str,
    task: str,
    num_episodes: int,
    seed: int = 0,
    action_horizon: int = 10,
):
    """Run full evaluation."""
    logger.info("=" * 60)
    logger.info("ALOHA Sim Evaluation")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {checkpoint_dir}")
    logger.info(f"Config: {config_name}")
    logger.info(f"Task: {task}")
    logger.info(f"Episodes: {num_episodes}")
    logger.info("=" * 60)

    # Determine prompt based on task
    if "TransferCube" in task:
        default_prompt = "Transfer the cube"
    elif "Insertion" in task:
        default_prompt = "Insert the peg"
    else:
        default_prompt = "Complete the task"

    logger.info(f"Default prompt: {default_prompt}")
    logger.info("Loading policy...")

    policy = create_policy(checkpoint_dir, config_name, default_prompt)

    logger.info("Creating environment...")
    env = create_environment(task, seed)

    # Run evaluation
    successes = []
    rewards = []

    logger.info(f"\nRunning {num_episodes} episodes...")

    for ep in range(num_episodes):
        np.random.seed(seed + ep)

        start_time = time.time()
        success, reward = run_episode(policy, env, action_horizon=action_horizon)
        elapsed = time.time() - start_time

        successes.append(success)
        rewards.append(reward)

        status = "✓" if success else "✗"
        logger.info(f"Episode {ep+1:3d}/{num_episodes}: {status} reward={reward:.3f} time={elapsed:.1f}s")

        # Print running statistics every 10 episodes
        if (ep + 1) % 10 == 0:
            running_sr = np.mean(successes) * 100
            running_reward = np.mean(rewards)
            logger.info(f"  Running: SR={running_sr:.1f}% Avg Reward={running_reward:.3f}")

    # Final results
    success_rate = np.mean(successes) * 100
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)

    logger.info("")
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"Task: {task}")
    logger.info(f"Episodes: {num_episodes}")
    logger.info(f"Success Rate: {success_rate:.1f}%")
    logger.info(f"Average Reward: {avg_reward:.3f} ± {std_reward:.3f}")
    logger.info("=" * 60)

    # Compare to baselines
    logger.info("")
    logger.info("Comparison to Pi0.6 Paper Benchmarks:")
    logger.info("-" * 40)
    if "TransferCube" in task:
        logger.info(f"Baseline (BC):     60%")
        logger.info(f"Pi0.6 (RECAP):     85%")
        logger.info(f"This model:        {success_rate:.1f}%")
    elif "Insertion" in task:
        logger.info(f"Baseline (BC):     ~50%")
        logger.info(f"Pi0.6 (RECAP):     ~80%")
        logger.info(f"This model:        {success_rate:.1f}%")
    logger.info("-" * 40)

    env.close()

    return {
        "task": task,
        "num_episodes": num_episodes,
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "std_reward": std_reward,
        "successes": successes,
        "rewards": rewards,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate ALOHA sim policy")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to checkpoint directory")
    parser.add_argument("--config", type=str, required=True,
                        help="Training config name (e.g., pi06_multi)")
    parser.add_argument("--task", type=str, default="gym_aloha/AlohaTransferCube-v0",
                        choices=[
                            "gym_aloha/AlohaTransferCube-v0",
                            "gym_aloha/AlohaInsertion-v0",
                        ],
                        help="ALOHA sim task to evaluate")
    parser.add_argument("--num_episodes", type=int, default=50,
                        help="Number of episodes to run")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--action_horizon", type=int, default=10,
                        help="Number of actions to execute per inference")
    args = parser.parse_args()

    evaluate(
        checkpoint_dir=args.checkpoint_dir,
        config_name=args.config,
        task=args.task,
        num_episodes=args.num_episodes,
        seed=args.seed,
        action_horizon=args.action_horizon,
    )


if __name__ == "__main__":
    main()
