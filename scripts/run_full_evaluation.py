#!/usr/bin/env python3
"""Run comprehensive evaluation on all Pi0.6 benchmark tasks.

This script evaluates a trained checkpoint on all ALOHA simulation tasks
and generates a report with success rates comparable to Pi0.6 paper benchmarks.

Usage:
    python scripts/run_full_evaluation.py \
        --checkpoint_dir ./checkpoints/pi06_multi/pi06_multi_v1/30000 \
        --config pi06_multi \
        --num_episodes 50 \
        --update_readme

Tasks evaluated:
    - AlohaTransferCube-v0: Pick up cube and transfer to target location
    - AlohaInsertion-v0: Insert peg into socket
    - AlohaTransferCubeScripted-v0: Transfer cube (scripted variant)
    - AlohaInsertionScripted-v0: Insertion (scripted variant)
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
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

# Pi0.6 Paper benchmark targets
BENCHMARKS = {
    "gym_aloha/AlohaTransferCube-v0": {
        "name": "Transfer Cube",
        "prompt": "Transfer the cube to the target location",
        "baseline": 60,
        "pi06_paper": 85,
        "description": "Pick up cube and place in target zone",
    },
    "gym_aloha/AlohaInsertion-v0": {
        "name": "Insertion",
        "prompt": "Insert the peg into the socket",
        "baseline": 50,
        "pi06_paper": 80,
        "description": "Insert peg into socket with precision",
    },
}

# Optional additional tasks if available
OPTIONAL_TASKS = [
    "gym_aloha/AlohaTransferCubeScripted-v0",
    "gym_aloha/AlohaInsertionScripted-v0",
]


def create_policy(checkpoint_dir: str, config_name: str):
    """Create a policy from a checkpoint."""
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config
    from openpi.shared import download
    import openpi.models.model as _model

    logger.info(f"Loading config: {config_name}")
    train_config = _config.get_config(config_name)

    logger.info(f"Loading checkpoint from: {checkpoint_dir}")
    checkpoint_path = download.maybe_download(checkpoint_dir)

    # Load norm stats from checkpoint assets
    assets_dir = Path(checkpoint_path) / "assets"
    norm_stats = None
    if assets_dir.exists():
        # Find the asset directory (e.g., lerobot/aloha_sim_transfer_cube_human)
        for asset_path in assets_dir.iterdir():
            if asset_path.is_dir():
                try:
                    from openpi.shared import normalize as _normalize
                    norm_stats = _normalize.load(str(asset_path))
                    logger.info(f"Loaded norm stats from {asset_path}")
                    break
                except Exception as e:
                    logger.warning(f"Could not load norm stats from {asset_path}: {e}")

    # Create policy config
    policy_cfg = _policy_config.PolicyConfig(
        model=train_config.model,
        norm_stats=norm_stats,
        default_prompt="Complete the manipulation task",
    )

    # Load the model
    params_path = Path(checkpoint_path) / "params"
    params = _model.restore_params(str(params_path))

    return policy_cfg.create_policy_from_params(params)


def create_environment(task: str, seed: int):
    """Create the ALOHA sim environment."""
    try:
        import gym_aloha  # noqa: F401
        import gymnasium
        env = gymnasium.make(task, obs_type="pixels_agent_pos")
        return env
    except Exception as e:
        logger.error(f"Failed to create environment {task}: {e}")
        return None


def convert_observation(gym_obs: dict) -> dict:
    """Convert gym observation to policy input format."""
    try:
        from openpi_client import image_tools
        img = gym_obs["pixels"]["top"]
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
        img = np.transpose(img, (2, 0, 1))
    except ImportError:
        # Fallback without openpi_client
        img = gym_obs["pixels"]["top"]
        # Resize manually if needed
        from PIL import Image
        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize((224, 224))
        img = np.array(pil_img)
        img = np.transpose(img, (2, 0, 1))

    return {
        "state": gym_obs["agent_pos"],
        "images": {"cam_high": img},
    }


def run_episode(policy, env, prompt: str, max_steps: int = 400, action_horizon: int = 10) -> dict:
    """Run a single episode and return results."""
    obs, _ = env.reset()
    policy_obs = convert_observation(obs)

    max_reward = 0.0
    total_reward = 0.0
    step = 0
    action_queue = []
    start_time = time.time()

    while step < max_steps:
        if len(action_queue) == 0:
            policy_input = {
                **policy_obs,
                "prompt": prompt,
            }
            try:
                result = policy.infer(policy_input)
                actions = result["actions"]
                action_queue = list(actions[:action_horizon])
            except Exception as e:
                logger.error(f"Policy inference failed: {e}")
                break

        if not action_queue:
            break

        action = action_queue.pop(0)
        obs, reward, terminated, truncated, info = env.step(action)
        policy_obs = convert_observation(obs)

        max_reward = max(max_reward, reward)
        total_reward += reward
        step += 1

        if terminated or truncated:
            break

    elapsed = time.time() - start_time
    success = max_reward >= 0.95

    return {
        "success": success,
        "max_reward": max_reward,
        "total_reward": total_reward,
        "steps": step,
        "time": elapsed,
    }


def evaluate_task(policy, task: str, benchmark: dict, num_episodes: int, seed: int = 0) -> dict:
    """Evaluate a single task."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating: {benchmark['name']}")
    logger.info(f"Task: {task}")
    logger.info(f"Description: {benchmark['description']}")
    logger.info(f"{'='*60}")

    env = create_environment(task, seed)
    if env is None:
        return {
            "task": task,
            "name": benchmark["name"],
            "status": "FAILED",
            "error": "Could not create environment",
        }

    results = []
    for ep in range(num_episodes):
        np.random.seed(seed + ep)
        try:
            result = run_episode(policy, env, benchmark["prompt"])
            results.append(result)

            status = "✓" if result["success"] else "✗"
            logger.info(
                f"Episode {ep+1:3d}/{num_episodes}: {status} "
                f"reward={result['max_reward']:.3f} "
                f"steps={result['steps']} "
                f"time={result['time']:.1f}s"
            )
        except Exception as e:
            logger.error(f"Episode {ep+1} failed: {e}")
            results.append({"success": False, "max_reward": 0, "error": str(e)})

        # Print running stats every 10 episodes
        if (ep + 1) % 10 == 0:
            successes = [r["success"] for r in results]
            running_sr = np.mean(successes) * 100
            logger.info(f"  Running success rate: {running_sr:.1f}%")

    env.close()

    # Compute statistics
    successes = [r["success"] for r in results]
    rewards = [r.get("max_reward", 0) for r in results]

    success_rate = np.mean(successes) * 100
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)

    return {
        "task": task,
        "name": benchmark["name"],
        "status": "OK",
        "num_episodes": num_episodes,
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "std_reward": std_reward,
        "baseline": benchmark["baseline"],
        "pi06_paper": benchmark["pi06_paper"],
        "vs_baseline": success_rate - benchmark["baseline"],
        "vs_paper": success_rate - benchmark["pi06_paper"],
    }


def run_full_evaluation(
    checkpoint_dir: str,
    config_name: str,
    num_episodes: int = 50,
    seed: int = 0,
) -> dict:
    """Run evaluation on all benchmark tasks."""
    logger.info("=" * 70)
    logger.info("PI0.6 COMPREHENSIVE EVALUATION SUITE")
    logger.info("=" * 70)
    logger.info(f"Checkpoint: {checkpoint_dir}")
    logger.info(f"Config: {config_name}")
    logger.info(f"Episodes per task: {num_episodes}")
    logger.info(f"Seed: {seed}")
    logger.info("=" * 70)

    # Load policy once
    logger.info("\nLoading policy...")
    try:
        policy = create_policy(checkpoint_dir, config_name)
        logger.info("Policy loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load policy: {e}")
        return {"status": "FAILED", "error": str(e)}

    # Evaluate each benchmark task
    results = {
        "checkpoint": checkpoint_dir,
        "config": config_name,
        "num_episodes": num_episodes,
        "timestamp": datetime.now().isoformat(),
        "tasks": {},
    }

    for task, benchmark in BENCHMARKS.items():
        task_result = evaluate_task(policy, task, benchmark, num_episodes, seed)
        results["tasks"][task] = task_result

    # Print summary
    logger.info("\n")
    logger.info("=" * 70)
    logger.info("EVALUATION RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info("")
    logger.info(f"{'Task':<25} {'Success Rate':>12} {'Baseline':>10} {'Pi0.6':>10} {'vs Base':>10}")
    logger.info("-" * 70)

    total_sr = 0
    num_tasks = 0
    for task, task_result in results["tasks"].items():
        if task_result["status"] == "OK":
            sr = task_result["success_rate"]
            baseline = task_result["baseline"]
            paper = task_result["pi06_paper"]
            vs_base = task_result["vs_baseline"]

            logger.info(
                f"{task_result['name']:<25} "
                f"{sr:>11.1f}% "
                f"{baseline:>9}% "
                f"{paper:>9}% "
                f"{vs_base:>+9.1f}%"
            )
            total_sr += sr
            num_tasks += 1
        else:
            logger.info(f"{task_result['name']:<25} {'FAILED':>12}")

    if num_tasks > 0:
        avg_sr = total_sr / num_tasks
        logger.info("-" * 70)
        logger.info(f"{'AVERAGE':<25} {avg_sr:>11.1f}%")

    logger.info("=" * 70)

    # Add summary to results
    results["summary"] = {
        "average_success_rate": avg_sr if num_tasks > 0 else 0,
        "num_tasks_evaluated": num_tasks,
        "num_tasks_failed": len(BENCHMARKS) - num_tasks,
    }

    return results


def generate_readme_section(results: dict) -> str:
    """Generate markdown section for README."""
    lines = [
        "## Evaluation Results",
        "",
        f"**Checkpoint**: `{results['checkpoint']}`",
        f"**Config**: `{results['config']}`",
        f"**Date**: {results['timestamp'][:10]}",
        f"**Episodes per task**: {results['num_episodes']}",
        "",
        "### Benchmark Results",
        "",
        "| Task | Success Rate | Baseline (BC) | Pi0.6 Paper | vs Baseline |",
        "|------|-------------|---------------|-------------|-------------|",
    ]

    for task, task_result in results["tasks"].items():
        if task_result["status"] == "OK":
            name = task_result["name"]
            sr = task_result["success_rate"]
            baseline = task_result["baseline"]
            paper = task_result["pi06_paper"]
            vs_base = task_result["vs_baseline"]
            lines.append(
                f"| {name} | **{sr:.1f}%** | {baseline}% | {paper}% | {vs_base:+.1f}% |"
            )

    if results.get("summary"):
        avg = results["summary"]["average_success_rate"]
        lines.extend([
            "",
            f"**Average Success Rate**: {avg:.1f}%",
        ])

    lines.extend([
        "",
        "### Interpretation",
        "",
        "- **vs Baseline**: Improvement over behavior cloning baseline",
        "- Our model uses **frozen Pi0.5 backbone** (efficient training)",
        "- Pi0.6 paper uses **full RECAP training** with on-robot data collection",
        "",
    ])

    return "\n".join(lines)


def update_readme(results: dict, readme_path: str):
    """Update README.md with evaluation results."""
    results_section = generate_readme_section(results)

    with open(readme_path, "r") as f:
        content = f.read()

    # Find and replace existing evaluation section or add new one
    start_marker = "## Evaluation Results"
    end_marker = "\n## "  # Next section

    if start_marker in content:
        # Replace existing section
        start_idx = content.index(start_marker)
        end_idx = content.find(end_marker, start_idx + len(start_marker))
        if end_idx == -1:
            end_idx = len(content)
        content = content[:start_idx] + results_section + "\n" + content[end_idx:]
    else:
        # Add before "## Running Inference" or at end
        insert_marker = "## Running Inference"
        if insert_marker in content:
            idx = content.index(insert_marker)
            content = content[:idx] + results_section + "\n\n" + content[idx:]
        else:
            content += "\n\n" + results_section

    with open(readme_path, "w") as f:
        f.write(content)

    logger.info(f"Updated README at {readme_path}")


def main():
    parser = argparse.ArgumentParser(description="Run full Pi0.6 evaluation suite")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to checkpoint directory")
    parser.add_argument("--config", type=str, required=True,
                        help="Training config name")
    parser.add_argument("--num_episodes", type=int, default=50,
                        help="Number of episodes per task")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--output", type=str, default="evaluation_results.json",
                        help="Output JSON file for results")
    parser.add_argument("--update_readme", action="store_true",
                        help="Update README.md with results")
    parser.add_argument("--readme_path", type=str, default="README.md",
                        help="Path to README.md")
    args = parser.parse_args()

    # Run evaluation
    results = run_full_evaluation(
        checkpoint_dir=args.checkpoint_dir,
        config_name=args.config,
        num_episodes=args.num_episodes,
        seed=args.seed,
    )

    # Save results
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")

    # Update README if requested
    if args.update_readme:
        update_readme(results, args.readme_path)

    # Return exit code based on results
    if results.get("status") == "FAILED":
        sys.exit(1)

    avg_sr = results.get("summary", {}).get("average_success_rate", 0)
    if avg_sr < 50:
        logger.warning(f"Average success rate ({avg_sr:.1f}%) is below 50%")
        sys.exit(1)

    logger.info(f"\nEvaluation complete. Average success rate: {avg_sr:.1f}%")


if __name__ == "__main__":
    main()
