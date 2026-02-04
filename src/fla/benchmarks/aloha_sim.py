from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AlohaBenchmarkSpec:
    name: str
    prompt: str
    description: str
    baseline: float | None = None
    pi06_paper: float | None = None


BENCHMARKS: dict[str, AlohaBenchmarkSpec] = {
    "gym_aloha/AlohaTransferCube-v0": AlohaBenchmarkSpec(
        name="Transfer Cube",
        prompt="Transfer the cube to the target location",
        description="Pick up cube and place in target zone",
        baseline=60.0,
        pi06_paper=85.0,
    ),
    "gym_aloha/AlohaInsertion-v0": AlohaBenchmarkSpec(
        name="Insertion",
        prompt="Insert the peg into the socket",
        description="Insert peg into socket with precision",
        baseline=50.0,
        pi06_paper=80.0,
    ),
}

OPTIONAL_TASKS: dict[str, AlohaBenchmarkSpec] = {
    "gym_aloha/AlohaTransferCubeScripted-v0": AlohaBenchmarkSpec(
        name="Transfer Cube (Scripted)",
        prompt="Transfer the cube to the target location",
        description="Transfer cube scripted variant",
    ),
    "gym_aloha/AlohaInsertionScripted-v0": AlohaBenchmarkSpec(
        name="Insertion (Scripted)",
        prompt="Insert the peg into the socket",
        description="Insertion scripted variant",
    ),
}


def create_policy(checkpoint_dir: str, config_name: str, default_prompt: str):
    """Create a policy from a checkpoint."""
    from fla.policies import policy_config as _policy_config
    from fla.training import config as _config

    train_config = _config.get_config(config_name)
    return _policy_config.create_trained_policy(
        train_config=train_config,
        checkpoint_dir=checkpoint_dir,
        default_prompt=default_prompt,
    )


def create_environment(task: str, seed: int) -> tuple[Any | None, str | None]:
    """Create the ALOHA sim environment. Returns (env, error)."""
    try:
        import gym_aloha  # noqa: F401
        import gymnasium
    except Exception as exc:  # noqa: BLE001
        return None, f"gym_aloha import failed: {exc}"

    try:
        env = gymnasium.make(task, obs_type="pixels_agent_pos")
        env.reset(seed=seed)
        return env, None
    except Exception as exc:  # noqa: BLE001
        return None, f"env creation failed for {task}: {exc}"


def convert_observation(gym_obs: dict) -> dict:
    """Convert gym observation to policy input format."""
    try:
        from openpi_client import image_tools

        img = gym_obs["pixels"]["top"]
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
        img = np.transpose(img, (2, 0, 1))
    except Exception:  # noqa: BLE001
        img = gym_obs["pixels"]["top"]
        from PIL import Image

        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize((224, 224))
        img = np.array(pil_img)
        img = np.transpose(img, (2, 0, 1))

    return {
        "state": gym_obs["agent_pos"],
        "images": {"cam_high": img},
    }


def run_episode(
    policy,
    env,
    prompt: str,
    *,
    max_steps: int = 400,
    action_horizon: int = 10,
) -> dict:
    """Run a single episode and return results."""
    obs, _ = env.reset()
    policy_obs = convert_observation(obs)

    max_reward = 0.0
    total_reward = 0.0
    step = 0
    action_queue = []
    start_time = time.time()

    while step < max_steps:
        if not action_queue:
            policy_input = {
                **policy_obs,
                "prompt": prompt,
            }
            result = policy.infer(policy_input)
            actions = result["actions"]
            action_queue = list(actions[:action_horizon])

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
        "max_reward": float(max_reward),
        "total_reward": float(total_reward),
        "steps": int(step),
        "time_sec": float(elapsed),
    }


def evaluate_task(
    policy,
    task: str,
    *,
    prompt: str,
    num_episodes: int,
    seed: int = 0,
    action_horizon: int = 10,
    max_steps: int = 400,
) -> dict:
    """Evaluate a single ALOHA sim task."""
    env, error = create_environment(task, seed)
    if env is None:
        return {
            "task": task,
            "status": "FAILED",
            "error": error,
        }

    results = []
    for ep in range(num_episodes):
        np.random.seed(seed + ep)
        try:
            result = run_episode(
                policy,
                env,
                prompt,
                max_steps=max_steps,
                action_horizon=action_horizon,
            )
            results.append(result)
        except Exception as exc:  # noqa: BLE001
            logger.error("Episode %s failed: %s", ep + 1, exc)
            results.append({"success": False, "max_reward": 0.0, "error": str(exc)})

    env.close()

    successes = [r.get("success", False) for r in results]
    rewards = [r.get("max_reward", 0.0) for r in results]

    success_rate = float(np.mean(successes) * 100.0) if results else 0.0
    avg_reward = float(np.mean(rewards)) if results else 0.0
    std_reward = float(np.std(rewards)) if results else 0.0

    return {
        "task": task,
        "status": "OK",
        "num_episodes": num_episodes,
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "std_reward": std_reward,
        "episodes": results,
    }


def run_suite(
    *,
    checkpoint_dir: str,
    config_name: str,
    tasks: list[str],
    num_episodes: int,
    seed: int = 0,
    action_horizon: int = 10,
    max_steps: int = 400,
) -> dict:
    """Run a suite of ALOHA sim evaluations."""
    logger.info("Loading policy from checkpoint: %s", checkpoint_dir)
    policy = create_policy(
        checkpoint_dir=checkpoint_dir,
        config_name=config_name,
        default_prompt="Complete the manipulation task",
    )

    results = {}
    for task in tasks:
        spec = BENCHMARKS.get(task) or OPTIONAL_TASKS.get(task)
        prompt = spec.prompt if spec else "Complete the task"

        logger.info("Evaluating task: %s", task)
        task_result = evaluate_task(
            policy,
            task,
            prompt=prompt,
            num_episodes=num_episodes,
            seed=seed,
            action_horizon=action_horizon,
            max_steps=max_steps,
        )

        if spec is not None:
            task_result["benchmark"] = {
                "name": spec.name,
                "description": spec.description,
                "baseline": spec.baseline,
                "pi06_paper": spec.pi06_paper,
                "prompt": spec.prompt,
            }

        results[task] = task_result

    success_rates = [
        r["success_rate"] for r in results.values() if r.get("status") == "OK"
    ]
    avg_success = float(np.mean(success_rates)) if success_rates else 0.0

    return {
        "suite": "aloha_sim",
        "checkpoint_dir": checkpoint_dir,
        "config": config_name,
        "num_tasks": len(tasks),
        "num_episodes": num_episodes,
        "avg_success_rate": avg_success,
        "tasks": results,
    }
