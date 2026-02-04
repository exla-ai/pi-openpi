from __future__ import annotations

import collections
import logging
from pathlib import Path

import numpy as np

from fla.policies import policy_config as _policy_config
from fla.training import config as _config

logger = logging.getLogger(__name__)

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]


def _require_libero():
    try:
        from libero.libero import benchmark  # noqa: F401
        from libero.libero import get_libero_path  # noqa: F401
        from libero.libero.envs import OffScreenRenderEnv  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "LIBERO is not installed. Install dependencies with "
            "`uv sync --group libero` or follow examples/libero/README.md."
        ) from exc


def _max_steps_for_suite(task_suite_name: str) -> int:
    match task_suite_name:
        case "libero_spatial":
            return 220
        case "libero_object":
            return 280
        case "libero_goal":
            return 300
        case "libero_10":
            return 520
        case "libero_90":
            return 400
    raise ValueError(f"Unknown task suite: {task_suite_name}")


def _get_libero_env(task, resolution: int, seed: int):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_description = task.language
    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    from scipy.spatial.transform import Rotation

    return Rotation.from_quat(quat).as_rotvec()


def _resize_image(img: np.ndarray, size: int) -> np.ndarray:
    try:
        from openpi_client import image_tools

        return image_tools.convert_to_uint8(image_tools.resize_with_pad(img, size, size))
    except Exception:
        from PIL import Image

        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize((size, size))
        return np.asarray(pil_img)


def _prepare_policy_input(obs: dict, task_description: str, resize_size: int) -> dict:
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = _resize_image(img, resize_size)
    wrist_img = _resize_image(wrist_img, resize_size)

    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            _quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    )

    return {
        "observation/image": img,
        "observation/wrist_image": wrist_img,
        "observation/state": state,
        "prompt": str(task_description),
    }


def run_suite(
    *,
    checkpoint_dir: str,
    config_name: str,
    task_suite_name: str = "libero_spatial",
    num_trials_per_task: int = 50,
    resize_size: int = 224,
    replan_steps: int = 5,
    num_steps_wait: int = 10,
    seed: int = 7,
    video_out_path: str | None = None,
    max_steps: int | None = None,
) -> dict:
    _require_libero()

    from libero.libero import benchmark

    train_config = _config.get_config(config_name)
    policy = _policy_config.create_trained_policy(
        train_config=train_config,
        checkpoint_dir=checkpoint_dir,
        default_prompt="Complete the task",
    )

    benchmark_dict = benchmark.get_benchmark_dict()
    if task_suite_name not in benchmark_dict:
        raise ValueError(f"Unknown LIBERO task suite: {task_suite_name}")

    task_suite = benchmark_dict[task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks

    max_steps = max_steps or _max_steps_for_suite(task_suite_name)

    save_videos = video_out_path is not None
    if save_videos:
        Path(video_out_path).mkdir(parents=True, exist_ok=True)

    total_episodes = 0
    total_successes = 0
    task_results = []

    for task_id in range(num_tasks_in_suite):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, resize_size, seed)

        task_episodes = 0
        task_successes = 0

        for episode_idx in range(num_trials_per_task):
            env.reset()
            action_plan = collections.deque()
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            done = False
            replay_images = []

            while t < max_steps + num_steps_wait:
                if t < num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                if save_videos:
                    replay_images.append(np.asarray(obs["agentview_image"][::-1, ::-1]))

                if not action_plan:
                    policy_input = _prepare_policy_input(obs, task_description, resize_size)
                    action_chunk = policy.infer(policy_input)["actions"]
                    action_plan.extend(action_chunk[:replan_steps])

                action = action_plan.popleft()
                obs, reward, done, info = env.step(action.tolist())
                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1

            task_episodes += 1
            total_episodes += 1

            if save_videos:
                import imageio

                suffix = "success" if done else "failure"
                task_segment = str(task_description).replace(" ", "_")
                imageio.mimwrite(
                    Path(video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )

        env.close()

        task_results.append(
            {
                "task_id": task_id,
                "task_description": str(task_description),
                "episodes": task_episodes,
                "successes": task_successes,
                "success_rate": float(task_successes) / float(task_episodes) if task_episodes else 0.0,
            }
        )

    overall_success = float(total_successes) / float(total_episodes) if total_episodes else 0.0

    return {
        "suite": "libero",
        "task_suite": task_suite_name,
        "num_tasks": num_tasks_in_suite,
        "num_trials_per_task": num_trials_per_task,
        "overall_success_rate": overall_success,
        "tasks": task_results,
    }
