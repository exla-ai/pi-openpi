from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

from fla.benchmarks import aloha_sim


def _default_tasks_for_suite(suite: str) -> list[str]:
    if suite == "aloha_sim":
        return list(aloha_sim.BENCHMARKS.keys())
    if suite == "aloha_sim_full":
        return list(aloha_sim.BENCHMARKS.keys()) + list(aloha_sim.OPTIONAL_TASKS.keys())
    raise ValueError(f"Unknown suite: {suite}")


def _parse_tasks(args) -> list[str]:
    if args.task:
        return args.task
    return _default_tasks_for_suite(args.suite)


def _write_output(results: dict, output: str | None) -> None:
    if not output:
        return
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))


def _parse_repo_id_to_prompt(items: Iterable[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in items:
        if ":" not in item:
            raise ValueError(f"Invalid --repo-id-to-prompt entry '{item}'. Use <repo_id>:<prompt> format.")
        repo_id, prompt = item.split(":", 1)
        mapping[repo_id.strip()] = prompt.strip()
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="FLA benchmark evaluation CLI")
    parser.add_argument(
        "--suite",
        type=str,
        default="aloha_sim",
        choices=[
            "aloha_sim",
            "aloha_sim_full",
            "libero",
            "dataset",
            "droid_manual",
            "full",
        ],
        help="Benchmark suite to run",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=False,
        help="Path to checkpoint directory",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        help="Training config name (e.g., pi06_multi, pi0_frozen_backbone)",
    )
    parser.add_argument(
        "--recipe",
        type=str,
        default=None,
        help="Optional recipe name (only used for dataset eval)",
    )
    parser.add_argument(
        "--task",
        type=str,
        action="append",
        help="Override tasks to evaluate (can be repeated)",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=50,
        help="Number of episodes per ALOHA task",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=10,
        help="Number of actions to execute per inference (ALOHA)",
    )
    parser.add_argument(
        "--model-action-dim",
        type=int,
        default=None,
        help="Override model action dim when using --recipe (dataset eval)",
    )
    parser.add_argument(
        "--model-action-horizon",
        type=int,
        default=None,
        help="Override model action horizon when using --recipe (dataset eval)",
    )
    parser.add_argument(
        "--use-delta-joint-actions",
        action="store_true",
        help="Convert actions to deltas when using --recipe (dataset eval)",
    )
    parser.add_argument(
        "--default-prompt",
        type=str,
        default=None,
        help="Default prompt for recipe-based dataset eval",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="pi05",
        choices=["pi0", "pi05"],
        help="Base model family (recipe-based dataset eval)",
    )
    parser.add_argument(
        "--init-from",
        type=str,
        default="base",
        choices=["base", "scratch"],
        help="Initialization mode (recipe-based dataset eval)",
    )
    parser.add_argument(
        "--paligemma-variant",
        type=str,
        default=None,
        help="Override PaliGemma variant (recipe-based dataset eval)",
    )
    parser.add_argument(
        "--action-expert-variant",
        type=str,
        default=None,
        help="Override action expert variant (recipe-based dataset eval)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=400,
        help="Max steps per episode (ALOHA/LIBERO override)",
    )
    parser.add_argument(
        "--libero-suite",
        type=str,
        default="libero_spatial",
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
        help="LIBERO task suite",
    )
    parser.add_argument(
        "--libero-num-trials",
        type=int,
        default=50,
        help="Number of rollouts per LIBERO task",
    )
    parser.add_argument(
        "--libero-replan-steps",
        type=int,
        default=5,
        help="Replan steps for LIBERO eval",
    )
    parser.add_argument(
        "--libero-resize",
        type=int,
        default=224,
        help="Resize images for LIBERO eval",
    )
    parser.add_argument(
        "--libero-num-steps-wait",
        type=int,
        default=10,
        help="Initial wait steps for LIBERO eval",
    )
    parser.add_argument(
        "--libero-video-out",
        type=str,
        default=None,
        help="Optional directory to save LIBERO rollout videos",
    )
    parser.add_argument(
        "--repo-ids",
        nargs="+",
        help="LeRobot dataset repo IDs (for dataset eval)",
    )
    parser.add_argument(
        "--repo-id-to-prompt",
        action="append",
        default=[],
        help="Mapping in the form <repo_id>:<prompt> (repeatable)",
    )
    parser.add_argument(
        "--prompt-from-task",
        action="store_true",
        help="Use prompt stored in the dataset task field (dataset eval)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1024,
        help="Max samples for dataset eval",
    )
    parser.add_argument(
        "--droid-left-camera-id",
        type=str,
        default="<your_camera_id>",
        help="DROID left camera id (manual eval)",
    )
    parser.add_argument(
        "--droid-right-camera-id",
        type=str,
        default="<your_camera_id>",
        help="DROID right camera id (manual eval)",
    )
    parser.add_argument(
        "--droid-wrist-camera-id",
        type=str,
        default="<your_camera_id>",
        help="DROID wrist camera id (manual eval)",
    )
    parser.add_argument(
        "--droid-external-camera",
        type=str,
        default="left",
        choices=["left", "right"],
        help="DROID external camera used for policy input",
    )
    parser.add_argument(
        "--droid-remote-host",
        type=str,
        default="0.0.0.0",
        help="DROID policy server host",
    )
    parser.add_argument(
        "--droid-remote-port",
        type=int,
        default=8000,
        help="DROID policy server port",
    )
    parser.add_argument(
        "--droid-max-timesteps",
        type=int,
        default=600,
        help="Max timesteps per DROID rollout",
    )
    parser.add_argument(
        "--droid-open-loop-horizon",
        type=int,
        default=8,
        help="Open-loop horizon for DROID rollouts",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON output path",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    repo_id_to_prompt = _parse_repo_id_to_prompt(args.repo_id_to_prompt)
    default_prompt = args.default_prompt
    if args.recipe and not repo_id_to_prompt and not args.prompt_from_task and default_prompt is None:
        default_prompt = "Complete the task"

    if args.suite in ("aloha_sim", "aloha_sim_full"):
        if not args.checkpoint_dir or not args.config:
            raise SystemExit("--checkpoint-dir and --config are required for ALOHA evaluation.")
        tasks = _parse_tasks(args)
        results = aloha_sim.run_suite(
            checkpoint_dir=args.checkpoint_dir,
            config_name=args.config,
            tasks=tasks,
            num_episodes=args.num_episodes,
            seed=args.seed,
            action_horizon=args.action_horizon,
            max_steps=args.max_steps,
        )
        _write_output(results, args.output)
        avg_sr = results.get("avg_success_rate", 0.0)
        logging.info("Evaluation complete. Avg success rate: %.1f%%", avg_sr)
        return

    if args.suite == "libero":
        if not args.checkpoint_dir or not args.config:
            raise SystemExit("--checkpoint-dir and --config are required for LIBERO evaluation.")
        from fla.benchmarks import libero

        results = libero.run_suite(
            checkpoint_dir=args.checkpoint_dir,
            config_name=args.config,
            task_suite_name=args.libero_suite,
            num_trials_per_task=args.libero_num_trials,
            resize_size=args.libero_resize,
            replan_steps=args.libero_replan_steps,
            num_steps_wait=args.libero_num_steps_wait,
            seed=args.seed,
            video_out_path=args.libero_video_out,
            max_steps=args.max_steps,
        )
        _write_output(results, args.output)
        logging.info("Evaluation complete. Overall success rate: %.1f%%", results["overall_success_rate"] * 100)
        return

    if args.suite == "dataset":
        if not args.checkpoint_dir:
            raise SystemExit("--checkpoint-dir is required for dataset evaluation.")
        from fla.benchmarks import dataset_eval

        train_config = None
        if args.recipe:
            if not args.repo_ids:
                raise SystemExit("--repo-ids are required when using --recipe.")
            from fla.finetune import recipes as _recipes

            overrides = _recipes.RecipeOverrides(
                repo_ids=tuple(args.repo_ids),
                repo_id_to_prompt=repo_id_to_prompt,
                exp_name="eval",
                base_model=args.base_model,
                init_from=args.init_from,
                paligemma_variant=args.paligemma_variant,
                action_expert_variant=args.action_expert_variant,
                action_dim=args.model_action_dim or 14,
                action_horizon=args.model_action_horizon or 50,
                use_delta_joint_actions=args.use_delta_joint_actions,
                prompt_from_task=args.prompt_from_task,
                default_prompt=default_prompt,
            )
            train_config = _recipes.build_train_config(args.recipe, overrides)
        else:
            if not args.config:
                raise SystemExit("--config is required for dataset evaluation unless --recipe is provided.")

        results = dataset_eval.run_dataset_eval(
            checkpoint_dir=args.checkpoint_dir,
            config_name=args.config,
            train_config=train_config,
            repo_ids=args.repo_ids,
            repo_id_to_prompt=repo_id_to_prompt or None,
            prompt_from_task=args.prompt_from_task if args.prompt_from_task else None,
            max_samples=args.max_samples,
            seed=args.seed,
        )
        _write_output(results, args.output)
        metrics = results["metrics"]
        logging.info("Evaluation complete. MSE=%.6f RMSE=%.6f L1=%.6f", metrics["mse"], metrics["rmse"], metrics["l1"])
        return

    if args.suite == "droid_manual":
        from fla.benchmarks import droid_manual

        results = droid_manual.launch_manual_eval(
            left_camera_id=args.droid_left_camera_id,
            right_camera_id=args.droid_right_camera_id,
            wrist_camera_id=args.droid_wrist_camera_id,
            external_camera=args.droid_external_camera,
            remote_host=args.droid_remote_host,
            remote_port=args.droid_remote_port,
            max_timesteps=args.droid_max_timesteps,
            open_loop_horizon=args.droid_open_loop_horizon,
        )
        _write_output(results, args.output)
        logging.info("DROID manual evaluation complete.")
        return

    if args.suite == "full":
        if not args.checkpoint_dir or not args.config:
            raise SystemExit("--checkpoint-dir and --config are required for full evaluation.")
        from fla.benchmarks import dataset_eval
        from fla.benchmarks import libero

        results = {"suite": "full", "components": {}}

        tasks = _parse_tasks(args)
        results["components"]["aloha_sim"] = aloha_sim.run_suite(
            checkpoint_dir=args.checkpoint_dir,
            config_name=args.config,
            tasks=tasks,
            num_episodes=args.num_episodes,
            seed=args.seed,
            action_horizon=args.action_horizon,
            max_steps=args.max_steps,
        )

        try:
            results["components"]["libero"] = libero.run_suite(
                checkpoint_dir=args.checkpoint_dir,
                config_name=args.config,
                task_suite_name=args.libero_suite,
                num_trials_per_task=args.libero_num_trials,
                resize_size=args.libero_resize,
                replan_steps=args.libero_replan_steps,
                num_steps_wait=args.libero_num_steps_wait,
                seed=args.seed,
                video_out_path=args.libero_video_out,
                max_steps=args.max_steps,
            )
        except Exception as exc:  # noqa: BLE001
            results["components"]["libero"] = {"status": "skipped", "error": str(exc)}

        if args.repo_ids:
            results["components"]["dataset"] = dataset_eval.run_dataset_eval(
                checkpoint_dir=args.checkpoint_dir,
                config_name=args.config,
                repo_ids=args.repo_ids,
                repo_id_to_prompt=repo_id_to_prompt or None,
                prompt_from_task=args.prompt_from_task if args.prompt_from_task else None,
                max_samples=args.max_samples,
                seed=args.seed,
            )
        else:
            results["components"]["dataset"] = {
                "status": "skipped",
                "error": "Provide --repo-ids to run dataset evaluation.",
            }

        _write_output(results, args.output)
        logging.info("Full evaluation complete.")
        return

    raise SystemExit(f"Unknown suite: {args.suite}")


if __name__ == "__main__":
    main()
