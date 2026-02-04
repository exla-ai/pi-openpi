#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys
from typing import Iterable


class RemoveStrings:
    def __call__(self, x: dict) -> dict:
        import numpy as np

        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}



def _parse_repo_id_to_prompt(items: Iterable[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in items:
        if ":" not in item:
            raise ValueError(f"Invalid --repo-id-to-prompt entry '{item}'. Use <repo_id>:<prompt> format.")
        repo_id, prompt = item.split(":", 1)
        mapping[repo_id.strip()] = prompt.strip()
    return mapping


def _load_train_module(project_root: pathlib.Path):
    train_path = project_root / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("fla_train_script", train_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load training module from {train_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _compute_norm_stats(config, max_frames: int | None = None) -> None:
    import numpy as np

    import fla.shared.normalize as normalize
    import fla.training.data_loader as data_loader

    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id to compute norm stats")
    asset_id = data_config.asset_id or data_config.repo_id

    if data_config.rlds_data_dir is not None:
        dataset = data_loader.create_rlds_dataset(
            data_config, config.model.action_horizon, config.batch_size, shuffle=False
        )
        dataset = data_loader.IterableTransformedDataset(
            dataset,
            [
                *data_config.repack_transforms.inputs,
                *data_config.data_transforms.inputs,
                RemoveStrings(),
            ],
            is_batched=True,
        )
        if max_frames is not None and max_frames < len(dataset):
            num_batches = max_frames // config.batch_size
        else:
            num_batches = len(dataset) // config.batch_size
        data_iter = iter(dataset)
    else:
        dataset = data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
        dataset = data_loader.TransformedDataset(
            dataset,
            [
                *data_config.repack_transforms.inputs,
                *data_config.data_transforms.inputs,
                RemoveStrings(),
            ],
        )
        if max_frames is not None and max_frames < len(dataset):
            num_batches = max_frames // config.batch_size
            shuffle = True
        else:
            num_batches = len(dataset) // config.batch_size
            shuffle = False
        torch_loader = data_loader.TorchDataLoader(
            dataset,
            local_batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=shuffle,
            num_batches=num_batches,
        )
        data_iter = iter(torch_loader)

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}
    for _ in range(num_batches):
        batch = next(data_iter)
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}
    output_path = config.assets_dirs / asset_id
    normalize.save(output_path, norm_stats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Pi0/Pi0.5 with FLA recipes")
    parser.add_argument("--list", action="store_true", help="List available recipes")
    parser.add_argument("--recipe", type=str, help="Recipe name (see --list)")
    parser.add_argument("--repo-ids", nargs="+", help="LeRobot dataset repo IDs")
    parser.add_argument(
        "--repo-id-to-prompt",
        action="append",
        default=[],
        help="Mapping in the form <repo_id>:<prompt> (repeatable)",
    )
    parser.add_argument(
        "--default-prompt",
        type=str,
        default=None,
        help="Default prompt if prompt_from_task is False and a repo_id is missing a prompt",
    )
    parser.add_argument(
        "--prompt-from-task",
        action="store_true",
        help="Use prompt stored in the dataset task field",
    )
    parser.add_argument("--action-dim", type=int, default=14, help="Action dimension (7 for single arm, 14 for bimanual)")
    parser.add_argument("--action-horizon", type=int, default=50, help="Action horizon")
    parser.add_argument("--use-delta-joint-actions", action="store_true", help="Convert actions to deltas")
    parser.add_argument(
        "--base-model",
        type=str,
        default="pi05",
        choices=["pi0", "pi05"],
        help="Base model family to fine-tune",
    )
    parser.add_argument(
        "--init-from",
        type=str,
        default="base",
        choices=["base", "scratch"],
        help="Initialize from base weights or train from scratch",
    )
    parser.add_argument(
        "--paligemma-variant",
        type=str,
        default=None,
        help="Override the VLM backbone variant (e.g., gemma_300m, gemma_2b, gemma_2b_lora)",
    )
    parser.add_argument(
        "--action-expert-variant",
        type=str,
        default=None,
        help="Override the action expert variant (e.g., gemma_300m, gemma_300m_lora)",
    )

    adapt_group = parser.add_mutually_exclusive_group()
    adapt_group.add_argument("--adapt-to-pi", action="store_true", help="Adapt actions to Pi action space")
    adapt_group.add_argument("--no-adapt-to-pi", action="store_true", help="Disable Pi action adaptation")

    parser.add_argument("--num-train-steps", type=int, default=None, help="Override number of training steps")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--num-workers", type=int, default=None, help="Override data loader workers")
    parser.add_argument("--peak-lr", type=float, default=None, help="Override peak learning rate")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Override base checkpoint path")

    parser.add_argument("--config-name", type=str, default=None, help="Override training config name")
    parser.add_argument("--exp-name", type=str, help="Experiment name (required for training)")
    parser.add_argument("--assets-base-dir", type=str, default=None, help="Override assets base directory")
    parser.add_argument("--checkpoint-base-dir", type=str, default=None, help="Override checkpoint base directory")

    wandb_group = parser.add_mutually_exclusive_group()
    wandb_group.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    wandb_group.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")

    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing checkpoints")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit")
    parser.add_argument(
        "--skip-norm-stats",
        action="store_true",
        help="Skip computing normalization stats (requires stats to already exist)",
    )
    parser.add_argument(
        "--max-norm-frames",
        type=int,
        default=None,
        help="Max frames to use when computing norm stats",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = pathlib.Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    src_root = project_root / "src"
    if src_root.exists() and str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    from fla.finetune import RecipeOverrides
    from fla.finetune import build_train_config
    from fla.finetune import list_recipes

    if args.list:
        for recipe in list_recipes():
            print(f"{recipe.name}: {recipe.description}")
        return

    if not args.recipe:
        raise SystemExit("--recipe is required unless --list is specified")
    if not args.repo_ids:
        raise SystemExit("--repo-ids is required")
    if not args.exp_name:
        raise SystemExit("--exp-name is required")

    repo_id_to_prompt = _parse_repo_id_to_prompt(args.repo_id_to_prompt)
    if not repo_id_to_prompt and not args.prompt_from_task and not args.default_prompt:
        raise SystemExit("Provide --repo-id-to-prompt, or use --prompt-from-task, or set --default-prompt")

    adapt_to_pi = None
    if args.adapt_to_pi:
        adapt_to_pi = True
    elif args.no_adapt_to_pi:
        adapt_to_pi = False

    wandb_enabled = None
    if args.wandb:
        wandb_enabled = True
    elif args.no_wandb:
        wandb_enabled = False

    overrides = RecipeOverrides(
        repo_ids=tuple(args.repo_ids),
        repo_id_to_prompt=repo_id_to_prompt,
        exp_name=args.exp_name,
        base_model=args.base_model,
        init_from=args.init_from,
        paligemma_variant=args.paligemma_variant,
        action_expert_variant=args.action_expert_variant,
        action_dim=args.action_dim,
        action_horizon=args.action_horizon,
        use_delta_joint_actions=args.use_delta_joint_actions,
        adapt_to_pi=adapt_to_pi,
        prompt_from_task=args.prompt_from_task,
        default_prompt=args.default_prompt,
        num_train_steps=args.num_train_steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        peak_lr=args.peak_lr,
        checkpoint_path=args.checkpoint_path,
        config_name=args.config_name,
        assets_base_dir=args.assets_base_dir,
        checkpoint_base_dir=args.checkpoint_base_dir,
        wandb_enabled=wandb_enabled,
        overwrite=args.overwrite if args.overwrite else None,
        resume=args.resume if args.resume else None,
    )

    config = build_train_config(args.recipe, overrides)

    if args.dry_run:
        print(config)
        return

    if not args.skip_norm_stats:
        data_config = config.data.create(config.assets_dirs, config.model)
        if data_config.repo_id is None:
            raise SystemExit("Data config repo_id is required for norm stats")
        asset_id = data_config.asset_id or data_config.repo_id
        norm_path = config.assets_dirs / asset_id / "norm_stats.json"
        if not norm_path.exists():
            print(f"[INFO] Computing norm stats to {norm_path}")
            _compute_norm_stats(config, max_frames=args.max_norm_frames)

    train_module = _load_train_module(project_root)
    train_module.main(config)


if __name__ == "__main__":
    main()
