from __future__ import annotations

import dataclasses
from typing import Iterable


import fla.models.pi0_config as pi0_config
import fla.shared.nnx_utils as nnx_utils
import fla.training.continual_learning as continual_learning
import fla.training.optimizer as _optimizer
import fla.training.weight_loaders as weight_loaders
from fla.training.config import MultiLeRobotDataConfig
from fla.training.config import TrainConfig


@dataclasses.dataclass(frozen=True)
class RecipeSpec:
    name: str
    description: str
    supports_lora: bool
    supports_full_finetune: bool
    supports_ewc: bool
    default_action_dim: int = 14
    default_action_horizon: int = 50
    default_batch_size: int = 16
    default_num_train_steps: int = 30_000
    default_peak_lr: float = 5e-5


@dataclasses.dataclass(frozen=True)
class RecipeOverrides:
    repo_ids: tuple[str, ...]
    repo_id_to_prompt: dict[str, str]
    exp_name: str
    base_model: str = "pi05"
    init_from: str = "base"
    paligemma_variant: str | None = None
    action_expert_variant: str | None = None
    action_dim: int = 14
    action_horizon: int = 50
    use_delta_joint_actions: bool = False
    adapt_to_pi: bool | None = None
    prompt_from_task: bool = False
    default_prompt: str | None = None
    num_train_steps: int | None = None
    batch_size: int | None = None
    num_workers: int | None = None
    peak_lr: float | None = None
    checkpoint_path: str | None = None
    config_name: str | None = None
    assets_base_dir: str | None = None
    checkpoint_base_dir: str | None = None
    wandb_enabled: bool | None = None
    overwrite: bool | None = None
    resume: bool | None = None


def _recipe_specs(prefix: str, model_label: str) -> list[RecipeSpec]:
    return [
        RecipeSpec(
            name=f"{prefix}frozen_backbone",
            description=f"Freeze VLM backbone, train action expert ({model_label} base).",
            supports_lora=True,
            supports_full_finetune=True,
            supports_ewc=True,
        ),
        RecipeSpec(
            name=f"{prefix}full_finetune",
            description="Full-parameter fine-tuning (no frozen weights).",
            supports_lora=False,
            supports_full_finetune=True,
            supports_ewc=True,
            default_peak_lr=1e-5,
        ),
        RecipeSpec(
            name=f"{prefix}lora",
            description="LoRA adapters on VLM + action expert for efficient fine-tuning.",
            supports_lora=True,
            supports_full_finetune=False,
            supports_ewc=False,
            default_peak_lr=1e-4,
            default_batch_size=32,
            default_num_train_steps=20_000,
        ),
        RecipeSpec(
            name=f"{prefix}ewc_finetune",
            description="Frozen-backbone fine-tuning with EWC regularization.",
            supports_lora=False,
            supports_full_finetune=False,
            supports_ewc=True,
        ),
    ]


def list_recipes() -> list[RecipeSpec]:
    return [
        *_recipe_specs("pi0_", "Pi0"),
        *_recipe_specs("pi05_", "Pi0.5"),
    ]


def _get_recipe_spec(name: str) -> RecipeSpec:
    specs = {spec.name: spec for spec in list_recipes()}
    if name not in specs:
        available = ", ".join(sorted(specs))
        raise ValueError(f"Unknown recipe '{name}'. Available: {available}")
    return specs[name]


def _validate_repo_mapping(
    repo_ids: Iterable[str],
    repo_id_to_prompt: dict[str, str],
    prompt_from_task: bool,
    default_prompt: str | None,
) -> None:
    if not repo_ids:
        raise ValueError("repo_ids must be provided")
    if prompt_from_task or default_prompt is not None:
        return
    if not prompt_from_task:
        missing = [rid for rid in repo_ids if rid not in repo_id_to_prompt]
        if missing:
            missing_str = ", ".join(missing)
            raise ValueError(f"Missing prompts for repo_ids: {missing_str}")


def _build_data_config(overrides: RecipeOverrides) -> MultiLeRobotDataConfig:
    adapt_to_pi = overrides.adapt_to_pi
    if adapt_to_pi is None:
        adapt_to_pi = overrides.action_dim == 14

    _validate_repo_mapping(
        overrides.repo_ids,
        overrides.repo_id_to_prompt,
        overrides.prompt_from_task,
        overrides.default_prompt,
    )

    return MultiLeRobotDataConfig(
        repo_ids=overrides.repo_ids,
        repo_id_to_prompt=overrides.repo_id_to_prompt,
        default_prompt=overrides.default_prompt,
        prompt_from_task=overrides.prompt_from_task,
        use_delta_joint_actions=overrides.use_delta_joint_actions,
        adapt_to_pi=adapt_to_pi,
    )


def _build_lr_schedule(spec: RecipeSpec, overrides: RecipeOverrides) -> _optimizer.LRScheduleConfig:
    peak_lr = overrides.peak_lr if overrides.peak_lr is not None else spec.default_peak_lr
    decay_steps = overrides.num_train_steps if overrides.num_train_steps is not None else spec.default_num_train_steps
    warmup_steps = min(1_000, max(1, decay_steps // 10))
    return _optimizer.CosineDecaySchedule(
        warmup_steps=warmup_steps,
        peak_lr=peak_lr,
        decay_steps=decay_steps,
        decay_lr=max(peak_lr * 0.02, 1e-6),
    )


def build_train_config(recipe_name: str, overrides: RecipeOverrides) -> TrainConfig:
    import flax.nnx as nnx
    spec = _get_recipe_spec(recipe_name)

    normalized_recipe = recipe_name
    if recipe_name.startswith("pi05_"):
        normalized_recipe = recipe_name.replace("pi05_", "pi0_", 1)
        if overrides.base_model == "pi0":
            raise ValueError("pi05_* recipes require base_model='pi05'")

    config_name = overrides.config_name or recipe_name
    num_train_steps = overrides.num_train_steps if overrides.num_train_steps is not None else spec.default_num_train_steps
    batch_size = overrides.batch_size if overrides.batch_size is not None else spec.default_batch_size
    base_model = overrides.base_model.lower()
    if base_model not in {"pi0", "pi05"}:
        raise ValueError("base_model must be 'pi0' or 'pi05'")
    is_pi05 = base_model == "pi05"
    checkpoint_path = overrides.checkpoint_path or (
        "gs://openpi-assets/checkpoints/pi05_base/params"
        if is_pi05
        else "gs://openpi-assets/checkpoints/pi0_base/params"
    )

    model_kwargs = dict(
        pi05=is_pi05,
        paligemma_variant=overrides.paligemma_variant or "gemma_2b",
        action_expert_variant=overrides.action_expert_variant or "gemma_300m",
        action_dim=overrides.action_dim,
        action_horizon=overrides.action_horizon,
        freeze_vision_backbone=True,
    )

    freeze_filter = nnx_utils.PathRegex(".*PaliGemma/llm/(?!.*_1).*")
    ema_decay: float | None = 0.99
    continual_cfg = continual_learning.ContinualLearningConfig()

    if normalized_recipe == "pi0_full_finetune":
        freeze_filter = nnx.Nothing
        model_kwargs["freeze_vision_backbone"] = False
        ema_decay = 0.99
    elif normalized_recipe == "pi0_lora":
        if overrides.paligemma_variant is None:
            model_kwargs["paligemma_variant"] = "gemma_2b_lora"
        if overrides.action_expert_variant is None:
            model_kwargs["action_expert_variant"] = "gemma_300m_lora"
        if "lora" not in model_kwargs["paligemma_variant"] and "lora" not in model_kwargs["action_expert_variant"]:
            raise ValueError("pi0_lora recipe requires a LoRA variant (paligemma_variant or action_expert_variant).")
        freeze_filter = pi0_config.Pi0Config(
            paligemma_variant=model_kwargs["paligemma_variant"],
            action_expert_variant=model_kwargs["action_expert_variant"],
        ).get_freeze_filter()
        ema_decay = None
    elif normalized_recipe == "pi0_ewc_finetune":
        continual_cfg = continual_learning.ContinualLearningConfig(
            ewc=continual_learning.EWCConfig(enabled=True)
        )

    model = pi0_config.Pi0Config(**model_kwargs)

    init_from = overrides.init_from.lower()
    if init_from not in {"base", "scratch"}:
        raise ValueError("init_from must be 'base' or 'scratch'")
    weight_loader = (
        weight_loaders.NoOpWeightLoader()
        if init_from == "scratch"
        else weight_loaders.FlexibleCheckpointWeightLoader(checkpoint_path)
    )

    config = TrainConfig(
        name=config_name,
        exp_name=overrides.exp_name,
        model=model,
        data=_build_data_config(overrides),
        weight_loader=weight_loader,
        freeze_filter=freeze_filter,
        lr_schedule=_build_lr_schedule(spec, overrides),
        num_train_steps=num_train_steps,
        batch_size=batch_size,
        ema_decay=ema_decay,
        continual_learning=continual_cfg,
    )

    if overrides.assets_base_dir is not None:
        config = dataclasses.replace(config, assets_base_dir=overrides.assets_base_dir)
    if overrides.checkpoint_base_dir is not None:
        config = dataclasses.replace(config, checkpoint_base_dir=overrides.checkpoint_base_dir)
    if overrides.num_workers is not None:
        config = dataclasses.replace(config, num_workers=overrides.num_workers)
    if overrides.wandb_enabled is not None:
        config = dataclasses.replace(config, wandb_enabled=overrides.wandb_enabled)
    if overrides.overwrite is not None:
        config = dataclasses.replace(config, overwrite=overrides.overwrite)
    if overrides.resume is not None:
        config = dataclasses.replace(config, resume=overrides.resume)

    return config
