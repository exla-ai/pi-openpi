from __future__ import annotations

import dataclasses
import logging
from typing import Iterable

import numpy as np

from fla.policies import policy_config as _policy_config
from fla.training import config as _config
from fla.training import data_loader as _data_loader

logger = logging.getLogger(__name__)


def _to_numpy(value):
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(value)


def _select_action_key(sample: dict, action_keys: Iterable[str]) -> str:
    for key in action_keys:
        if key in sample:
            return key
    if "actions" in sample:
        return "actions"
    if "action" in sample:
        return "action"
    raise KeyError(f"Could not find actions in sample. Available keys: {list(sample.keys())}")


def _align_actions(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = _to_numpy(pred)
    gt = _to_numpy(gt)

    if pred.ndim == 1:
        pred = pred[None, :]
    if gt.ndim == 1:
        gt = gt[None, :]

    min_t = min(pred.shape[0], gt.shape[0])
    min_d = min(pred.shape[-1], gt.shape[-1])
    pred = pred[:min_t, :min_d]
    gt = gt[:min_t, :min_d]
    return pred, gt


def run_dataset_eval(
    *,
    checkpoint_dir: str,
    config_name: str | None = None,
    train_config: _config.TrainConfig | None = None,
    repo_ids: list[str] | None = None,
    repo_id_to_prompt: dict[str, str] | None = None,
    prompt_from_task: bool | None = None,
    max_samples: int = 1024,
    seed: int = 0,
) -> dict:
    """Evaluate a checkpoint on a LeRobot dataset by action prediction error."""
    if train_config is None:
        if config_name is None:
            raise ValueError("Provide config_name or train_config.")
        train_config = _config.get_config(config_name)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)

    if repo_ids:
        data_config = dataclasses.replace(
            data_config,
            repo_id=repo_ids[0],
            additional_repo_ids=tuple(repo_ids[1:]),
        )
    if repo_id_to_prompt:
        data_config = dataclasses.replace(data_config, repo_id_to_prompt=repo_id_to_prompt)
    if prompt_from_task is not None:
        data_config = dataclasses.replace(data_config, prompt_from_task=prompt_from_task)

    if data_config.repo_id is None:
        raise ValueError("No dataset specified. Provide --repo-ids or use a config with repo_id set.")

    logger.info("Loading dataset: %s", data_config.repo_id)
    dataset = _data_loader.create_torch_dataset(
        data_config,
        train_config.model.action_horizon,
        train_config.model,
    )

    logger.info("Loading policy from checkpoint: %s", checkpoint_dir)
    policy = _policy_config.create_trained_policy(
        train_config=train_config,
        checkpoint_dir=checkpoint_dir,
        repack_transforms=data_config.repack_transforms,
    )

    rng = np.random.default_rng(seed)
    dataset_len = len(dataset)
    sample_count = min(max_samples, dataset_len)

    if sample_count <= 0:
        raise ValueError("Dataset has no samples to evaluate.")

    indices = rng.choice(dataset_len, size=sample_count, replace=sample_count > dataset_len)
    action_key = None

    total_mse = 0.0
    total_l1 = 0.0

    for idx in indices:
        sample = dataset[int(idx)]
        if action_key is None:
            action_key = _select_action_key(sample, data_config.action_sequence_keys)

        pred = policy.infer(sample)["actions"]
        gt = sample[action_key]
        pred_arr, gt_arr = _align_actions(pred, gt)

        diff = pred_arr - gt_arr
        total_mse += float(np.mean(diff ** 2))
        total_l1 += float(np.mean(np.abs(diff)))

    mean_mse = total_mse / sample_count
    mean_l1 = total_l1 / sample_count
    rmse = float(np.sqrt(mean_mse))

    return {
        "suite": "dataset",
        "config": config_name,
        "checkpoint_dir": checkpoint_dir,
        "repo_ids": [data_config.repo_id, *data_config.additional_repo_ids],
        "num_samples": sample_count,
        "metrics": {
            "mse": mean_mse,
            "rmse": rmse,
            "l1": mean_l1,
        },
    }
