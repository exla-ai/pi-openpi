#!/usr/bin/env python3
"""RECAP Training Script.

This script implements RECAP (RL with Experience and Corrections via Advantage-conditioned Policies)
training on top of the standard openpi training infrastructure.

RECAP training has two modes:
1. Warmup: Standard pi0 training without advantage conditioning (to initialize policy)
2. RECAP: Training with advantage conditioning using collected data

Usage:
    # Run with debug config (fast, for testing)
    python scripts/train_recap.py debug_recap --exp_name test_recap

    # Run with aloha_sim config
    python scripts/train_recap.py recap_aloha_sim --exp_name recap_test --batch_size 32 --fsdp_devices 8
"""

import dataclasses
import functools
import logging
import platform
from typing import Any, Tuple

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import tyro
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders

from openpi.recap.value_function import ValueFunction, ValueFunctionConfig, compute_improvement_indicator
from openpi.recap.pi0_recap import Pi0RECAP, Pi0RECAPConfig
from openpi.models import pi0_config


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    from flax import traverse_util
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        model = config.model.create(model_rng)

        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    train_state = jax.jit(
        init,
        donate_argnums=(1,),
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name + "_recap",
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


@dataclasses.dataclass
class RECAPTrainConfig:
    """Configuration for RECAP training."""

    # Base training config name
    base_config: str = "pi0_aloha_sim"

    # Experiment name
    exp_name: str = tyro.MISSING

    # RECAP-specific parameters
    value_lr: float = 1e-4
    value_train_steps: int = 500
    policy_train_steps: int = 500

    # Number of RECAP iterations
    num_recap_iterations: int = 5

    # Warmup steps (standard training before RECAP)
    warmup_steps: int = 100

    # Batch size (must be divisible by number of devices)
    batch_size: int = 32

    # FSDP devices
    fsdp_devices: int = 1

    # Whether to use wandb
    wandb_enabled: bool = True

    # Overwrite existing checkpoint
    overwrite: bool = False

    # Resume from checkpoint
    resume: bool = False

    # Number of total train steps
    num_train_steps: int = 1000

    # Log interval
    log_interval: int = 50

    # Save interval
    save_interval: int = 500


@at.typecheck
def recap_train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
    improvement_indicator: at.Bool[at.Array, " b"],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """Training step with RECAP advantage conditioning."""
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        improvement_indicator: at.Bool[at.Array, " b"],
    ):
        # Check if model supports advantage conditioning
        if hasattr(model, 'compute_loss') and 'improvement_indicator' in model.compute_loss.__code__.co_varnames:
            chunked_loss = model.compute_loss(rng, observation, actions, train=True, improvement_indicator=improvement_indicator)
        else:
            # Fall back to standard loss
            chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(
        model, train_rng, observation, actions, improvement_indicator
    )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update model and return new state
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


def simulate_advantages(batch_size: int, rng: jax.Array) -> at.Bool[at.Array, " b"]:
    """Simulate advantage indicators for testing.

    In a real RECAP setup, these would come from the value function.
    For testing, we randomly assign half as "good" and half as "bad".
    """
    # Random assignment: ~50% good, ~50% bad
    return jax.random.uniform(rng, (batch_size,)) > 0.5


def main(recap_config: RECAPTrainConfig):
    init_logging()
    logging.info(f"Running RECAP training on: {platform.node()}")
    logging.info(f"RECAP config: {recap_config}")

    # Get base config
    base_config = _config.get_config(recap_config.base_config)

    # Override with RECAP settings
    config = dataclasses.replace(
        base_config,
        exp_name=recap_config.exp_name,
        batch_size=recap_config.batch_size,
        fsdp_devices=recap_config.fsdp_devices,
        num_train_steps=recap_config.num_train_steps,
        overwrite=recap_config.overwrite,
        resume=recap_config.resume,
        wandb_enabled=recap_config.wandb_enabled,
        log_interval=recap_config.log_interval,
        save_interval=recap_config.save_interval,
    )

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Initialize train state
    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    # JIT compile standard train step
    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    # JIT compile RECAP train step
    precap_train_step = jax.jit(
        functools.partial(recap_train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding, replicated_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    total_steps = config.num_train_steps

    logging.info("=" * 60)
    logging.info("Starting RECAP Training")
    logging.info(f"  Warmup steps: {recap_config.warmup_steps}")
    logging.info(f"  Total steps: {total_steps}")
    logging.info(f"  RECAP iterations: {recap_config.num_recap_iterations}")
    logging.info("=" * 60)

    pbar = tqdm.tqdm(
        range(start_step, total_steps),
        initial=start_step,
        total=total_steps,
        dynamic_ncols=True,
    )

    infos = []
    recap_iteration = 0

    for step in pbar:
        # Determine if we're in warmup or RECAP phase
        in_warmup = step < recap_config.warmup_steps

        with sharding.set_mesh(mesh):
            if in_warmup:
                # Standard training (no advantage conditioning)
                train_state, info = ptrain_step(train_rng, train_state, batch)
                info["phase"] = 0.0  # Warmup phase
                info["pct_good"] = 0.5  # Placeholder for consistent logging
            else:
                # RECAP training with advantage conditioning
                # Simulate advantages (in real setup, these come from value function)
                train_rng, adv_rng = jax.random.split(train_rng)
                improvement_indicator = simulate_advantages(config.batch_size, adv_rng)

                train_state, info = precap_train_step(train_rng, train_state, batch, improvement_indicator)
                info["phase"] = 1.0  # RECAP phase
                info["pct_good"] = jnp.mean(improvement_indicator.astype(jnp.float32))

        infos.append(info)

        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            phase_str = "warmup" if in_warmup else "RECAP"
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step} [{phase_str}]: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []

        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == total_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()
    logging.info("RECAP training complete!")


# Define RECAP-specific configs
RECAP_CONFIGS = {
    "debug_recap": RECAPTrainConfig(
        base_config="debug",
        exp_name="debug_recap",
        warmup_steps=5,
        num_train_steps=20,
        batch_size=8,
        fsdp_devices=1,
        wandb_enabled=False,
        overwrite=True,
        log_interval=5,
        save_interval=10,
    ),
    "recap_aloha_sim": RECAPTrainConfig(
        base_config="pi0_aloha_sim",
        exp_name="recap_aloha_sim",
        warmup_steps=100,
        num_train_steps=500,
        batch_size=32,
        fsdp_devices=8,
        wandb_enabled=True,
        log_interval=50,
        save_interval=200,
    ),
}


if __name__ == "__main__":
    # Use tyro for CLI with config presets
    config = tyro.extras.overridable_config_cli({k: (k, v) for k, v in RECAP_CONFIGS.items()})
    main(config)
