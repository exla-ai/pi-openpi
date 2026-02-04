#!/usr/bin/env python3
"""Compute EWC (Elastic Weight Consolidation) state from a trained checkpoint.

This script computes the Fisher information matrix for a trained model,
which is used to prevent catastrophic forgetting when training on new tasks.

Usage:
    # Compute Fisher from a trained checkpoint
    python scripts/compute_ewc_state.py pi06_multi --exp-name v1 --checkpoint-step 50000

    # Then enable EWC in your next training run
    python scripts/train.py pi06_new_task --exp-name v1 \
        --continual_learning.ewc.enabled=True \
        --continual_learning.ewc.lambda_ewc=1000
"""

import argparse
import logging
import pathlib

import jax
import flax.nnx as nnx

import fla.training.config as _config
import fla.training.checkpoints as _checkpoints
import fla.training.continual_learning as _continual_learning
import fla.training.data_loader as _data_loader
import fla.training.sharding as sharding
import fla.training.utils as training_utils


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


def main():
    parser = argparse.ArgumentParser(
        description="Compute EWC state (Fisher information) from a trained checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Compute Fisher from the latest checkpoint
    python scripts/compute_ewc_state.py pi06_multi --exp-name v1

    # Compute Fisher from a specific checkpoint step
    python scripts/compute_ewc_state.py pi06_multi --exp-name v1 --checkpoint-step 50000

    # Compute Fisher with more samples for better estimation
    python scripts/compute_ewc_state.py pi06_multi --exp-name v1 --num-samples 500

After computing, the EWC state will be saved and automatically used when training
on new tasks with --continual_learning.ewc.enabled=True
""",
    )

    parser.add_argument("config_name", help="Name of the training config")
    parser.add_argument("--exp-name", required=True, help="Experiment name")
    parser.add_argument("--checkpoint-step", type=int, default=None, help="Checkpoint step (default: latest)")
    parser.add_argument("--num-samples", type=int, default=200, help="Number of samples for Fisher estimation")
    parser.add_argument("--output", type=str, default=None, help="Output path for EWC state (default: checkpoint_dir/ewc_state.pkl)")

    args = parser.parse_args()

    init_logging()

    # Get the config
    config = _config.get_config(args.config_name)
    config = config.model_copy(update={"exp_name": args.exp_name})

    logging.info(f"Loading checkpoint from {config.checkpoint_dir}")

    # Initialize mesh and sharding
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))

    # Initialize checkpoint manager
    checkpoint_manager, _ = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=False,
        resume=True,  # We're loading an existing checkpoint
    )

    # Create data loader
    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )

    # Initialize train state shape
    rng = jax.random.key(config.seed)
    _, init_rng = jax.random.split(rng)

    from fla.training.optimizer import create_optimizer
    from fla.training.weight_loaders import NoOpWeightLoader

    # Create model to get state shape
    def init_model(rng):
        model_rng = jax.random.split(rng)[1]
        model = config.model.create(model_rng)
        params = nnx.state(model)
        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None),
            opt_state=None,
            ema_decay=None,
            ema_params=None,
        )

    train_state_shape = jax.eval_shape(init_model, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=False)

    # Restore checkpoint
    train_state = _checkpoints.restore_state(checkpoint_manager, train_state_shape, data_loader)
    logging.info(f"Loaded checkpoint at step {train_state.step}")

    # Create model from state
    model = nnx.merge(train_state.model_def, train_state.params)

    # Compute Fisher information
    logging.info(f"Computing Fisher information from {args.num_samples} samples...")
    ewc_rng = jax.random.key(42)

    fisher = _continual_learning.compute_fisher_information(
        model=model,
        data_loader=data_loader,
        trainable_filter=config.trainable_filter,
        num_samples=args.num_samples,
        rng=ewc_rng,
    )

    # Create EWC state
    current_params = train_state.params.filter(config.trainable_filter).to_pure_dict()
    ewc_state = _continual_learning.EWCState(
        fisher=fisher,
        optimal_params=current_params,
        task_count=1,
    )

    # Save EWC state
    output_path = args.output
    if output_path is None:
        output_path = str(config.checkpoint_dir.parent / "ewc_state.pkl")

    _continual_learning.save_ewc_state(ewc_state, output_path)
    logging.info(f"Saved EWC state to {output_path}")
    logging.info("")
    logging.info("To use continual learning in your next training run:")
    logging.info(f"  python scripts/train.py <new_config> --exp-name <name> \\")
    logging.info(f"      --continual_learning.ewc.enabled=True \\")
    logging.info(f"      --continual_learning.ewc.lambda_ewc=1000")


if __name__ == "__main__":
    main()
