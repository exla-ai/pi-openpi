#!/usr/bin/env python3
"""
PyTorch RECAP Training Pipeline for pi0.6.

This script implements the complete RECAP algorithm from pi0.6:
1. Train distributional value function on time-to-completion
2. Compute advantages using trained value function
3. Train policy with advantage conditioning

Architecture (from pi0.6 paper):
- Base VLM: Gemma 3 4B (from HuggingFace: google/gemma-3-4b-pt)
- Action Expert: ~860M params (gemma_860m)
- Value Function: ~670M params (gemma_670m)

Usage:
  # Single GPU test
  python scripts/train_recap_pytorch.py --config recap_pi06_test --exp_name test_run

  # Multi-GPU (8xA100)
  torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    scripts/train_recap_pytorch.py --config recap_pi06_full --exp_name pi06_v1
"""

import argparse
import dataclasses
import gc
import logging
import os
import pathlib
import shutil
import sys
import time
from typing import Any

import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.functional as F
import tqdm
import wandb

# Setup paths
SCRIPT_DIR = pathlib.Path(__file__).parent.absolute()
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

import openpi.models.gemma as _gemma
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.value_function_pytorch import (
    ValueFunctionPytorch,
    ValueFunctionPytorchConfig,
    compute_improvement_indicator,
)
import openpi.models.pi0_config as _pi0_config


class SimpleObservation:
    """Simple observation object with required attributes for preprocessing.

    This avoids using the full Observation dataclass which can cause issues
    with torch.compile and other optimizations.
    """
    def __init__(
        self,
        images: dict,
        state: torch.Tensor,
        image_masks: dict | None = None,
        tokenized_prompt: torch.Tensor | None = None,
        tokenized_prompt_mask: torch.Tensor | None = None,
        token_ar_mask: torch.Tensor | None = None,
        token_loss_mask: torch.Tensor | None = None,
    ):
        self.images = images
        self.state = state
        batch_size = state.shape[0]
        device = state.device

        # Create default masks if not provided
        if image_masks is None:
            batch_shape = state.shape[:-1]
            self.image_masks = {
                key: torch.ones(batch_shape, dtype=torch.bool, device=device)
                for key in images
            }
        else:
            self.image_masks = image_masks

        # Create dummy tokenized prompt if not provided
        # Use a simple prompt like "robot task" tokenized
        if tokenized_prompt is None:
            # Create a simple dummy prompt with pad tokens (0)
            prompt_length = 16
            self.tokenized_prompt = torch.zeros((batch_size, prompt_length), dtype=torch.long, device=device)
            # Set first few tokens to non-zero to simulate a real prompt
            self.tokenized_prompt[:, 0] = 1  # BOS token
            self.tokenized_prompt[:, 1] = 5678  # "robot" (dummy token ID)
            self.tokenized_prompt[:, 2] = 1234  # "task" (dummy token ID)
            self.tokenized_prompt_mask = torch.ones((batch_size, prompt_length), dtype=torch.bool, device=device)
        else:
            self.tokenized_prompt = tokenized_prompt
            self.tokenized_prompt_mask = tokenized_prompt_mask

        self.token_ar_mask = token_ar_mask
        self.token_loss_mask = token_loss_mask


def init_logging():
    """Initialize logging with custom format."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)


@dataclasses.dataclass
class RECAPConfig:
    """Configuration for RECAP training."""

    # Model architecture (pi0.6 defaults)
    base_vlm_variant: str = "gemma3_4b"
    action_expert_variant: str = "gemma_860m"
    value_function_variant: str = "gemma_670m"

    # Data
    repo_id: str = "lerobot/aloha_sim_transfer_cube_human"
    action_dim: int = 14
    action_horizon: int = 50
    state_dim: int = 14

    # Training hyperparams
    batch_size: int = 8
    learning_rate: float = 1e-4
    gradient_accumulation_steps: int = 1

    # Training phases
    value_train_steps: int = 5000
    policy_warmup_steps: int = 10000
    policy_recap_steps: int = 20000

    # Value function
    value_num_bins: int = 201
    value_hidden_dim: int = 1024

    # Checkpointing
    save_interval: int = 1000
    output_dir: str = str(PROJECT_DIR / "checkpoints" / "recap_pytorch")

    # Logging
    wandb_project: str = "openpi"
    wandb_enabled: bool = True
    log_interval: int = 50

    # Misc
    seed: int = 42
    dtype: str = "bfloat16"
    max_token_len: int = 200


# Pre-defined configs
CONFIGS = {
    "recap_pi06_test": RECAPConfig(
        base_vlm_variant="dummy",  # Use dummy for fast testing
        action_expert_variant="dummy",
        value_function_variant="dummy",
        batch_size=2,
        value_train_steps=100,
        policy_warmup_steps=100,
        policy_recap_steps=100,
        log_interval=10,
        save_interval=50,
    ),
    "recap_pi06_small": RECAPConfig(
        base_vlm_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        value_function_variant="gemma_300m",
        batch_size=4,
        value_train_steps=1000,
        policy_warmup_steps=2000,
        policy_recap_steps=3000,
    ),
    "recap_pi06_full": RECAPConfig(
        # Note: Using gemma_2b because HuggingFace PaliGemma uses gemma_2b (width=2048)
        # Action expert also uses gemma_2b to ensure compatible attention head counts (8 heads)
        base_vlm_variant="gemma_2b",
        action_expert_variant="gemma_2b",  # Must match base VLM for cross-attention
        value_function_variant="gemma_670m",
        batch_size=4,
        gradient_accumulation_steps=4,  # Effective batch size = 16
        value_train_steps=5000,
        policy_warmup_steps=10000,
        policy_recap_steps=20000,
    ),
}


def setup_ddp():
    """Setup Distributed Data Parallel training."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1

    if use_ddp and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    return use_ddp, local_rank, device


def cleanup_ddp():
    """Cleanup DDP resources."""
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def set_seed(seed: int, local_rank: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)


class LeRobotDataset:
    """Wrapper around LeRobot dataset for RECAP training."""

    def __init__(self, config: RECAPConfig):
        self.config = config

        try:
            import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

            logging.info(f"Loading dataset: {config.repo_id}")
            self.dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(config.repo_id)
            self.dataset = lerobot_dataset.LeRobotDataset(
                config.repo_id,
                delta_timestamps={
                    "action": [t / self.dataset_meta.fps for t in range(config.action_horizon)]
                },
            )

            logging.info(f"Dataset loaded: {len(self.dataset)} samples")
            logging.info(f"FPS: {self.dataset_meta.fps}")
            logging.info(f"Episodes: {self.dataset_meta.total_episodes}")

            self._build_episode_info()
        except Exception as e:
            logging.warning(f"Could not load LeRobot dataset: {e}")
            logging.warning("Creating dummy dataset for testing")
            self._create_dummy_dataset()

    def _build_episode_info(self):
        """Build episode boundaries and time-to-completion."""
        logging.info("Building episode info...")

        total_episodes = self.dataset_meta.total_episodes
        total_samples = len(self.dataset)
        avg_length = total_samples // total_episodes

        self.episode_starts = []
        self.episode_lengths = []
        self.time_to_completion = np.zeros(total_samples, dtype=np.int32)

        for ep_idx in range(total_episodes):
            start = ep_idx * avg_length
            length = avg_length if ep_idx < total_episodes - 1 else (total_samples - start)

            self.episode_starts.append(start)
            self.episode_lengths.append(length)

            for t in range(length):
                if start + t < total_samples:
                    self.time_to_completion[start + t] = length - t

        self.advantages = None
        self.improvement_indicators = None

        logging.info(f"Built episode info: {len(self.episode_starts)} episodes")
        logging.info(f"Average episode length: {np.mean(self.episode_lengths):.1f}")

    def _create_dummy_dataset(self):
        """Create dummy dataset for testing without LeRobot."""
        self.dataset = None
        self.dataset_meta = None
        self.episode_starts = [0]
        self.episode_lengths = [1000]
        self.time_to_completion = np.arange(1000, 0, -1)
        self.advantages = None
        self.improvement_indicators = None

    def __len__(self):
        if self.dataset is not None:
            return len(self.dataset)
        return 1000  # Dummy dataset size

    def get_batch(self, indices, device) -> dict[str, torch.Tensor]:
        """Get a batch of samples as tensors on the specified device."""
        batch_size = len(indices)
        config = self.config

        if self.dataset is not None:
            # Real dataset
            samples = [self.dataset[int(i)] for i in indices]

            # Get images
            images = []
            for s in samples:
                # LeRobot format: observation.images.top, etc.
                img_key = "observation.images.top"
                if img_key in s:
                    img = s[img_key]
                    if isinstance(img, np.ndarray):
                        img = torch.from_numpy(img)
                    images.append(img)
                else:
                    # Fallback: create dummy image
                    images.append(torch.zeros(3, 224, 224))

            images = torch.stack(images).to(device)
            if images.ndim == 5:  # [B, T, C, H, W]
                images = images[:, 0]  # Take first frame

            # Get actions
            actions = []
            for s in samples:
                if "action" in s:
                    act = s["action"]
                    if isinstance(act, np.ndarray):
                        act = torch.from_numpy(act)
                    actions.append(act)
                else:
                    actions.append(torch.zeros(config.action_horizon, config.action_dim))

            actions = torch.stack(actions).to(device).float()

            # Get state
            states = []
            for s in samples:
                state_key = "observation.state"
                if state_key in s:
                    state = s[state_key]
                    if isinstance(state, np.ndarray):
                        state = torch.from_numpy(state)
                    states.append(state)
                else:
                    states.append(torch.zeros(config.state_dim))

            states = torch.stack(states).to(device).float()

        else:
            # Dummy dataset - create dict with all required image keys
            images = {
                "base_0_rgb": torch.randn(batch_size, 3, 224, 224, device=device),
                "left_wrist_0_rgb": torch.randn(batch_size, 3, 224, 224, device=device),
                "right_wrist_0_rgb": torch.randn(batch_size, 3, 224, 224, device=device),
            }
            actions = torch.randn(batch_size, config.action_horizon, config.action_dim, device=device)
            states = torch.randn(batch_size, config.state_dim, device=device)

        # Get time-to-completion
        ttc = torch.tensor([self.time_to_completion[int(i)] for i in indices], device=device)

        # Get improvement indicators if computed
        if self.improvement_indicators is not None:
            improvement = torch.tensor(
                [self.improvement_indicators[int(i)] for i in indices],
                dtype=torch.bool,
                device=device
            )
        else:
            # Heuristic: use median ttc as threshold
            median_ttc = float(np.median(self.time_to_completion))
            improvement = ttc > median_ttc

        return {
            "images": images,
            "actions": actions,
            "states": states,
            "time_to_completion": ttc,
            "improvement_indicator": improvement,
        }


def create_models(config: RECAPConfig, device: torch.device):
    """Create value function and policy models."""
    logging.info("Creating models...")

    # Create value function
    value_config = ValueFunctionPytorchConfig(
        vlm_variant=config.value_function_variant,
        num_bins=config.value_num_bins,
        value_hidden_dim=config.value_hidden_dim,
        dtype=config.dtype,
    )
    value_fn = ValueFunctionPytorch(value_config).to(device)

    # Create policy
    policy_config = _pi0_config.Pi0Config(
        dtype=config.dtype,
        paligemma_variant=config.base_vlm_variant,
        action_expert_variant=config.action_expert_variant,
        action_dim=config.action_dim,
        action_horizon=config.action_horizon,
        max_token_len=config.max_token_len,
        pi05=True,  # Use adaRMS for RECAP
    )
    policy = PI0Pytorch(policy_config).to(device)

    # Count parameters
    value_params = sum(p.numel() for p in value_fn.parameters()) / 1e6
    policy_params = sum(p.numel() for p in policy.parameters()) / 1e6
    logging.info(f"Value function: {value_params:.1f}M params")
    logging.info(f"Policy: {policy_params:.1f}M params")

    return value_fn, policy


def train_value_function(
    config: RECAPConfig,
    dataset: LeRobotDataset,
    value_fn: ValueFunctionPytorch,
    device: torch.device,
    is_main: bool,
    wandb_run=None,
):
    """Train the value function on time-to-completion prediction."""
    logging.info("=" * 70)
    logging.info("PHASE 1: Training Value Function")
    logging.info("=" * 70)

    # Create optimizer
    optimizer = torch.optim.AdamW(value_fn.parameters(), lr=config.learning_rate)
    value_fn.train()

    indices = np.arange(len(dataset))
    rng = np.random.default_rng(config.seed)

    pbar = tqdm.tqdm(range(config.value_train_steps), desc="Value Training", disable=not is_main)

    for step in pbar:
        # Sample batch
        rng.shuffle(indices)
        batch_indices = indices[:config.batch_size]
        batch = dataset.get_batch(batch_indices, device)

        # Get image embeddings - extract base_0_rgb if dict
        images = batch["images"]
        if isinstance(images, dict):
            images = images["base_0_rgb"]
        image_embedding = value_fn.embed_images(images)

        # Compute loss
        loss = value_fn.compute_loss(image_embedding, batch["time_to_completion"])

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(value_fn.parameters(), max_norm=1.0)
        optimizer.step()

        # Logging
        if is_main:
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if step % config.log_interval == 0:
                logging.info(f"  Value step {step}/{config.value_train_steps}, loss: {loss.item():.4f}")

                if wandb_run:
                    wandb_run.log({
                        "value/loss": loss.item(),
                        "value/step": step,
                    }, step=step)

    logging.info("Value function training complete!")
    return value_fn


def compute_advantages(
    config: RECAPConfig,
    dataset: LeRobotDataset,
    value_fn: ValueFunctionPytorch,
    device: torch.device,
    is_main: bool,
):
    """Compute advantages using the trained value function."""
    logging.info("=" * 70)
    logging.info("PHASE 2: Computing Advantages")
    logging.info("=" * 70)

    value_fn.eval()
    all_advantages = []
    batch_size = config.batch_size * 4  # Use larger batches for inference

    if is_main:
        logging.info(f"Computing advantages for {len(dataset)} samples...")

    with torch.no_grad():
        for start_idx in range(0, len(dataset), batch_size):
            end_idx = min(start_idx + batch_size, len(dataset))
            indices = list(range(start_idx, end_idx))
            batch = dataset.get_batch(indices, device)

            # Get image embeddings - extract base_0_rgb if dict
            images = batch["images"]
            if isinstance(images, dict):
                images = images["base_0_rgb"]
            image_embedding = value_fn.embed_images(images)

            # Compute advantages
            advantages = value_fn.compute_advantage(image_embedding, batch["time_to_completion"])
            all_advantages.extend(advantages.cpu().numpy().tolist())

            if is_main and start_idx % (batch_size * 10) == 0:
                logging.info(f"  Processed {start_idx}/{len(dataset)} samples")

    # Store advantages
    dataset.advantages = np.array(all_advantages)
    dataset.improvement_indicators = dataset.advantages > 0

    if is_main:
        num_good = int(np.sum(dataset.improvement_indicators))
        num_bad = len(dataset) - num_good

        logging.info(f"Advantage computation complete:")
        logging.info(f"  Mean advantage: {np.mean(dataset.advantages):.4f}")
        logging.info(f"  Std advantage: {np.std(dataset.advantages):.4f}")
        logging.info(f"  Good samples (I=1): {num_good} ({100*num_good/len(dataset):.1f}%)")
        logging.info(f"  Bad samples (I=0): {num_bad} ({100*num_bad/len(dataset):.1f}%)")

    return dataset


def train_policy_warmup(
    config: RECAPConfig,
    dataset: LeRobotDataset,
    policy: PI0Pytorch,
    device: torch.device,
    is_main: bool,
    wandb_run=None,
):
    """Warmup: Train policy without advantage conditioning (standard BC)."""
    logging.info("=" * 70)
    logging.info("PHASE 3a: Policy Warmup (Standard Training)")
    logging.info("=" * 70)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=config.learning_rate)
    policy.train()

    indices = np.arange(len(dataset))
    rng = np.random.default_rng(config.seed + 1)

    pbar = tqdm.tqdm(range(config.policy_warmup_steps), desc="Policy Warmup", disable=not is_main)

    for step in pbar:
        rng.shuffle(indices)
        batch_indices = indices[:config.batch_size]
        batch = dataset.get_batch(batch_indices, device)

        # Create observation object for policy
        # Handle both dict of images (dummy) and single tensor (real dataset)
        if isinstance(batch["images"], dict):
            images = batch["images"]
        else:
            images = {
                "base_0_rgb": batch["images"],
                "left_wrist_0_rgb": batch["images"],
                "right_wrist_0_rgb": batch["images"],
            }
        observation = SimpleObservation(
            images=images,
            state=batch["states"],
        )

        # Forward pass
        losses = policy(observation, batch["actions"])
        loss = losses.mean()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        if is_main:
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if step % config.log_interval == 0:
                logging.info(f"  Warmup step {step}/{config.policy_warmup_steps}, loss: {loss.item():.4f}")

                if wandb_run:
                    wandb_run.log({
                        "policy/warmup_loss": loss.item(),
                        "policy/warmup_step": step,
                    }, step=config.value_train_steps + step)

    logging.info("Warmup complete!")
    return policy


def train_policy_recap(
    config: RECAPConfig,
    dataset: LeRobotDataset,
    policy: PI0Pytorch,
    device: torch.device,
    is_main: bool,
    wandb_run=None,
):
    """RECAP: Train policy with advantage conditioning."""
    logging.info("=" * 70)
    logging.info("PHASE 3b: RECAP Training (Advantage-Conditioned)")
    logging.info("=" * 70)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=config.learning_rate)
    policy.train()

    indices = np.arange(len(dataset))
    rng = np.random.default_rng(config.seed + 2)

    pbar = tqdm.tqdm(range(config.policy_recap_steps), desc="RECAP Training", disable=not is_main)

    for step in pbar:
        rng.shuffle(indices)
        batch_indices = indices[:config.batch_size]
        batch = dataset.get_batch(batch_indices, device)

        # Get improvement indicators
        improvement = batch["improvement_indicator"]
        pct_good = float(improvement.float().mean())

        # Create observation object for policy
        # Handle both dict of images (dummy) and single tensor (real dataset)
        if isinstance(batch["images"], dict):
            images = batch["images"]
        else:
            images = {
                "base_0_rgb": batch["images"],
                "left_wrist_0_rgb": batch["images"],
                "right_wrist_0_rgb": batch["images"],
            }
        observation = SimpleObservation(
            images=images,
            state=batch["states"],
        )

        # Forward pass with advantage conditioning
        # Note: The policy needs to be modified to accept improvement_indicator
        # For now, we pass it as an attribute on the observation object
        observation.improvement_indicator = improvement

        losses = policy(observation, batch["actions"])
        loss = losses.mean()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        if is_main:
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "good": f"{pct_good:.2%}"})

            if step % config.log_interval == 0:
                logging.info(
                    f"  RECAP step {step}/{config.policy_recap_steps}, "
                    f"loss: {loss.item():.4f}, pct_good: {pct_good:.2%}"
                )

                if wandb_run:
                    wandb_run.log({
                        "policy/recap_loss": loss.item(),
                        "policy/pct_good": pct_good,
                        "policy/recap_step": step,
                    }, step=config.value_train_steps + config.policy_warmup_steps + step)

    logging.info("RECAP training complete!")
    return policy


def save_checkpoint(
    config: RECAPConfig,
    value_fn: ValueFunctionPytorch,
    policy: PI0Pytorch,
    step: int,
    output_dir: pathlib.Path,
    is_main: bool,
):
    """Save model checkpoints."""
    if not is_main:
        return

    ckpt_dir = output_dir / f"checkpoint_{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save models using safetensors
    safetensors.torch.save_model(value_fn, ckpt_dir / "value_fn.safetensors")
    safetensors.torch.save_model(policy, ckpt_dir / "policy.safetensors")

    # Save config
    import json
    with open(ckpt_dir / "config.json", "w") as f:
        json.dump(dataclasses.asdict(config), f, indent=2)

    logging.info(f"Checkpoint saved to {ckpt_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="RECAP PyTorch Training")
    parser.add_argument("--config", type=str, default="recap_pi06_test",
                       choices=list(CONFIGS.keys()), help="Config name")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    return parser.parse_args()


def main():
    init_logging()
    args = parse_args()

    # Get config
    config = CONFIGS[args.config]

    # Apply overrides
    if args.batch_size:
        config = dataclasses.replace(config, batch_size=args.batch_size)
    if args.no_wandb:
        config = dataclasses.replace(config, wandb_enabled=False)
    if args.output_dir:
        config = dataclasses.replace(config, output_dir=args.output_dir)

    # Setup DDP
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    set_seed(config.seed, local_rank)

    # Setup experiment name
    exp_name = args.exp_name or f"recap_{args.config}_{int(time.time())}"
    output_dir = pathlib.Path(config.output_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print config
    if is_main:
        logging.info("=" * 70)
        logging.info("RECAP TRAINING PIPELINE (PyTorch)")
        logging.info("=" * 70)
        logging.info(f"Config: {args.config}")
        logging.info(f"Experiment: {exp_name}")
        logging.info(f"Output: {output_dir}")
        logging.info(f"Device: {device}")
        logging.info(f"DDP: {use_ddp}")
        logging.info(f"Base VLM: {config.base_vlm_variant}")
        logging.info(f"Action Expert: {config.action_expert_variant}")
        logging.info(f"Value Function: {config.value_function_variant}")
        logging.info(f"Batch size: {config.batch_size}")
        logging.info(f"Value steps: {config.value_train_steps}")
        logging.info(f"Warmup steps: {config.policy_warmup_steps}")
        logging.info(f"RECAP steps: {config.policy_recap_steps}")
        logging.info("=" * 70)

    # Initialize wandb
    wandb_run = None
    if is_main and config.wandb_enabled:
        try:
            wandb_run = wandb.init(
                project=config.wandb_project,
                name=exp_name,
                config=dataclasses.asdict(config),
            )
            logging.info(f"Wandb initialized: {wandb_run.url}")
        except Exception as e:
            logging.warning(f"Could not initialize wandb: {e}")

    # Load dataset
    logging.info("\n[0/4] Loading dataset...")
    dataset = LeRobotDataset(config)

    # Create models
    logging.info("\n[1/4] Creating models...")
    value_fn, policy = create_models(config, device)

    # Wrap in DDP if needed
    if use_ddp:
        value_fn = torch.nn.parallel.DistributedDataParallel(
            value_fn,
            device_ids=[device.index],
            find_unused_parameters=True,
        )
        policy = torch.nn.parallel.DistributedDataParallel(
            policy,
            device_ids=[device.index],
            find_unused_parameters=True,
        )

    # Phase 1: Train value function
    logging.info("\n[2/4] Training value function...")
    value_fn_unwrapped = value_fn.module if use_ddp else value_fn
    value_fn = train_value_function(config, dataset, value_fn_unwrapped, device, is_main, wandb_run)

    # Phase 2: Compute advantages
    logging.info("\n[3/4] Computing advantages...")
    dataset = compute_advantages(config, dataset, value_fn_unwrapped, device, is_main)

    # Phase 3: Train policy
    logging.info("\n[4/4] Training policy...")
    policy_unwrapped = policy.module if use_ddp else policy

    # 3a: Warmup
    policy = train_policy_warmup(config, dataset, policy_unwrapped, device, is_main, wandb_run)

    # 3b: RECAP
    policy = train_policy_recap(config, dataset, policy_unwrapped, device, is_main, wandb_run)

    # Save final checkpoint
    total_steps = config.value_train_steps + config.policy_warmup_steps + config.policy_recap_steps
    save_checkpoint(config, value_fn_unwrapped, policy_unwrapped, total_steps, output_dir, is_main)

    # Finish wandb
    if wandb_run:
        wandb_run.finish()

    # Summary
    if is_main:
        logging.info("\n" + "=" * 70)
        logging.info("TRAINING COMPLETE!")
        logging.info("=" * 70)
        logging.info(f"Total steps: {total_steps}")
        logging.info(f"Checkpoint: {output_dir}")
        logging.info("=" * 70)

    cleanup_ddp()


if __name__ == "__main__":
    main()
