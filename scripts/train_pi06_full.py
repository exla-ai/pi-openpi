#!/usr/bin/env python3
"""Full Pi0.6 Training Pipeline.

This script runs the complete Pi0.6 training pipeline:
1. Stage 1: Multi-task fine-tuning on diverse datasets (pi06_comprehensive)
2. Stage 2: Task-specific fine-tuning (pi06_aloha_sim or pi06_libero)
3. Stage 3: RECAP training for policy improvement

Usage:
    # Full pipeline
    python scripts/train_pi06_full.py --exp_name pi06_v1

    # Skip to stage 2 (if stage 1 already done)
    python scripts/train_pi06_full.py --exp_name pi06_v1 --start_stage 2 \
        --stage1_checkpoint ./checkpoints/pi06_comprehensive/pi06_v1_stage1/100000/params

    # Only run specific stage
    python scripts/train_pi06_full.py --exp_name pi06_v1 --only_stage 1
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True)
        logger.info(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} failed with return code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Full Pi0.6 Training Pipeline")
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name")
    parser.add_argument("--start_stage", type=int, default=1, choices=[1, 2, 3],
                        help="Stage to start from (default: 1)")
    parser.add_argument("--only_stage", type=int, default=None, choices=[1, 2, 3],
                        help="Only run specific stage")
    parser.add_argument("--stage1_checkpoint", type=str, default=None,
                        help="Checkpoint path from stage 1 (required if starting from stage 2)")
    parser.add_argument("--stage2_checkpoint", type=str, default=None,
                        help="Checkpoint path from stage 2 (required if starting from stage 3)")
    parser.add_argument("--task", type=str, default="aloha_sim",
                        choices=["aloha_sim", "libero", "aloha_real"],
                        help="Task for stage 2 fine-tuning")
    parser.add_argument("--fsdp_devices", type=int, default=8,
                        help="Number of GPUs for FSDP")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without running")
    args = parser.parse_args()

    checkpoint_dir = Path("./checkpoints")

    # Determine stages to run
    if args.only_stage:
        stages = [args.only_stage]
    else:
        stages = list(range(args.start_stage, 4))

    logger.info("=" * 60)
    logger.info("Pi0.6 Full Training Pipeline")
    logger.info("=" * 60)
    logger.info(f"Experiment: {args.exp_name}")
    logger.info(f"Stages to run: {stages}")
    logger.info(f"Task: {args.task}")
    logger.info("=" * 60)

    # Stage 1: Multi-task fine-tuning
    if 1 in stages:
        logger.info("")
        logger.info("=" * 60)
        logger.info("STAGE 1: Multi-task Fine-tuning (pi06_comprehensive)")
        logger.info("=" * 60)

        stage1_name = f"{args.exp_name}_stage1"
        cmd = [
            "python", "scripts/train.py", "pi06_comprehensive",
            "--exp_name", stage1_name,
            "--fsdp_devices", str(args.fsdp_devices),
        ]

        if args.dry_run:
            logger.info(f"[DRY RUN] Would run: {' '.join(cmd)}")
        else:
            if not run_command(cmd, "Stage 1: Multi-task fine-tuning"):
                logger.error("Stage 1 failed. Exiting.")
                sys.exit(1)

        args.stage1_checkpoint = str(
            checkpoint_dir / "pi06_comprehensive" / stage1_name / "100000" / "params"
        )

    # Stage 2: Task-specific fine-tuning
    if 2 in stages:
        if not args.stage1_checkpoint and args.start_stage <= 1:
            args.stage1_checkpoint = str(
                checkpoint_dir / "pi06_comprehensive" / f"{args.exp_name}_stage1" / "100000" / "params"
            )

        logger.info("")
        logger.info("=" * 60)
        logger.info(f"STAGE 2: Task-specific Fine-tuning (pi06_{args.task})")
        logger.info("=" * 60)

        stage2_name = f"{args.exp_name}_stage2"
        config_name = f"pi06_{args.task}"

        cmd = [
            "python", "scripts/train.py", config_name,
            "--exp_name", stage2_name,
            "--fsdp_devices", str(args.fsdp_devices),
        ]

        # Add checkpoint from stage 1 if available
        if args.stage1_checkpoint and Path(args.stage1_checkpoint).exists():
            cmd.extend(["--weight_loader.params_path", args.stage1_checkpoint])
            logger.info(f"Loading weights from: {args.stage1_checkpoint}")

        if args.dry_run:
            logger.info(f"[DRY RUN] Would run: {' '.join(cmd)}")
        else:
            if not run_command(cmd, f"Stage 2: {args.task} fine-tuning"):
                logger.error("Stage 2 failed. Exiting.")
                sys.exit(1)

        # Determine checkpoint path
        step_map = {"aloha_sim": "30000", "libero": "20000", "aloha_real": "20000"}
        args.stage2_checkpoint = str(
            checkpoint_dir / config_name / stage2_name / step_map[args.task] / "params"
        )

    # Stage 3: RECAP training
    if 3 in stages:
        if not args.stage2_checkpoint and args.start_stage <= 2:
            step_map = {"aloha_sim": "30000", "libero": "20000", "aloha_real": "20000"}
            args.stage2_checkpoint = str(
                checkpoint_dir / f"pi06_{args.task}" / f"{args.exp_name}_stage2" / step_map[args.task]
            )

        logger.info("")
        logger.info("=" * 60)
        logger.info("STAGE 3: RECAP Training")
        logger.info("=" * 60)

        stage3_name = f"{args.exp_name}_recap"
        recap_config = f"recap_{args.task}"

        cmd = [
            "python", "scripts/train_recap_full.py",
            "--config", recap_config,
            "--experiment_name", stage3_name,
        ]

        if args.stage2_checkpoint:
            cmd.extend(["--resume_from", args.stage2_checkpoint])
            logger.info(f"Resuming from: {args.stage2_checkpoint}")

        if args.dry_run:
            logger.info(f"[DRY RUN] Would run: {' '.join(cmd)}")
        else:
            if not run_command(cmd, "Stage 3: RECAP training"):
                logger.error("Stage 3 failed. Exiting.")
                sys.exit(1)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Pi0.6 Training Pipeline Complete!")
    logger.info("=" * 60)

    # Print final checkpoint location
    final_checkpoint = checkpoint_dir / "recap_full" / f"{args.exp_name}_recap" / "final"
    logger.info(f"Final checkpoint: {final_checkpoint}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Run benchmarks: python scripts/benchmark_recap.py --compare-paper")
    logger.info("2. Upload to HuggingFace: python scripts/upload_to_huggingface.py")


if __name__ == "__main__":
    main()
