# FLA - Finetune VLAs

**Fine-tune Vision-Language-Action models on any robotics dataset.**

This repository provides infrastructure for fine-tuning [Physical Intelligence's Pi0/Pi0.5](https://www.physicalintelligence.company/) VLA models on robotics manipulation datasets. It supports any dataset you can package in the [LeRobot](https://huggingface.co/lerobot) format (including all major open-source LeRobot datasets).

---

## Why This Repository?

| Feature | Description |
|---------|-------------|
| **Any Dataset** | Train on DROID, LIBERO, Open X-Embodiment, ALOHA, or your own LeRobot-format data |
| **Cross-Embodiment** | Single model works across different robot types (7-DOF, 14-DOF, bimanual) |
| **Efficient Training** | Frozen backbone fine-tuning: 300M trainable params, 4-8 hours on A100 |
| **Production Ready** | Docker-based evaluation, model serving, and deployment |

---

## Evaluation Quickstart

```bash
# ALOHA sim (local)
fla-benchmark \
  --suite aloha_sim \
  --checkpoint-dir ./checkpoints/pi06_multi/pi06_multi_v1/30000 \
  --config pi06_multi \
  --num-episodes 50

# Dataset eval (offline MSE/RMSE)
fla-benchmark \
  --suite dataset \
  --checkpoint-dir ./checkpoints/your_config/your_exp/30000 \
  --config your_config \
  --repo-ids your-org/your_dataset \
  --max-samples 1024

# Full eval (ALOHA + LIBERO + dataset)
fla-benchmark \
  --suite full \
  --checkpoint-dir ./checkpoints/your_config/your_exp/30000 \
  --config your_config \
  --repo-ids your-org/your_dataset \
  --output evaluation_results.json
```

---

## VLA Fine-Tuning Methods (Research & Open-Source)

| Method | Key Idea | Action Representation | Compute Footprint | Reference | FLA Support |
|--------|----------|-----------------------|-------------------|-----------|-------------|
| **PEFT (LoRA/QLoRA)** | Parameter-efficient adapters for VLM/VLA fine-tuning | Model-specific | Low | [LoRA paper](https://arxiv.org/abs/2106.09685) | External |
| **OpenVLA Full** | Full-parameter fine-tuning of OpenVLA | Discrete action tokens | High | [OpenVLA repo](https://github.com/openvla/openvla) | External |
| **OpenVLA‑OFT** | Optimized fine-tuning: action chunking + continuous actions + L1 regression | Continuous actions | Medium | [OFT paper](https://arxiv.org/abs/2502.19645) • [OFT site](https://openvla-oft.github.io/) | External |
| **Octo Fine‑Tuning** | Diffusion-policy fine-tuning for generalist robot policies | Diffusion actions | Medium | [Octo repo](https://github.com/octo-models/octo) | External |
| **RT‑2 Co‑Fine‑Tuning** | Co-train on web VLM data + robot data, actions as text tokens | Action tokens | High | [RT‑2 project](https://robotics-transformer2.github.io/) | External |
| **Knowledge Insulation (PI)** | Insulate VLM backbone while training action expert on continuous actions | Continuous actions | Medium | [Knowledge Insulation](https://www.physicalintelligence.company/research/knowledge_insulation) | Supported (Pi0/Pi0.5 frozen-backbone recipes) |

**Legend:**  
**Supported** = implemented in FLA • **External** = reference implementation elsewhere

---

## Supported Datasets

We support all major robotics datasets in [LeRobot format](https://huggingface.co/lerobot):

### Large-Scale Datasets

| Dataset | Episodes | Frames | Robot Types | Source |
|---------|----------|--------|-------------|--------|
| [**DROID**](https://huggingface.co/datasets/lerobot/droid_1.0.1) | 92,223 | 27M | Franka | [droid-dataset.github.io](https://droid-dataset.github.io/) |
| [**Open X-Embodiment**](https://huggingface.co/collections/lerobot/open-x-embodiment) | 1M+ | - | 22 robot types | [robotics-transformer-x.github.io](https://robotics-transformer-x.github.io/) |
| [**RoboMIND**](https://x-humanoid-robomind.github.io/) | 107k | - | Franka, UR5e, AgileX, Humanoid | RSS 2025 |

### Benchmark Datasets

| Dataset | Tasks | Episodes | Description |
|---------|-------|----------|-------------|
| [**LIBERO**](https://huggingface.co/datasets/lerobot/libero) | 130+ | 50/task | Diverse manipulation benchmark |
| [**LIBERO-10**](https://huggingface.co/datasets/lerobot/libero_10) | 10 | 50/task | Core benchmark subset |

### ALOHA Datasets (Bimanual)

| Dataset | Episodes | Task |
|---------|----------|------|
| [lerobot/aloha_sim_transfer_cube_human](https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_human) | 50 | Cube transfer |
| [lerobot/aloha_sim_insertion_human](https://huggingface.co/datasets/lerobot/aloha_sim_insertion_human) | 50 | Peg insertion |
| [lerobot/aloha_static_towel](https://huggingface.co/datasets/lerobot/aloha_static_towel) | - | Towel folding |
| [lerobot/aloha_mobile_cabinet](https://huggingface.co/datasets/lerobot/aloha_mobile_cabinet) | - | Mobile manipulation |

### Quick Dataset Download

```bash
# Download commonly used datasets
python scripts/download_datasets.py

# Or download specific datasets
python -c "from lerobot.common.datasets.lerobot_dataset import LeRobotDataset; LeRobotDataset('lerobot/libero_10')"
```

---

## Objective

The goal of this repository is to provide a **reliable VLA fine-tuning pipeline** for robotics research:

1. **Fine-tune on any dataset** - Works with all LeRobot-format datasets
2. **Cross-embodiment support** - Train once, deploy on multiple robot types
3. **Verified benchmarks** - Honest performance numbers, not just paper claims
4. **Production infrastructure** - Docker serving, evaluation, deployment

---

## Fine-Tuning on Your Data

This repository enables fine-tuning Pi0 models on **any robot demonstration data**. The workflow:

```
Your Robot Data → LeRobot Format → Fine-tune Pi0 → Policy for Your Task
```

### What You Need to Provide

| Data | Format | Example |
|------|--------|---------|
| **Camera images** | RGB frames | `(N, 480, 640, 3)` |
| **Robot state** | Joint positions | `(N, 7)` for single arm, `(N, 14)` for bimanual |
| **Actions** | Target joint positions | Same dim as state |
| **Task prompt** | Text | `"Pick up the red block"` |

### Fine-Tuning Process

```bash
# 1. Convert your data to LeRobot format
python scripts/convert_to_lerobot.py --input your_data/ --output lerobot_dataset/

# 2. Upload to HuggingFace (or use locally)
huggingface-cli upload your-org/my-robot-dataset ./lerobot_dataset

# 3. Fine-tune (recipe CLI)
python scripts/finetune.py \
  --recipe pi0_frozen_backbone \
  --repo-ids your-org/my-robot-dataset \
  --repo-id-to-prompt "your-org/my-robot-dataset:Pick up the red block" \
  --action-dim 7 \
  --exp-name v1

To fine-tune Pi0.5 specifically, use the pi05 recipes (e.g. `pi05_frozen_backbone`, `pi05_lora`, `pi05_full_finetune`).

# Optional: compute norm stats (advanced)
# python scripts/compute_norm_stats.py <config_name>

# 4. Deploy
python scripts/serve_policy.py policy:checkpoint --policy.dir ./checkpoints/...
```

### Recipe CLI (Recommended)

```bash
# List available recipes
python scripts/finetune.py --list
```

Multi‑dataset fine‑tuning (each dataset gets its own prompt):

```bash
python scripts/finetune.py \
  --recipe pi0_frozen_backbone \
  --repo-ids your-org/isaac_franka_cabinet your-org/isaac_franka_lift \
  --repo-id-to-prompt "your-org/isaac_franka_cabinet:Open the cabinet drawer" \
  --repo-id-to-prompt "your-org/isaac_franka_lift:Lift the object" \
  --action-dim 9 \
  --exp-name multi_task_v1
```

Fine‑tuning different base models: set `--base-model pi0` or `--base-model pi05` and optionally pass `--checkpoint-path` to point at custom weights (e.g., `gs://openpi-assets/checkpoints/pi0_base/params` or `gs://openpi-assets/checkpoints/pi05_base/params`).

Quick smoke test (no base checkpoint download):

```bash
python scripts/finetune.py \
  --recipe pi0_frozen_backbone \
  --init-from scratch \
  --paligemma-variant gemma_300m \
  --action-expert-variant gemma_300m \
  --repo-ids your-org/my-robot-dataset \
  --repo-id-to-prompt "your-org/my-robot-dataset:Pick up the red block" \
  --action-dim 7 \
  --exp-name smoke_test
```

Python API:

```python
from fla.finetune import build_train_config, RecipeOverrides

config = build_train_config(
    "pi0_frozen_backbone",
    RecipeOverrides(
        repo_ids=("your-org/my-robot-dataset",),
        repo_id_to_prompt={"your-org/my-robot-dataset": "Pick up the red block"},
        base_model="pi05",
        action_dim=7,
        exp_name="v1",
    ),
)
```

### Fresh Dataset Example (Isaac Lab → LeRobot)

This example records **fresh** Isaac Lab trajectories so the fine‑tune data is guaranteed not to overlap with pretraining.

```bash
# 1) Collect two small, fresh datasets
python scripts/isaaclab_data_collection.py \
  --task Isaac-Franka-Cabinet-Direct-v0 \
  --num_episodes 30 \
  --dataset_name franka_cabinet \
  --format numpy \
  --output_dir data/isaaclab

python scripts/isaaclab_data_collection.py \
  --task Isaac-Lift-Franka-v0 \
  --num_episodes 30 \
  --dataset_name franka_lift \
  --format numpy \
  --output_dir data/isaaclab

# 2) Convert to LeRobot format
python scripts/isaaclab_to_lerobot.py \
  --input-dir data/isaaclab/franka_cabinet_numpy \
  --repo-id your-org/isaac_franka_cabinet \
  --task "Open the cabinet drawer"

python scripts/isaaclab_to_lerobot.py \
  --input-dir data/isaaclab/franka_lift_numpy \
  --repo-id your-org/isaac_franka_lift \
  --task "Lift the object"

# 3) Multi-dataset fine-tune
python scripts/finetune.py \
  --recipe pi0_frozen_backbone \
  --repo-ids your-org/isaac_franka_cabinet your-org/isaac_franka_lift \
  --repo-id-to-prompt "your-org/isaac_franka_cabinet:Open the cabinet drawer" \
  --repo-id-to-prompt "your-org/isaac_franka_lift:Lift the object" \
  --action-dim 9 \
  --exp-name isaaclab_demo
```

Example training output (H100 smoke run):

```text
Step 0: grad_norm=15.5867, loss=3.1071, param_norm=829.2279, task_loss=3.1071
...
```

If Isaac Lab isn’t installed, `scripts/isaaclab_data_collection.py` will fall back to a mock collector. For real trajectories, follow the setup in `docs/isaac_lab_setup.md`.

**Overlap note:** If your fine‑tuning dataset overlaps with pretraining (same robot/task distribution), gains may be smaller. Fresh data or a held‑out split yields clearer improvements.

### Benchmark Evaluation (ALOHA Sim)

Use the built-in benchmark runner to evaluate checkpoints on standard ALOHA sim tasks:

```bash
fla-benchmark \
  --suite aloha_sim \
  --checkpoint-dir ./checkpoints/pi06_multi/pi06_multi_v1/30000 \
  --config pi06_multi \
  --num-episodes 50 \
  --output evaluation_results.json
```

Run a single task:

```bash
fla-benchmark \
  --suite aloha_sim \
  --task gym_aloha/AlohaTransferCube-v0 \
  --checkpoint-dir ./checkpoints/pi06_multi/pi06_multi_v1/30000 \
  --config pi06_multi \
  --num-episodes 50
```

Note: `gym-aloha` is required for these tasks (installed by default via `pyproject.toml`).

Example result (H100, 2026‑02‑04, **short run**, 1,000 training steps, 10 episodes, `MUJOCO_GL=osmesa`):

```text
Task: gym_aloha/AlohaTransferCube-v0
Checkpoint: ./checkpoints/pi0_frozen_backbone_aloha_1k/aloha_1k/999
Success Rate: 10.0%
Avg Reward: 0.1
```

This is still a quick sanity check only. For meaningful success rates, run longer training and evaluate with more episodes.

### Benchmark Evaluation (LIBERO)

Install LIBERO dependencies and run evaluation locally:

```bash
uv sync --group libero

fla-benchmark \
  --suite libero \
  --libero-suite libero_spatial \
  --checkpoint-dir ./checkpoints/pi05_libero/pi05_libero_v1/30000 \
  --config pi05_libero \
  --libero-num-trials 50
```

### Dataset Evaluation (Offline)

Evaluate action prediction error on a LeRobot dataset (works for DROID, ALOHA, LIBERO, etc.):

```bash
fla-benchmark \
  --suite dataset \
  --checkpoint-dir ./checkpoints/your_config/your_exp/30000 \
  --config your_config \
  --repo-ids your-org/your_dataset \
  --max-samples 1024
```

If you used a recipe-based fine‑tune, you can evaluate without a config file by passing `--recipe`:

```bash
fla-benchmark \
  --suite dataset \
  --checkpoint-dir ./checkpoints/pi0_frozen_backbone/isaaclab_demo/4 \
  --recipe pi0_frozen_backbone \
  --repo-ids your-org/your_dataset \
  --default-prompt "Complete the task" \
  --model-action-dim 9 \
  --model-action-horizon 20 \
  --max-samples 1024
```

Example result (H100, 2026‑02‑04, offline dataset eval on 8 samples):

```text
Dataset: fla/isaac_franka_cabinet + fla/isaac_franka_lift
Checkpoint: ./checkpoints/pi0_frozen_backbone/isaaclab_demo/4
Metrics: MSE=3.010680, RMSE=1.735131, L1=1.374045
```

### Full Suite

Run ALOHA + LIBERO + dataset eval in one go:

```bash
fla-benchmark \
  --suite full \
  --checkpoint-dir ./checkpoints/pi06_multi/pi06_multi_v1/30000 \
  --config pi06_multi \
  --repo-ids lerobot/droid_1.0.1 \
  --output evaluation_results.json
```

### Core Algorithm: Flow Matching

The Pi0 model uses **flow matching** for action generation:

```
Training:
1. Sample (observation, action) pairs from demonstrations
2. Add noise: x_t = t * noise + (1-t) * action
3. Train to predict velocity: v_t = noise - action
4. Loss = MSE(predicted_v_t, true_v_t)

Inference:
1. Start with noise x_1
2. Denoise iteratively: x_{t+dt} = x_t + v_t * dt
3. After 10 steps → clean action trajectory
```

### Architecture: Frozen Backbone

```
┌─────────────────────────────────────────────────┐
│  PaliGemma VLM (2.4B params) - FROZEN           │
│  ├── SigLIP Vision Encoder (400M)               │
│  └── Gemma 2B Language Model                    │
└─────────────────────────────────────────────────┘
                      ↓ stop_gradient()
┌─────────────────────────────────────────────────┐
│  Action Expert (300M params) - TRAINABLE        │
│  ├── Action input projection                    │
│  ├── Time embedding MLP                         │
│  ├── Gemma 300M transformer                     │
│  └── Action output projection                   │
└─────────────────────────────────────────────────┘
```

**Why frozen backbone works:**
- VLM already understands images + language
- Only need to learn action mapping (~300M params)
- Reduces GPU memory: 70GB → 33GB
- Trains 5-10x faster

### Data Requirements

| Approach | Min Episodes | Typical | Notes |
|----------|-------------|---------|-------|
| Frozen backbone | 50 | 100-500 | Leverages pretrained VLM |
| Full fine-tuning | 500 | 1000+ | Needs more data |

---

## Cross-Embodiment Training

Train a single model that works across **multiple robot types** with different action dimensions.

### How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                    Cross-Embodiment Model                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input: Image + Language Prompt (specifies robot + task)    │
│                                                              │
│  ┌─────────────────┐  ┌─────────────────┐                   │
│  │ ALOHA bimanual  │  │ Single-arm      │                   │
│  │ 14 DOF actions  │  │ 7 DOF + padding │                   │
│  └─────────────────┘  └─────────────────┘                   │
│           ↓                    ↓                             │
│  ┌─────────────────────────────────────────┐                │
│  │      Unified 14-dim Action Space        │                │
│  │  (smaller robots zero-padded)           │                │
│  └─────────────────────────────────────────┘                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Key Techniques

| Technique | Description |
|-----------|-------------|
| **Action Padding** | Smaller action spaces (7 DOF) padded to max (14 DOF) |
| **Embodiment Prompts** | Language specifies robot type: "ALOHA bimanual robot: ..." |
| **Multi-Dataset Training** | Combine data from different robots in one training run |

### Available Configs

```bash
# Cross-embodiment: ALOHA (14 DOF) + LIBERO (7 DOF)
python scripts/train.py pi06_cross_embodiment --exp-name v1

# Just ALOHA variants (simpler, single embodiment)
python scripts/train.py pi06_cross_embodiment_aloha --exp-name v1
```

### Adding Your Own Robot

To add a new robot type to cross-embodiment training:

1. **Convert data to LeRobot format** with your action dimension
2. **Add to config** with embodiment-aware prompts:

```python
data=MultiLeRobotDataConfig(
    repo_ids=(
        "lerobot/aloha_sim_transfer_cube_human",  # 14 DOF
        "your-org/your-single-arm-robot",          # 7 DOF (will be padded)
    ),
    repo_id_to_prompt={
        "lerobot/aloha_sim_transfer_cube_human": "ALOHA bimanual: Transfer the cube",
        "your-org/your-single-arm-robot": "UR5 single arm: Pick up the object",
    },
),
```

3. **Set action_dim to maximum** across all robots (e.g., 14 for ALOHA)

### Inference with Cross-Embodiment Model

```python
# Specify embodiment in prompt
actions = policy.infer({
    "image": camera_image,
    "state": robot_state,  # Padded to 14 dim if needed
    "prompt": "ALOHA bimanual robot: Pick up the red block"
})

# For 7-DOF robot, only use first 7 action dimensions
single_arm_actions = actions[:, :7]
```

---

## What You Can Use This Repository For Today

### ✅ Production Ready

| Use Case | Model | Performance | How to Use |
|----------|-------|-------------|------------|
| **LIBERO benchmark tasks** | π₀.₅ (PI's checkpoint) | **96.85%** success | [LIBERO Guide](examples/libero/README.md) |
| **DROID robot manipulation** | π₀.₅-DROID | State-of-the-art | [DROID Guide](examples/droid/README.md) |
| **ALOHA towel folding** | π₀-ALOHA-towel | Works zero-shot | [ALOHA Guide](examples/aloha_real/README.md) |

### ⚠️ Experimental (Mixed Results)

| Use Case | Model | Performance | Notes |
|----------|-------|-------------|-------|
| **Custom task fine-tuning** | pi06 configs (frozen backbone) | 38-66% on ALOHA sim | Task-dependent, see benchmarks below |
| **RECAP training** | Experimental | Not tested | Paper claims 60%→85%, unverified |

---

## Verified Benchmark Results

These are actual evaluation results from running the models in this repository:

### ALOHA Simulation (50 episodes each)

| Task | Model | Success Rate | Paper Baseline | Status |
|------|-------|--------------|----------------|--------|
| **Insertion** | pi06_multi @ 30k steps | **66%** | 50% | ✅ Exceeds baseline |
| **Transfer Cube** | pi06_multi @ 30k steps | **38%** | 60% | ❌ Below baseline |

### LIBERO Benchmark (Physical Intelligence's checkpoint)

| Task Suite | Success Rate |
|------------|--------------|
| LIBERO Spatial | 98.8% |
| LIBERO Object | 98.2% |
| LIBERO Goal | 98.0% |
| LIBERO-10 | 92.4% |
| **Average** | **96.85%** |

> **Interpretation**: The frozen backbone approach works well for some tasks (Insertion: 66%) but not others (Transfer Cube: 38%). If you need consistent high performance, use Physical Intelligence's official checkpoints.

---

## Quick Start

### Option 1: Use Pre-trained Models (Recommended)

```python
from fla.training import config as _config
from fla.policies import policy_config

# Load Pi0.5 for LIBERO
config = _config.get_config("pi05_libero")
policy = policy_config.create_trained_policy(
    config,
    "gs://openpi-assets/checkpoints/pi05_libero"
)

# Run inference
actions = policy.infer({
    "image": image,
    "state": robot_state,
    "prompt": "pick up the red block"
})
```

### Option 2: Fine-tune on Your Data

```bash
# 1. Prepare your dataset in LeRobot format
# 2. Train with frozen backbone (efficient)
python scripts/train.py pi06_aloha_sim --exp-name my_experiment

# 3. Evaluate (see Evaluation section below for Docker-based approach)
python scripts/evaluate_aloha_sim.py \
    --checkpoint_dir ./checkpoints/pi06_aloha_sim/my_experiment/30000 \
    --config pi06_aloha_sim \
    --task gym_aloha/AlohaTransferCube-v0
```

See [Fine-Tuning Guide](#fine-tuning-on-your-own-tasks) for details.

---

## Next Steps & Roadmap

### If You Want High Performance Today
1. **Use Physical Intelligence's official checkpoints** - They work and are verified
2. **Start with LIBERO or DROID** - Best documented, best results

### If You Want to Fine-tune on Custom Tasks
1. **Start with pi06_aloha_sim config** - Simplest setup
2. **Expect 40-70% success** - Results vary by task
3. **Run your own evaluation** - Don't trust paper numbers

### What Needs Community Contribution
| Area | Current State | What's Needed |
|------|---------------|---------------|
| **Transfer Cube performance** | 38% (below baseline) | Hyperparameter tuning, more training |
| **RECAP training** | Code exists, untested | Someone to run and verify |
| **Gemma 3 4B configs** | Untested | Large-scale training experiments |
| **More benchmarks** | Only ALOHA sim tested | LIBERO with custom models, real robot |

---

## Model Overview

| Model | Parameters | Best For | Checkpoint |
|-------|------------|----------|------------|
| **π₀.₅** | 2.7B | Production use, LIBERO, DROID | `gs://openpi-assets/checkpoints/pi05_base` |
| **π₀.₅-DROID** | 2.7B | DROID robot platform | `gs://openpi-assets/checkpoints/pi05_droid` |
| **π₀.₅-LIBERO** | 2.7B | LIBERO benchmark | `gs://openpi-assets/checkpoints/pi05_libero` |
| **pi06 configs** | 2.7B | Custom fine-tuning (experimental) | Train your own |

---

## Honest Assessment

### What Works Well
- ✅ **Pi0.5 on LIBERO**: 96.85% - state-of-the-art
- ✅ **Pi0.5 on DROID**: Best open-source generalist policy
- ✅ **Frozen backbone training**: Runs, produces working models
- ✅ **Inference pipeline**: Fast, reliable

### What Doesn't Work Well (Yet)
- ❌ **Transfer Cube with frozen backbone**: 38% vs 60% baseline
- ❌ **RECAP training**: Unverified claims (60%→85%)
- ❌ **Consistent multi-task performance**: Task-dependent results

### What We Don't Know
- ❓ Whether more training improves Transfer Cube
- ❓ Whether RECAP actually achieves paper claims
- ❓ How Gemma 3 4B configs perform

---

## Updates

See the [upstream Physical Intelligence repository](https://github.com/Physical-Intelligence/openpi) for official updates. 


## Requirements

To run the models in this repository, you will need an NVIDIA GPU with at least the following specifications. These estimations assume a single GPU, but you can also use multiple GPUs with model parallelism to reduce per-GPU memory requirements by configuring `fsdp_devices` in the training config. Please also note that the current training script does not yet support multi-node training.

| Mode               | Memory Required | Example GPU        |
| ------------------ | --------------- | ------------------ |
| Inference          | > 8 GB          | RTX 4090           |
| Fine-Tuning (LoRA) | > 22.5 GB       | RTX 4090           |
| Fine-Tuning (Full) | > 70 GB         | A100 (80GB) / H100 |

The repo has been tested with Ubuntu 22.04, we do not currently support other operating systems.

## Installation

FLA is a Python library. Install it into a virtual environment and import `fla` in your own training scripts, or use the included CLI scripts.

### Option A: Install as a Python library (recommended)

We use `uv` to manage Python dependencies (fast and reproducible). The simplest way to install FLA is directly from Git:

```bash
uv pip install "git+https://github.com/your-org/fla.git"
```

To pin to a tag or commit:

```bash
uv pip install "git+https://github.com/your-org/fla.git@v0.1.0"
```

Quick sanity check:

```bash
python -c "import fla; print('FLA import OK')"
```

### Option B: Clone for development

When cloning this repo, make sure to update submodules:

```bash
git clone --recurse-submodules https://github.com/your-org/fla.git

# Or if you already cloned the repo:
git submodule update --init --recursive
```

Then install the editable package using `uv`:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

NOTE: `GIT_LFS_SKIP_SMUDGE=1` is needed to pull LeRobot as a dependency.

### Option C: Docker

As an alternative, we provide Docker-based setup instructions in `docs/docker.md`. This can simplify CUDA/driver mismatches on some systems.




## Model Checkpoints

### Base Models
We provide multiple base VLA model checkpoints. These checkpoints have been pre-trained on 10k+ hours of robot data, and can be used for fine-tuning.

| Model        | Use Case    | Description                                                                                                 | Checkpoint Path                                |
| ------------ | ----------- | ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| $\pi_0$      | Fine-Tuning | Base [π₀ model](https://www.physicalintelligence.company/blog/pi0) for fine-tuning                | `gs://openpi-assets/checkpoints/pi0_base`      |
| $\pi_0$-FAST | Fine-Tuning | Base autoregressive [π₀-FAST model](https://www.physicalintelligence.company/research/fast) for fine-tuning | `gs://openpi-assets/checkpoints/pi0_fast_base` |
| $\pi_{0.5}$    | Fine-Tuning | Base [π₀.₅ model](https://www.physicalintelligence.company/blog/pi05) for fine-tuning    | `gs://openpi-assets/checkpoints/pi05_base`      |
| pi06 configs   | Fine-Tuning | Experimental: π₀.₅ with frozen backbone for efficient fine-tuning. **Not the Pi0.6 paper** (which uses RECAP). | N/A (train your own) |

### Fine-Tuned Models
We also provide "expert" checkpoints for various robot platforms and tasks. These models are fine-tuned from the base models above and intended to run directly on the target robot. These may or may not work on your particular robot. Since these checkpoints were fine-tuned on relatively small datasets collected with more widely available robots, such as ALOHA and the DROID Franka setup, they might not generalize to your particular setup, though we found some of these, especially the DROID checkpoint, to generalize quite broadly in practice.

| Model                    | Use Case    | Description                                                                                                                                                                                              | Checkpoint Path                                       |
| ------------------------ | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| $\pi_0$-FAST-DROID       | Inference   | $\pi_0$-FAST model fine-tuned on the [DROID dataset](https://droid-dataset.github.io/): can perform a wide range of simple table-top manipulation tasks 0-shot in new scenes on the DROID robot platform | `gs://openpi-assets/checkpoints/pi0_fast_droid`       |
| $\pi_0$-DROID            | Fine-Tuning | $\pi_0$ model fine-tuned on the [DROID dataset](https://droid-dataset.github.io/): faster inference than $\pi_0$-FAST-DROID, but may not follow language commands as well                                | `gs://openpi-assets/checkpoints/pi0_droid`            |
| $\pi_0$-ALOHA-towel      | Inference   | $\pi_0$ model fine-tuned on internal [ALOHA](https://tonyzhaozh.github.io/aloha/) data: can fold diverse towels 0-shot on ALOHA robot platforms                                                          | `gs://openpi-assets/checkpoints/pi0_aloha_towel`      |
| $\pi_0$-ALOHA-tupperware | Inference   | $\pi_0$ model fine-tuned on internal [ALOHA](https://tonyzhaozh.github.io/aloha/) data: can unpack food from a tupperware container                                                                                                             | `gs://openpi-assets/checkpoints/pi0_aloha_tupperware` |
| $\pi_0$-ALOHA-pen-uncap  | Inference   | $\pi_0$ model fine-tuned on public [ALOHA](https://dit-policy.github.io/) data: can uncap a pen                                                                                                          | `gs://openpi-assets/checkpoints/pi0_aloha_pen_uncap`  |
| $\pi_{0.5}$-LIBERO      | Inference   | $\pi_{0.5}$ model fine-tuned for the [LIBERO](https://libero-project.github.io/datasets) benchmark: gets state-of-the-art performance (see [LIBERO README](examples/libero/README.md)) | `gs://openpi-assets/checkpoints/pi05_libero`      |
| $\pi_{0.5}$-DROID      | Inference / Fine-Tuning | $\pi_{0.5}$ model fine-tuned on the [DROID dataset](https://droid-dataset.github.io/) with [knowledge insulation](https://www.physicalintelligence.company/research/knowledge_insulation): fast inference and good language-following | `gs://openpi-assets/checkpoints/pi05_droid`      |


By default, checkpoints are automatically downloaded from `gs://openpi-assets` and are cached in `~/.cache/fla` when needed. You can overwrite the download path by setting the `FLA_DATA_HOME` environment variable (or `OPENPI_DATA_HOME` for backward compatibility).


## Efficient Fine-Tuning with Frozen Backbone (pi06 configs)

> **Important Clarification**: The `pi06_*` configs in this repo are **NOT the same as the Pi0.6 paper**.
> - **This repo's pi06 configs**: Pi0.5 + frozen VLM backbone (behavior cloning only)
> - **The Pi0.6 paper**: Uses RECAP (advantage-conditioned RL) for iterative improvement
>
> For the actual RECAP method from the paper, see [RECAP Training](#recap-training-experimental).

The `pi06_*` configs provide efficient fine-tuning by freezing the VLM backbone:

```bash
# Train on ALOHA sim datasets with frozen Pi0.5 backbone
python scripts/train.py pi06_multi --exp_name pi06_v1 --fsdp_devices 8
```

This approach:
- Uses **pretrained Pi0.5 weights** (already trained on 20M+ robot samples)
- **Freezes the VLM backbone** - only trains the 300M action expert
- Works well with **limited data** (~100k frames)
- Trains in **~1 day on 8x A100**

**Note**: This is standard behavior cloning. No published benchmark results are available for these configs.

### Architecture Options

| Variant | VLM Backbone | Action Expert | Data Required | Notes |
|---------|--------------|---------------|---------------|-------|
| **Pi0.5 frozen** | Gemma 2B (frozen) | 300M | ~100k frames | Recommended for most users |
| **Gemma 3 configs** | Gemma 3 4B | 860M | Millions | Experimental, unverified |

### Quick Start

```bash
# 1. Download datasets
python scripts/download_datasets.py --skip_oxe

# 2. Train (recommended config)
python scripts/train.py pi06_multi --exp_name pi06_v1 --fsdp_devices 8

# 3. After training, upload to HuggingFace
python scripts/upload_to_huggingface.py \
    --checkpoint_path ./checkpoints/pi06_multi/pi06_v1/30000/params \
    --repo_id your-org/pi06-aloha
```

### Available Configs

**Available Frozen Backbone Configs:**

| Config | Robot | Tasks | Action Dim | Training Time |
|--------|-------|-------|------------|---------------|
| `pi06_multi` | ALOHA (bimanual) | Cube transfer, Peg insertion | 14-DOF | ~7h on A100 |
| `pi06_libero` | Single arm | 50+ diverse tasks | 7-DOF | ~7h on A100 |
| `pi06_aloha_sim` | ALOHA (bimanual) | Cube transfer only | 14-DOF | ~4h on A100 |

**Training Commands:**

```bash
# ALOHA bimanual (cube transfer + peg insertion with language prompts)
python scripts/train.py pi06_multi --exp-name pi06_multi_v1

# LIBERO (50+ diverse manipulation tasks)
python scripts/train.py pi06_libero --exp-name pi06_libero_v1

# Single task (for testing)
python scripts/train.py pi06_aloha_sim --exp-name pi06_aloha_v1
```

### Multi-Task Training Details

**pi06_multi** trains on 4 ALOHA sim datasets with task-specific language prompts:
- "Pick up the cube and transfer it to a new location"
- "Insert the peg into the socket"
- (and scripted variants)

**pi06_libero** trains on 50+ diverse tasks from the LIBERO benchmark:
- Pick and place objects
- Open/close drawers and cabinets
- Stack blocks
- Pour liquids
- Many more...

Each task uses its natural language description as the prompt, enabling the model to respond to language instructions at inference time.

### Gemma 3 Configs (Experimental, Unverified):
| Config | Description | Training Steps |
|--------|-------------|----------------|
| `pi06_gemma3_aloha_sim` | Full Gemma 3 4B | 100k |
| `pi06_gemma3_frozen` | Frozen Gemma 3 | 30k |
| `pi06_gemma3_lora` | Gemma 3 + LoRA | 20k |

> **Warning**: These configs are experimental. No benchmark results have been published.

### Why Pi0.5 Frozen Backbone is Recommended

**Pi0.5 was pretrained on robot manipulation data** (DROID, 20M+ samples). Gemma 3 4B is just a language model with no robot knowledge.

With limited data:
- **Pi0.5 frozen**: Transfer learning from a robot-capable model
- **Gemma 3 4B**: Training 860M action expert from scratch → needs much more data

### Technical Note: Frozen Backbone Memory Optimization

When training with a frozen VLM backbone, we use `freeze_vision_backbone=True` in the model config to significantly reduce GPU memory usage.

**The Problem**: Simply using `freeze_filter` in the training config does NOT reduce memory because:
- `freeze_filter` only affects which parameters receive optimizer updates
- JAX still computes gradients through all parameters in the backward pass
- This requires ~38GB per GPU even with batch_size=8, exceeding A100-40GB capacity

**The Solution**: Setting `freeze_vision_backbone=True` applies `jax.lax.stop_gradient` to the frozen backbone's output during the forward pass. This:
- Prevents gradient computation through the frozen VLM backbone
- Reduces memory from ~38GB to ~33.5GB per GPU
- Enables training on A100-40GB GPUs with batch_size=16

**Memory Requirements** (Pi0.5 architecture with max_token_len=200):
| Configuration | Memory per GPU | Batch Size | A100-40GB |
|--------------|----------------|------------|-----------|
| freeze_filter only | ~38GB | 32 | OOM |
| freeze_vision_backbone=True | ~33.5GB | 32 | OOM |
| freeze_vision_backbone=True | ~33.5GB | 16 | Works |
| Full fine-tuning | ~48GB | any | OOM |

The `pi06_aloha_sim` config uses `freeze_vision_backbone=True` with `batch_size=16` for A100-40GB compatibility.

### Evaluation

We provide two methods for evaluating trained models:

#### Option A: Docker-Based (Recommended)

This is the recommended approach - it handles dependencies cleanly.

```bash
# 1. Build Docker images (first time only)
sudo docker build . -t openpi_server -f scripts/docker/serve_policy.Dockerfile
sudo docker build . -t aloha_sim -f examples/aloha_sim/Dockerfile

# 2. Start policy server
export CHECKPOINT="./checkpoints/pi06_multi/your_exp/30000"
sudo docker run --rm -d --name openpi_server \
  --network host --gpus all \
  -v $(pwd):/app \
  -e SERVER_ARGS="policy:checkpoint --policy.config pi06_multi --policy.dir /app/${CHECKPOINT}" \
  openpi_server

# 3. Wait for server (check logs until you see "listening on 0.0.0.0:8000")
sudo docker logs openpi_server 2>&1 | tail -5

# 4. Run evaluation
sudo docker run --rm --name aloha_eval \
  --network host \
  -e MUJOCO_GL=egl \
  -v $(pwd):/app \
  aloha_sim \
  /bin/bash -c "source /.venv/bin/activate && python /app/examples/aloha_sim/evaluate.py \
    --args.num-episodes 50 \
    --args.task gym_aloha/AlohaTransferCube-v0 \
    --args.prompt 'Pick up the cube and transfer it to a new location'"

# 5. Cleanup
sudo docker stop openpi_server
```

#### Option B: Direct Evaluation

If you have all dependencies installed locally:

```bash
python scripts/evaluate_aloha_sim.py \
    --checkpoint_dir ./checkpoints/pi06_multi/your_exp/30000 \
    --config pi06_multi \
    --task gym_aloha/AlohaTransferCube-v0 \
    --num_episodes 50
```

See [docs/evaluation.md](docs/evaluation.md) for detailed evaluation documentation.

---

## Fine-Tuning on Your Own Tasks

This section explains how to fine-tune on your own robot tasks using the frozen backbone approach. The process is simple:
1. Prepare your dataset
2. Add a config entry
3. Run training

### Step 1: Prepare Your Dataset

Your data should be in [LeRobot format](https://github.com/huggingface/lerobot). The key fields are:

| Field | Description | Example Shape |
|-------|-------------|---------------|
| `observation.images.*` | Camera images | (480, 640, 3) |
| `observation.state` | Robot joint positions | (7,) or (14,) |
| `action` | Target actions | (7,) or (14,) |

Upload to HuggingFace Hub:
```bash
huggingface-cli upload your-username/my-robot-dataset ./my_dataset
```

### Step 2: Add Your Config

**Option A: Use the helper script (easiest)**

```bash
python scripts/create_finetune_config.py \
    --name pi06_my_task \
    --datasets "your-username/my-dataset" \
    --prompts "Pick up the object and place it on the table" \
    --action_dim 7

# For multi-task:
python scripts/create_finetune_config.py \
    --name pi06_multi_task \
    --datasets "user/pick-place" "user/pour-water" \
    --prompts "Pick and place the object" "Pour water into the glass"
```

Then paste the generated config into `src/fla/training/config.py`.

**Option B: Manually edit config.py**

Edit `src/fla/training/config.py` and copy the template:

```python
TrainConfig(
    name="pi06_my_custom_task",  # Unique name for your config
    model=pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        action_dim=7,  # 7 for single arm, 14 for bimanual
        action_horizon=50,
        freeze_vision_backbone=True,
    ),
    data=MultiLeRobotDataConfig(
        repo_ids=(
            "your-username/my-robot-dataset",
            # Add more datasets here for multi-task training
        ),
        repo_id_to_prompt={
            # Each dataset gets a task-specific prompt
            # The model learns to respond to these instructions
            "your-username/my-robot-dataset": "Pick up the red block and place it on the plate",
        },
        use_delta_joint_actions=False,
        adapt_to_pi=False,  # False for non-ALOHA robots
    ),
    weight_loader=weight_loaders.FlexibleCheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"
    ),
    freeze_filter=nnx_utils.PathRegex(".*PaliGemma/llm/(?!.*_1).*"),
    num_train_steps=30_000,
    batch_size=16,
),
```

### Step 3: Compute Normalization Stats

```bash
python scripts/compute_norm_stats.py pi06_my_custom_task
```

### Step 4: Train

```bash
python scripts/train.py pi06_my_custom_task --exp-name my_experiment_v1
```

### Step 5: Run Inference

```python
from fla.training import config
from fla.policies import policy_config

cfg = config.get_config("pi06_my_custom_task")
policy = policy_config.create_trained_policy(
    cfg,
    "./checkpoints/pi06_my_custom_task/my_experiment_v1/30000"
)

# Run inference
actions = policy.infer({
    "observation.images.cam_high": image,
    "observation.state": robot_state,
    "prompt": "Pick up the red block and place it on the plate",
})
```

### Multi-Task Training

To train on multiple tasks, add multiple datasets with different prompts:

```python
data=MultiLeRobotDataConfig(
    repo_ids=(
        "your-username/pick-place-dataset",
        "your-username/pour-water-dataset",
        "your-username/open-drawer-dataset",
    ),
    repo_id_to_prompt={
        "your-username/pick-place-dataset": "Pick up the object and place it in the bin",
        "your-username/pour-water-dataset": "Pour water from the pitcher into the glass",
        "your-username/open-drawer-dataset": "Open the drawer",
    },
),
```

The model will learn to:
- Distinguish between tasks based on language
- Perform the correct task when given the matching prompt
- Generalize to similar prompts (e.g., "Move the object to the bin")

### Tips for Good Results

| Tip | Details |
|-----|---------|
| **Data quality > quantity** | 50 high-quality demos often beat 500 poor ones |
| **Consistent camera angles** | Keep camera positions similar across episodes |
| **Clear task prompts** | Use specific, descriptive prompts |
| **Start with frozen backbone** | Use `freeze_vision_backbone=True` first |
| **Monitor loss** | Loss should drop to ~0.01-0.02 |

### Troubleshooting

| Issue | Solution |
|-------|----------|
| OOM Error | Reduce `batch_size` to 8 |
| Loss not decreasing | Increase `peak_lr` to 1e-4 |
| Overfitting | Reduce `num_train_steps` |
| Poor generalization | Add more diverse data or prompts |

---

## Running Inference for a Pre-Trained Model

Our pre-trained model checkpoints can be run with a few lines of code (here our $\pi_0$-FAST-DROID model):
```python
from fla.training import config as _config
from fla.policies import policy_config
from fla.shared import download

config = _config.get_config("pi05_droid")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")

# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# Run inference on a dummy example.
example = {
    "observation/exterior_image_1_left": ...,
    "observation/wrist_image_left": ...,
    ...
    "prompt": "pick up the fork"
}
action_chunk = policy.infer(example)["actions"]
```
You can also test this out in the [example notebook](examples/inference.ipynb).

We provide detailed step-by-step examples for running inference of our pre-trained checkpoints on [DROID](examples/droid/README.md) and [ALOHA](examples/aloha_real/README.md) robots.

**Remote Inference**: We provide [examples and code](docs/remote_inference.md) for running inference of our models **remotely**: the model can run on a different server and stream actions to the robot via a websocket connection. This makes it easy to use more powerful GPUs off-robot and keep robot and policy environments separate.

**Test inference without a robot**: We provide a [script](examples/simple_client/README.md) for testing inference without a robot. This script will generate a random observation and run inference with the model. See [here](examples/simple_client/README.md) for more details.





## Fine-Tuning Base Models on Your Own Data

We will fine-tune the $\pi_{0.5}$ model on the [LIBERO dataset](https://libero-project.github.io/datasets) as a running example for how to fine-tune a base model on your own data. We will explain three steps:
1. Convert your data to a LeRobot dataset (which we use for training)
2. Defining training configs and running training
3. Spinning up a policy server and running inference

### 1. Convert your data to a LeRobot dataset

We provide a minimal example script for converting LIBERO data to a LeRobot dataset in [`examples/libero/convert_libero_data_to_lerobot.py`](examples/libero/convert_libero_data_to_lerobot.py). You can easily modify it to convert your own data! You can download the raw LIBERO dataset from [here](https://huggingface.co/datasets/openvla/modified_libero_rlds), and run the script with:

```bash
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/libero/data
```

**Note:** If you just want to fine-tune on LIBERO, you can skip this step, because our LIBERO fine-tuning configs point to a pre-converted LIBERO dataset. This step is merely an example that you can adapt to your own data.

### 2. Defining training configs and running training

To fine-tune a base model on your own data, you need to define configs for data processing and training. We provide example configs with detailed comments for LIBERO below, which you can modify for your own dataset:

- [`LiberoInputs` and `LiberoOutputs`](src/fla/policies/libero_policy.py): Defines the data mapping from the LIBERO environment to the model and vice versa. Will be used for both, training and inference.
- [`LeRobotLiberoDataConfig`](src/fla/training/config.py): Defines how to process raw LIBERO data from LeRobot dataset for training.
- [`TrainConfig`](src/fla/training/config.py): Defines fine-tuning hyperparameters, data config, and weight loader.

We provide example fine-tuning configs for [π₀](src/fla/training/config.py), [π₀-FAST](src/fla/training/config.py), and [π₀.₅](src/fla/training/config.py) on LIBERO data.

Before we can run training, we need to compute the normalization statistics for the training data. Run the script below with the name of your training config:

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_libero
```

Now we can kick off training with the following command (the `--overwrite` flag is used to overwrite existing checkpoints if you rerun fine-tuning with the same config):

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero --exp-name=my_experiment --overwrite
```

The command will log training progress to the console and save checkpoints to the `checkpoints` directory. You can also monitor training progress on the Weights & Biases dashboard. For maximally using the GPU memory, set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` before running training -- this enables JAX to use up to 90% of the GPU memory (vs. the default of 75%).

**Note:** We provide functionality for *reloading* normalization statistics for state / action normalization from pre-training. This can be beneficial if you are fine-tuning to a new task on a robot that was part of our pre-training mixture. For more details on how to reload normalization statistics, see the [norm_stats.md](docs/norm_stats.md) file.

### 3. Spinning up a policy server and running inference

Once training is complete, we can run inference by spinning up a policy server and then querying it from a LIBERO evaluation script. Launching a model server is easy (we use the checkpoint for iteration 20,000 for this example, modify as needed):

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero --policy.dir=checkpoints/pi05_libero/my_experiment/20000
```

This will spin up a server that listens on port 8000 and waits for observations to be sent to it. We can then run an evaluation script (or robot runtime) that queries the server.

For running the LIBERO eval in particular, we provide (and recommend using) a Dockerized workflow that handles both the policy server and the evaluation script together. See the [LIBERO README](examples/libero/README.md) for more details.

If you want to embed a policy server call in your own robot runtime, we have a minimal example of how to do so in the [remote inference docs](docs/remote_inference.md).



### More Examples

We provide more examples for how to fine-tune and run inference with our models on the ALOHA platform in the following READMEs:
- [ALOHA Simulator](examples/aloha_sim)
- [ALOHA Real](examples/aloha_real)
- [UR5](examples/ur5)

---

## Continual Learning (Preventing Catastrophic Forgetting)

When fine-tuning on new tasks, models can "forget" previously learned tasks - this is known as **catastrophic forgetting**. We provide built-in support for **Elastic Weight Consolidation (EWC)**, which prevents this by identifying important weights and protecting them during new task training.

### How EWC Works

1. **After training on Task A**: Compute the Fisher information matrix, which measures how important each weight is for Task A
2. **When training on Task B**: Add a regularization term that penalizes changes to important weights
3. **Result**: The model learns Task B while preserving Task A performance

### Quick Start

**Step 1: Train on your first task normally**
```bash
# Train on initial task(s)
uv run scripts/train.py pi06_multi --exp-name=v1

# After training, compute Fisher information for EWC
uv run scripts/compute_ewc_state.py pi06_multi --exp-name=v1
```

**Step 2: Train on new tasks with EWC protection**
```bash
# Add new tasks and train with EWC enabled
uv run scripts/train.py pi06_new_tasks --exp-name=v1 \
    --continual_learning.ewc.enabled=True \
    --continual_learning.ewc.lambda_ewc=1000
```

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `continual_learning.ewc.enabled` | False | Enable EWC regularization |
| `continual_learning.ewc.lambda_ewc` | 1000 | Regularization strength (higher = less forgetting, slower learning) |
| `continual_learning.ewc.fisher_samples` | 200 | Number of samples for Fisher estimation |
| `continual_learning.ewc.online_ewc` | True | Accumulate Fisher across tasks (recommended) |
| `continual_learning.ewc.gamma` | 0.95 | Decay factor for online EWC |

### Recommended Workflow for Adding New Tasks

1. **Start with multi-task training**: Train on multiple initial tasks together
   ```bash
   uv run scripts/train.py pi06_multi --exp-name=v1
   ```

2. **Compute EWC state**: After training completes, compute Fisher information
   ```bash
   uv run scripts/compute_ewc_state.py pi06_multi --exp-name=v1
   ```

3. **Add new tasks**: Create a new config with your additional tasks

4. **Continue training with EWC**: Train on new tasks while protecting old ones
   ```bash
   uv run scripts/train.py pi06_expanded --exp-name=v1 \
       --continual_learning.ewc.enabled=True \
       --continual_learning.ewc.lambda_ewc=1000
   ```

### Tips

- **Start with `lambda_ewc=1000`** and adjust based on results:
  - If forgetting old tasks: increase lambda (try 5000-10000)
  - If not learning new tasks well: decrease lambda (try 100-500)

- **Use multi-task training first**: It's better to train on all available tasks together initially, then use EWC when adding truly new tasks later

- **Monitor both old and new task metrics**: Track performance on all tasks to ensure good balance

- **Online EWC (default)** accumulates importance across tasks, which is better for sequential task addition

---

## PyTorch Support

FLA now provides PyTorch implementations of π₀ and π₀.₅ models alongside the original JAX versions! The PyTorch implementation has been validated on the LIBERO benchmark (both inference and finetuning). A few features are currently not supported (this may change in the future):

- The π₀-FAST model
- Mixed precision training
- FSDP (fully-sharded data parallelism) training
- LoRA (low-rank adaptation) training
- EMA (exponential moving average) weights during training

### Setup
1. Make sure that you have the latest version of all dependencies installed: `uv sync`

2. Double check that you have transformers 4.53.2 installed: `uv pip show transformers`

3. Apply the transformers library patches:
   ```bash
   cp -r ./src/fla/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
   ```

This overwrites several files in the transformers library with necessary model changes: 1) supporting AdaRMS, 2) correctly controlling the precision of activations, and 3) allowing the KV cache to be used without being updated.

**WARNING**: With the default uv link mode (hardlink), this will permanently affect the transformers library in your uv cache, meaning the changes will survive reinstallations of transformers and could even propagate to other projects that use transformers. To fully undo this operation, you must run `uv cache clean transformers`.

### Converting JAX Models to PyTorch

To convert a JAX model checkpoint to PyTorch format:

```bash
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir /path/to/jax/checkpoint \
    --config_name <config name> \
    --output_path /path/to/converted/pytorch/checkpoint
```

### Running Inference with PyTorch

The PyTorch implementation uses the same API as the JAX version - you only need to change the checkpoint path to point to the converted PyTorch model:

```python
from fla.training import config as _config
from fla.policies import policy_config
from fla.shared import download

config = _config.get_config("pi05_droid")
checkpoint_dir = "/path/to/converted/pytorch/checkpoint"

# Create a trained policy (automatically detects PyTorch format)
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# Run inference (same API as JAX)
action_chunk = policy.infer(example)["actions"]
```

### Policy Server with PyTorch

The policy server works identically with PyTorch models - just point to the converted checkpoint directory:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_droid \
    --policy.dir=/path/to/converted/pytorch/checkpoint
```

### Finetuning with PyTorch

To finetune a model in PyTorch:

1. Convert the JAX base model to PyTorch format:
   ```bash
   uv run examples/convert_jax_model_to_pytorch.py \
       --config_name <config name> \
       --checkpoint_dir /path/to/jax/base/model \
       --output_path /path/to/pytorch/base/model
   ```

2. Specify the converted PyTorch model path in your config using `pytorch_weight_path`

3. Launch training using one of these modes:

```bash
# Single GPU training:
uv run scripts/train_pytorch.py <config_name> --exp_name <run_name> --save_interval <interval>

# Example:
uv run scripts/train_pytorch.py debug --exp_name pytorch_test
uv run scripts/train_pytorch.py debug --exp_name pytorch_test --resume  # Resume from latest checkpoint

# Multi-GPU training (single node):
uv run torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>

# Example:
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test --resume

# Multi-Node Training:
uv run torchrun \
    --nnodes=<num_nodes> \
    --nproc_per_node=<gpus_per_node> \
    --node_rank=<rank_of_node> \
    --master_addr=<master_ip> \
    --master_port=<port> \
    scripts/train_pytorch.py <config_name> --exp_name=<run_name> --save_interval <interval>
```

### Precision Settings

JAX and PyTorch implementations handle precision as follows:

**JAX:**
1. Inference: most weights and computations in bfloat16, with a few computations in float32 for stability
2. Training: defaults to mixed precision: weights and gradients in float32, (most) activations and computations in bfloat16. You can change to full float32 training by setting `dtype` to float32 in the config.

**PyTorch:**
1. Inference: matches JAX -- most weights and computations in bfloat16, with a few weights converted to float32 for stability
2. Training: supports either full bfloat16 (default) or full float32. You can change it by setting `pytorch_training_precision` in the config. bfloat16 uses less memory but exhibits higher losses compared to float32. Mixed precision is not yet supported.

With torch.compile, inference speed is comparable between JAX and PyTorch.

## RECAP Training (Experimental)

RECAP (RL with Experience and Corrections via Advantage-conditioned Policies) is the method from the **actual Pi0.6 paper**. It enables iterative policy improvement using advantage conditioning.

> **Status**: This is an experimental implementation. **No benchmark results have been verified** against the paper's claims (60% → 85% on ALOHA Transfer Cube, 45% → 75% on Franka Cabinet).

### How RECAP Works

RECAP improves upon standard behavior cloning by:
1. **Training a value function** to predict time-to-completion for each state
2. **Computing advantages** to identify which trajectories are better than average
3. **Conditioning the policy** on an improvement indicator (I=1 for good, I=0 for bad trajectories)
4. **Iteratively collecting new data** with the improved policy and retraining

At inference time, we set I=1 to generate actions from "good" trajectory behavior.

### Quick Start

```bash
# Option 1: Train with LeRobot ALOHA simulation data
python scripts/train_recap_full.py --config recap_aloha_sim

# Option 2: Collect data from Isaac Lab and train
python scripts/isaaclab_data_collection.py --task Isaac-Franka-Cabinet-Direct-v0 --num_episodes 100
python scripts/train_recap_full.py --config recap_aloha_sim

# Run benchmarks (targets from pi0.6 paper - not yet verified in this repo)
python scripts/benchmark_recap.py --compare-paper --run_training
```

### RECAP Training Phases

1. **Value Function Training**: Learns to predict time-to-completion
   - Uses distributional value function with 201 bins
   - Trained with cross-entropy loss on (observation, time_remaining) pairs

2. **Advantage Computation**: A(o_t) = V(o_t) - actual_time_remaining
   - Positive advantage = trajectory is better than average
   - Sets improvement indicator I_t = 1 if A > 0

3. **Policy Warmup**: Standard BC training without advantage conditioning

4. **RECAP Training**: Policy training with advantage conditioning
   - Learns to imitate good trajectories (I=1)
   - Learns to avoid behaviors from bad trajectories (I=0)

### Using a Trained RECAP Policy

```python
from fla.recap.pi0_recap import Pi0RECAPConfig
import jax

# Create and load policy
config = Pi0RECAPConfig(
    paligemma_variant="gemma_2b",
    action_expert_variant="gemma_2b",
    action_dim=9,
    action_horizon=50,
)
rng = jax.random.key(42)
policy = config.create(rng)

# Load checkpoint
# ... (see checkpoint loading example in train_recap_full.py)

# Sample actions with I=1 (want good trajectories)
improvement_indicator = jnp.ones(batch_size, dtype=jnp.bool_)
actions = policy.sample_actions(rng, observation, improvement_indicator=improvement_indicator)
```

### Configuration

See `configs/benchmark_recap.yaml` for detailed configuration options. Key parameters:
- `warmup_steps`: Steps of standard training before RECAP (default: 50000)
- `recap_iterations`: Number of collect-train cycles (default: 3)
- `steps_per_iteration`: Training steps per cycle (default: 10000)

### Documentation

- [RECAP Training Guide](docs/recap_training.md): Detailed training instructions
- [Isaac Lab Setup](docs/isaac_lab_setup.md): Setting up Isaac Lab for data collection

## Troubleshooting

We will collect common issues and their solutions here. If you encounter an issue, please check here first. If you can't find a solution, please file an issue on the repo (see [here](CONTRIBUTING.md) for guidelines).

| Issue                                     | Resolution                                                                                                                                                                                   |
| ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `uv sync` fails with dependency conflicts | Try removing the virtual environment directory (`rm -rf .venv`) and running `uv sync` again. If issues persist, check that you have the latest version of `uv` installed (`uv self update`). |
| Training runs out of GPU memory           | Make sure you set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` (or higher) before running training to allow JAX to use more GPU memory. You can also use `--fsdp-devices <n>` where `<n>` is your number of GPUs, to enable [fully-sharded data parallelism](https://engineering.fb.com/2021/07/15/open-source/fsdp/), which reduces memory usage in exchange for slower training (the amount of slowdown depends on your particular setup). If you are still running out of memory, you may want to consider disabling EMA.        |
| Policy server connection errors           | Check that the server is running and listening on the expected port. Verify network connectivity and firewall settings between client and server.                                            |
| Missing norm stats error when training    | Run `scripts/compute_norm_stats.py` with your config name before starting training.                                                                                                          |
| Dataset download fails                    | Check your internet connection. For HuggingFace datasets, ensure you're logged in (`huggingface-cli login`).                                                                                 |
| CUDA/GPU errors                           | Verify NVIDIA drivers are installed correctly. For Docker, ensure nvidia-container-toolkit is installed. Check GPU compatibility. You do NOT need CUDA libraries installed at a system level --- they will be installed via uv. You may even want to try *uninstalling* system CUDA libraries if you run into CUDA issues, since system libraries can sometimes cause conflicts. |
| Import errors when running examples       | Make sure you've installed all dependencies with `uv sync`. Some examples may have additional requirements listed in their READMEs.                    |
| Action dimensions mismatch                | Verify your data processing transforms match the expected input/output dimensions of your robot. Check the action space definitions in your policy classes.                                  |
| Diverging training loss                            | Check the `q01`, `q99`, and `std` values in `norm_stats.json` for your dataset. Certain dimensions that are rarely used can end up with very small `q01`, `q99`, or `std` values, leading to huge states and actions after normalization. You can manually adjust the norm stats as a workaround. |
