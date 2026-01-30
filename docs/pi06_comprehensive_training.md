# Pi0.6 Comprehensive Training & Evaluation Guide

This guide covers how to train Pi0.6 to match or exceed the official benchmarks.

## Quick Start (Recommended)

**For most users with limited training data**, use the Pi0.5-based frozen backbone approach:

```bash
# 1. Download ALOHA sim datasets
python scripts/download_datasets.py --skip_oxe

# 2. Train with frozen Pi0.5 backbone (RECOMMENDED)
python scripts/train.py pi06_multi --exp_name pi06_v1 --fsdp_devices 8

# 3. Upload to HuggingFace after training
python scripts/upload_to_huggingface.py \
    --checkpoint_path ./checkpoints/pi06_multi/pi06_v1/30000/params \
    --repo_id your-org/pi06-aloha
```

**Why this is recommended:**
- Pi0.5 backbone has **robot knowledge** (pretrained on 20M+ samples)
- Only trains **300M action expert** (frozen 2B backbone)
- Works well with **~100k frames** of data

---

## Full Pipeline (Large Datasets Only)

The following is only recommended if you have access to **millions of robot demonstrations** (DROID, OXE, etc.).

## Target Benchmarks (from Pi0.6 Paper)

| Benchmark | Baseline | RECAP | Target |
|-----------|----------|-------|--------|
| ALOHA Sim Transfer Cube | 60% | 85% | **85%+** |
| Franka Cabinet | 45% | 75% | **75%+** |
| LIBERO Spatial | - | - | **98%+** |
| LIBERO Object | - | - | **98%+** |
| LIBERO Goal | - | - | **98%+** |
| LIBERO-10 | - | - | **92%+** |

## Training Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 1: BASE PRETRAINING                     │
│  Gemma 3 4B + Multi-Dataset (DROID, Bridge, Fractal, RT-1, OXE) │
│                    ~200k steps, ~4 days                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  PHASE 2: TASK FINE-TUNING                       │
│  Fine-tune on target benchmark datasets (LIBERO, ALOHA)         │
│                    ~50k steps per task                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 3: RECAP TRAINING                       │
│  Value function + Advantage-conditioned policy training         │
│                    3 iterations, ~30k steps each                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      PHASE 4: EVALUATION                         │
│  Run benchmarks: ALOHA Sim, LIBERO suite, Franka Cabinet        │
└─────────────────────────────────────────────────────────────────┘
```

## Phase 1: Base Pretraining

### Datasets Required

Download all datasets to `/data/rlds_datasets/`:

```bash
# 1. DROID (~2TB) - Primary manipulation dataset
# Download from: https://droid-dataset.github.io/

# 2. Open X-Embodiment datasets (~500GB total)
python -c "
import tensorflow_datasets as tfds

datasets = [
    'bridge_dataset',           # WidowX manipulation
    'fractal20220817_data',     # Google robot
    'rt1_robot_action',         # RT-1 language-conditioned
    'kuka',                     # Kuka manipulation
    'taco_play',                # TACO benchmark
    'jaco_play',                # Jaco robot
    'berkeley_cable_routing',   # Cable routing
    'berkeley_autolab_ur5',     # UR5 manipulation
]

for ds in datasets:
    print(f'Downloading {ds}...')
    tfds.load(ds, data_dir='/data/rlds_datasets', download=True)
"
```

### Training Command

```bash
# Phase 1: Base pretraining on multi-dataset
python scripts/train.py pi06_base \
    --exp_name pi06_base_v1 \
    --batch_size 256 \
    --fsdp_devices 8 \
    --num_train_steps 200000
```

**Expected**: ~4 days on 8x H100s

## Phase 2: Task Fine-Tuning

### 2a. LIBERO Fine-Tuning

```bash
# Fine-tune for LIBERO benchmark
python scripts/train.py pi06_libero \
    --exp_name pi06_libero_v1 \
    --weight_loader.params_path ./checkpoints/pi06_base/pi06_base_v1/200000/params \
    --num_train_steps 30000
```

### 2b. ALOHA Fine-Tuning

```bash
# Fine-tune for ALOHA sim
python scripts/train.py pi06_aloha_sim \
    --exp_name pi06_aloha_v1 \
    --weight_loader.params_path ./checkpoints/pi06_base/pi06_base_v1/200000/params \
    --num_train_steps 50000
```

## Phase 3: RECAP Training

RECAP improves policy beyond behavior cloning by learning from advantages.

```bash
# Run full RECAP training
python scripts/train_recap_full.py \
    --config recap_aloha_sim \
    --model_variant gemma3_4b \
    --resume_from ./checkpoints/pi06_aloha_sim/pi06_aloha_v1/50000 \
    --value_train_steps 5000 \
    --policy_warmup_steps 5000 \
    --policy_recap_steps 20000 \
    --recap_iterations 3 \
    --experiment_name pi06_recap_v1
```

**Expected improvement**: 60% → 85% success rate on ALOHA Transfer Cube

## Phase 4: Evaluation

### 4a. ALOHA Sim Evaluation

```bash
# Terminal 1: Start policy server
uv run scripts/serve_policy.py \
    --env ALOHA_SIM \
    policy:checkpoint \
    --policy.config pi06_aloha_sim \
    --policy.dir ./checkpoints/recap_full/pi06_recap_v1/final

# Terminal 2: Run evaluation
cd examples/aloha_sim
python main.py --task gym_aloha/AlohaTransferCube-v0 --num_episodes 100
```

### 4b. LIBERO Evaluation

```bash
# Terminal 1: Start policy server
SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi06_libero --policy.dir ./checkpoints/pi06_libero/pi06_libero_v1/30000" \
docker compose -f examples/libero/compose.yml up --build

# Results will show per-task success rates
```

### 4c. Benchmark Comparison

```bash
# Run comprehensive benchmark comparison
python scripts/benchmark_recap.py \
    --checkpoint_path ./checkpoints/recap_full/pi06_recap_v1/final \
    --compare-paper \
    --output_dir ./benchmark_results/pi06_v1
```

## Comprehensive Config

Add this config for maximum performance:

```python
# In config.py - add this config
TrainConfig(
    name="pi06_comprehensive",
    model=pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="gemma3_4b",
        action_expert_variant="gemma_860m",
        action_dim=7,
        action_horizon=50,
    ),
    data=RLDSDroidDataConfig(
        rlds_data_dir="/data/rlds_datasets",
        action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        datasets=(
            # Primary: DROID (50%)
            RLDSDataset(name="droid", version="1.0.1", weight=0.50,
                       filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json"),
            # Bridge V2 (15%)
            RLDSDataset(name="bridge_dataset", version="1.0.0", weight=0.15),
            # Fractal (10%)
            RLDSDataset(name="fractal20220817_data", version="0.1.0", weight=0.10),
            # RT-1 (10%)
            RLDSDataset(name="rt1_robot_action", version="0.1.0", weight=0.10),
            # Kuka (5%)
            RLDSDataset(name="kuka", version="0.1.0", weight=0.05),
            # TACO (5%)
            RLDSDataset(name="taco_play", version="0.1.0", weight=0.05),
            # Berkeley UR5 (5%)
            RLDSDataset(name="berkeley_autolab_ur5", version="0.1.0", weight=0.05),
        ),
    ),
    weight_loader=weight_loaders.Gemma3WeightLoader(),
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=5_000,
        peak_lr=1e-4,
        decay_steps=200_000,
        decay_lr=1e-5,
    ),
    num_train_steps=200_000,
    batch_size=256,
    save_interval=10_000,
    keep_period=50_000,
    num_workers=0,
)
```

## Expected Results

After full training pipeline:

| Benchmark | Baseline BC | After RECAP | Pi0.6 Paper |
|-----------|-------------|-------------|-------------|
| ALOHA Transfer Cube | ~60% | **85%+** | 85% |
| Franka Cabinet | ~45% | **75%+** | 75% |
| LIBERO Average | ~90% | **96%+** | 96.85% |

## Compute Requirements

| Phase | GPUs | Time | Memory/GPU |
|-------|------|------|------------|
| Base Pretraining | 8x H100 | ~4 days | 80GB |
| Task Fine-tuning | 8x H100 | ~1 day | 80GB |
| RECAP Training | 8x H100 | ~1 day | 80GB |
| **Total** | **8x H100** | **~6 days** | **80GB** |

## Monitoring

Track these metrics in W&B:

1. **Value Function Loss**: Should decrease to < 1.0
2. **Policy Loss**: Should decrease to ~0.05-0.10
3. **Advantage Gap**: good_loss - bad_loss should be > 0.05
4. **% Good Samples**: Should be ~50% if value function is calibrated

## Troubleshooting

### Out of Memory
- Reduce batch_size to 128 or 64
- Enable gradient checkpointing
- Use `gemma3_4b_lora` for LoRA fine-tuning

### Slow Convergence
- Check learning rate (try 5e-5 for fine-tuning)
- Increase warmup_steps
- Verify dataset loading is correct

### Poor RECAP Performance
- Ensure value function converges first (loss < 1.0)
- Check advantage distribution (~50% good samples)
- Increase value_train_steps
