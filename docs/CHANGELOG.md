# Changelog

All notable changes to the OpenPI project will be documented in this file.

## [Unreleased] - 2026-01-30

### Recommended Training Approach

For users with **limited training data** (< 1M samples), we recommend:

```bash
# Download ALOHA sim datasets
python scripts/download_datasets.py --skip_oxe

# Train with frozen Pi0.5 backbone (RECOMMENDED)
python scripts/train.py pi06_multi --exp_name pi06_v1 --fsdp_devices 8
```

**Why Pi0.5-based is recommended:**
- Pi0.5 was pretrained on **20M+ robot samples** (DROID)
- Frozen backbone means only **300M action expert** is trained
- Works well with **~100k frames** of training data
- Gemma 3 4B has **no robot knowledge** - needs much more data

### Current Model Checkpoint

Training checkpoint will be available at:
- **HuggingFace**: `openpi/pi06-aloha-multi` (after training completes)
- **Local**: `./checkpoints/pi06_multi/pi06_v1/30000/params`

---

### True Pi0.6 Architecture (Gemma 3 4B)

Added support for the **actual Pi0.6 architecture** from the paper:

| Component | Pi0.5-based | True Pi0.6 |
|-----------|-------------|------------|
| VLM Backbone | Gemma 2B | **Gemma 3 4B** |
| Action Expert | 300M | **860M** |
| Total Params | ~2.7B | **~5B** |

#### New Gemma 3 Configs

```python
# Full training (best results, requires 8x H100)
pi06_gemma3_aloha_sim   # ALOHA simulation
pi06_gemma3_libero      # LIBERO benchmark

# Memory-efficient variants
pi06_gemma3_frozen      # Frozen Gemma 3 backbone
pi06_gemma3_lora        # Gemma 3 with LoRA adapters
```

#### Gemma 3 Weight Loading

New `Gemma3WeightLoader` automatically:
1. Downloads Gemma 3 4B weights from HuggingFace
2. Loads PaliGemma's SigLIP vision encoder
3. Maps weights to OpenPI format
4. Initializes 860M action expert from scratch

```python
weight_loader=weight_loaders.Gemma3WeightLoader()
```

---

### Pi0.6 Training Strategy (Efficient)

Pi0.6 can also use **Pi0.5 as base** with **frozen VLM backbone** for efficient training:

```
Pi0.5 Base (pretrained on DROID ~20M samples)
    ↓ Load weights, freeze VLM backbone
Only train: Action Expert + Projection Layers
    ↓ Fine-tune on task data
RECAP Training (advantage conditioning)
    ↓
Pi0.6
```

#### Key Features

- **Frozen Backbone**: VLM (gemma_2b) frozen, only action expert trains
- **Pi0.5 Weights**: Leverages existing robot knowledge from Pi0.5
- **Fast Training**: ~10x faster than training from scratch
- **LoRA Option**: Even more efficient with LoRA adapters

#### Training Efficiency

| Approach | Training Time | Memory |
|----------|---------------|--------|
| Pi0.5 frozen backbone | ~1 day (8x H100) | 40GB/GPU |
| Pi0.5 + LoRA | ~0.5 days (8x H100) | 24GB/GPU |
| From scratch (old) | ~6 days (8x H100) | 80GB/GPU |

### Configs Updated

All Pi0.6 configs now use Pi0.5 base with frozen backbone:

| Config | Description | Training Steps |
|--------|-------------|----------------|
| `pi06_aloha_sim` | ALOHA sim fine-tuning | 30k |
| `pi06_base` | DROID fine-tuning | 50k |
| `pi06_base_lora` | DROID with LoRA | 30k |
| `pi06_aloha_real` | Real ALOHA fine-tuning | 20k |
| `pi06_multi` | Multi ALOHA sim | 30k |
| `pi06_comprehensive` | Multi-dataset (7 datasets) | 50k |
| `pi06_libero` | LIBERO benchmark | 20k |

### Freeze Filter

Uses regex to freeze VLM backbone while training action expert:
```python
freeze_filter=nnx_utils.PathRegex(".*PaliGemma/llm/(?!.*_1).*")
```
- Freezes: `PaliGemma/llm/*` (main VLM)
- Trains: `PaliGemma/llm/*_1/*` (action expert), projection layers

### Architecture (Same as Pi0.5)

| Component | Variant | Parameters |
|-----------|---------|------------|
| VL Backbone | gemma_2b | ~2B |
| Action Expert | gemma_300m | ~300M |
| Vision Encoder | SigLIP | ~400M |
| **Total** | | **~2.7B** |

### Comprehensive Training Guide

New documentation at `docs/pi06_comprehensive_training.md` covering:
- Full training pipeline (pretraining → fine-tuning → RECAP)
- Dataset download instructions
- Evaluation on all benchmarks
- Expected results vs Pi0.6 paper

### Target Benchmarks

| Benchmark | Baseline | Target (RECAP) |
|-----------|----------|----------------|
| ALOHA Sim Transfer Cube | 60% | **85%+** |
| Franka Cabinet | 45% | **75%+** |
| LIBERO Average | ~90% | **96%+** |

### RECAP Training

- RECAP training methodology documented in `docs/recap_training.md`
- Training scripts: `train_recap.py`, `train_recap_full.py`
- Value function for advantage estimation
- Advantage-conditioned policy training

---

## [0.5] - Previous Release

### Pi0.5 Features
- PaliGemma-based architecture (Gemma 2B + SigLIP)
- 300M action expert
- DROID pretraining
- Flow matching for action generation
- Support for ALOHA, LIBERO, and custom robots
