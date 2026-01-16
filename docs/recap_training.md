# RECAP Training Guide

This guide covers training the RECAP algorithm (pi0.6) for iterative policy improvement.

## Overview

RECAP (RL with Experience and Corrections via Advantage-conditioned Policies) is a training algorithm that enables:
- Learning from both successful and unsuccessful trajectories
- Iterative policy improvement through advantage conditioning
- More efficient use of collected robot data

## Algorithm

### Key Concepts

1. **Distributional Value Function V(o_t)**
   - Predicts the expected time-to-completion from state o_t
   - Uses 201 bins for discrete distribution over 0-200+ steps
   - Trained with cross-entropy loss

2. **Advantage A(o_t)**
   - A(o_t) = V(o_t) - actual_time_remaining
   - Positive advantage: trajectory is doing better than average
   - Negative advantage: trajectory is doing worse than average

3. **Improvement Indicator I_t**
   - I_t = 1 if A(o_t) > 0 (good trajectory)
   - I_t = 0 if A(o_t) <= 0 (bad trajectory)
   - Used to condition the policy during training

4. **Advantage-Conditioned Policy**
   - Policy learns: π(a | o, I)
   - During training: learns from both I=1 and I=0 data
   - During inference: always use I=1 to generate good actions

## Training Pipeline

### Phase 1: Value Function Training

```bash
# The value function is trained as part of the full pipeline
python scripts/train_recap_full.py --config recap_aloha_sim
```

The value function:
- Takes image observations and robot state
- Predicts distribution over time-to-completion
- Is trained for `value_train_steps` iterations

### Phase 2: Advantage Computation

After value function training, advantages are computed for all samples:

```python
# For each timestep t in episode:
predicted_ttc = value_fn.predict_value(observation)
actual_ttc = episode_length - t
advantage = predicted_ttc - actual_ttc
improvement_indicator = advantage > 0
```

Expected distribution:
- ~50% of samples should have I=1 (if value function is well-calibrated)
- Mean advantage should be near 0

### Phase 3: Policy Training

**Warmup Phase (Standard BC)**
- Train without advantage conditioning
- Establishes baseline policy behavior
- Uses `policy_warmup_steps` iterations

**RECAP Phase (Advantage-Conditioned)**
- Train with improvement indicator as conditioning
- Policy learns to differentiate good/bad behaviors
- Uses `policy_recap_steps` iterations

## Configuration Options

### Command Line Arguments

```bash
python scripts/train_recap_full.py \
    --config recap_aloha_sim \           # Config name
    --model_variant gemma_2b \           # Model size (dummy for testing, gemma_2b for full)
    --value_train_steps 500 \            # Value function training steps
    --policy_warmup_steps 200 \          # Policy warmup steps
    --policy_recap_steps 300 \           # RECAP training steps
    --batch_size 32 \                    # Batch size per GPU
    --experiment_name my_experiment \    # Experiment name for wandb
    --no_wandb                           # Disable wandb logging
```

### Resuming Training

```bash
# Resume from checkpoint (skips warmup)
python scripts/train_recap_full.py \
    --config recap_aloha_sim \
    --resume_from checkpoints/recap_full/experiment/checkpoint_500 \
    --skip_value_training
```

### Full Production Config

For reproducing pi0.6 paper results:

```yaml
# From configs/benchmark_recap.yaml
production:
  warmup_steps: 50000
  steps_per_iteration: 10000
  recap_iterations: 3
  episodes_per_iteration: 100
  batch_size: 32
  learning_rate: 1e-4
```

## Data Sources

### Option 1: LeRobot Datasets

```bash
# Use public ALOHA simulation data
python scripts/train_recap_full.py --config recap_aloha_sim
```

Available datasets:
- `lerobot/aloha_sim_transfer_cube_human`
- Other LeRobot robotics datasets

### Option 2: Isaac Lab Collection

```bash
# Collect from Isaac Lab simulation
python scripts/isaaclab_data_collection.py \
    --task Isaac-Franka-Cabinet-Direct-v0 \
    --num_episodes 100 \
    --output_dir data/isaaclab

# Train on collected data
# (Note: requires custom data loader configuration)
```

### Option 3: Custom Data

To use your own data:
1. Format data with observations, actions, and episode boundaries
2. Compute time-to-completion for each timestep
3. Create a dataset class similar to `LeRobotRECAPDataset`

## Monitoring Training

### Wandb Metrics

Key metrics to monitor:
- `value/loss`: Value function cross-entropy loss (should decrease)
- `policy/warmup_loss`: Policy loss during warmup (should decrease)
- `policy/recap_loss`: Policy loss during RECAP (should decrease)
- `policy/pct_good`: Fraction of samples with I=1 (should be ~50%)

### Expected Behavior

**Value Function:**
- Loss should start around 5.0 (random predictions over 201 bins)
- Should decrease to < 1.0 for well-trained value function

**Policy:**
- Warmup loss should decrease similarly to standard BC
- RECAP loss may be lower than warmup due to advantage conditioning
- Final loss around 0.05-0.10 is typical

### Common Issues

**All samples marked as bad (I=0)**
- Value function not converged
- Increase value training steps
- Check advantage distribution statistics

**Loss not decreasing**
- Learning rate too high/low
- Batch size issues
- Check gradient norms in wandb

**Out of memory**
- Reduce batch size
- Use model_variant="dummy" for testing
- Enable gradient checkpointing

## Iterative RECAP Loop

For maximum performance, run multiple RECAP iterations:

```bash
# Iteration 1: Train on initial data
python scripts/train_recap_full.py --config recap_aloha_sim --experiment_name iter1

# Collect new data with trained policy
python scripts/isaaclab_data_collection.py \
    --policy checkpoint \
    --checkpoint_path checkpoints/recap_full/iter1/checkpoint_final \
    --num_episodes 100

# Iteration 2: Train on new data
python scripts/train_recap_full.py --config recap_aloha_sim \
    --experiment_name iter2 \
    --resume_from checkpoints/recap_full/iter1/checkpoint_final

# Repeat for 3-5 iterations
```

Each iteration should improve policy quality as:
1. Better policy generates better data
2. Better data improves value function accuracy
3. More accurate advantages improve policy training

## Benchmarking

Compare your results with pi0.6 paper:

```bash
python scripts/benchmark_recap.py --compare-paper --run_training
```

Expected results on ALOHA Sim Transfer Cube:
| Metric | Baseline | RECAP |
|--------|----------|-------|
| Success Rate | ~60% | ~85% |
| Policy Loss | 0.14 | 0.05 |

## Resources

- [Pi0.6 Paper](https://www.physicalintelligence.company/): Original RECAP paper
- [Isaac Lab Setup](isaac_lab_setup.md): Data collection environment
- [Benchmark Config](../configs/benchmark_recap.yaml): Paper-matching parameters
