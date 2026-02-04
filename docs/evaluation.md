# Evaluation Guide

This document explains how to evaluate trained models on benchmark tasks.

## Quick Start

### Local CLI (no Docker)

If you have `gym-aloha` installed locally, the easiest way to run benchmarks is:

```bash
fla-benchmark \
  --suite aloha_sim \
  --checkpoint-dir ./checkpoints/pi06_multi/pi06_multi_v1/30000 \
  --config pi06_multi \
  --num-episodes 50 \
  --output evaluation_results.json
```

To run a single task:

```bash
fla-benchmark \
  --suite aloha_sim \
  --task gym_aloha/AlohaTransferCube-v0 \
  --checkpoint-dir ./checkpoints/pi06_multi/pi06_multi_v1/30000 \
  --config pi06_multi \
  --num-episodes 50
```

### LIBERO (local)

Install LIBERO dependencies:

```bash
uv sync --group libero
```

Run evaluation:

```bash
fla-benchmark \
  --suite libero \
  --libero-suite libero_spatial \
  --checkpoint-dir ./checkpoints/pi05_libero/pi05_libero_v1/30000 \
  --config pi05_libero \
  --libero-num-trials 50
```

### Dataset Evaluation (offline)

Evaluate action prediction error on a LeRobot dataset:

```bash
fla-benchmark \
  --suite dataset \
  --checkpoint-dir ./checkpoints/your_config/your_exp/30000 \
  --config your_config \
  --repo-ids your-org/your_dataset \
  --max-samples 1024
```

Recipe-based evaluation (no config file needed):

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

### DROID Manual Evaluation (real robot)

Launch the interactive DROID evaluation script (requires DROID robot setup):

```bash
fla-benchmark \
  --suite droid_manual \
  --droid-left-camera-id 24259877 \
  --droid-right-camera-id 24514023 \
  --droid-wrist-camera-id 13062452 \
  --droid-external-camera left \
  --droid-remote-host 192.168.1.100 \
  --droid-remote-port 8000
```

### Full Suite

Run ALOHA + LIBERO + dataset eval in one command:

```bash
fla-benchmark \
  --suite full \
  --checkpoint-dir ./checkpoints/pi06_multi/pi06_multi_v1/30000 \
  --config pi06_multi \
  --repo-ids your-org/your_dataset \
  --output evaluation_results.json
```

### 1. Start the Policy Server

```bash
# Build Docker images (first time only)
sudo docker build . -t openpi_server -f scripts/docker/serve_policy.Dockerfile
sudo docker build . -t aloha_sim -f examples/aloha_sim/Dockerfile

# Start server with your checkpoint
export CHECKPOINT="./checkpoints/pi06_multi/pi06_multi_v3/30000"
export SERVER_ARGS="policy:checkpoint --policy.config pi06_multi --policy.dir /app/${CHECKPOINT}"

sudo docker run --rm -d --name openpi_server \
  --network host \
  --gpus all \
  -v $(pwd):/app \
  -e SERVER_ARGS="$SERVER_ARGS" \
  openpi_server
```

### 2. Run Evaluation

```bash
# Wait for server to be ready (check logs)
sudo docker logs openpi_server 2>&1 | tail -5
# Should show: "server listening on 0.0.0.0:8000"

# Run evaluation
sudo docker run --rm --name aloha_eval \
  --network host \
  -e MUJOCO_GL=egl \
  -v $(pwd):/app \
  aloha_sim \
  /bin/bash -c "source /.venv/bin/activate && python /app/examples/aloha_sim/evaluate.py \
    --args.num-episodes 50 \
    --args.task gym_aloha/AlohaTransferCube-v0 \
    --args.prompt 'Pick up the cube and transfer it to a new location'"
```

### 3. Available Tasks

| Task | Prompt | Paper Baseline |
|------|--------|----------------|
| `gym_aloha/AlohaTransferCube-v0` | "Pick up the cube and transfer it to a new location" | 60% |
| `gym_aloha/AlohaInsertion-v0` | "Insert the peg into the socket" | 50% |

## Benchmark Results

### Our Trained Model (pi06_multi @ 30k steps)

| Task | Episodes | Success Rate | Paper Baseline | Status |
|------|----------|--------------|----------------|--------|
| ALOHA Transfer Cube | 50 | **38%** | 60% | ⚠️ Below baseline |
| ALOHA Insertion | 50 | **66%** | 50% | ✅ Above baseline |

> **Key Finding**: The model shows task-dependent performance. It exceeds baseline on insertion but underperforms on cube transfer. This suggests:
> 1. The multi-task training may need rebalancing
> 2. Transfer cube may require RECAP for improvement
> 3. Different tasks may need different prompt engineering

### Physical Intelligence's Official Checkpoints

| Model | Benchmark | Success Rate |
|-------|-----------|--------------|
| π₀.₅-LIBERO | LIBERO Spatial | 98.8% |
| π₀.₅-LIBERO | LIBERO Object | 98.2% |
| π₀.₅-LIBERO | LIBERO Goal | 98.0% |
| π₀.₅-LIBERO | LIBERO-10 | 92.4% |
| π₀.₅-LIBERO | **Average** | **96.85%** |

## Evaluation Script Options

```bash
python examples/aloha_sim/evaluate.py --help
```

| Option | Default | Description |
|--------|---------|-------------|
| `--args.task` | `gym_aloha/AlohaTransferCube-v0` | Task environment |
| `--args.prompt` | "Pick up the cube..." | Language instruction |
| `--args.num-episodes` | 50 | Number of episodes |
| `--args.max-steps` | 400 | Max steps per episode |
| `--args.seed` | 0 | Random seed |
| `--args.host` | 0.0.0.0 | Policy server host |
| `--args.port` | 8000 | Policy server port |

## Interpreting Results

- **Success Rate**: Percentage of episodes where reward >= 0.95
- **Max Reward**: 2.0 = perfect task completion, 1.0 = partial success, 0.0 = failure
- **Steps**: Number of environment steps taken (max 400)

## Troubleshooting

### Server not responding
```bash
# Check server logs
sudo docker logs openpi_server

# Restart server
sudo docker stop openpi_server
# Re-run the start command
```

### Out of GPU memory
- Ensure only one evaluation is running at a time
- Check GPU usage: `nvidia-smi`

### Evaluation stuck
- The first inference takes ~30-60 seconds for model warmup
- Each episode takes ~50 seconds
