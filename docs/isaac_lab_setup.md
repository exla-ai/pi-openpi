# Isaac Lab Setup Guide

This guide covers setting up Isaac Lab for RECAP data collection and evaluation.

## Prerequisites

- NVIDIA GPU with CUDA support (RTX 30xx/40xx or A100/H100)
- Ubuntu 22.04 LTS
- Python 3.10+
- At least 32GB RAM
- At least 100GB disk space for Isaac Sim

## Installing Isaac Sim

Isaac Lab requires NVIDIA Isaac Sim as its simulation backend.

### Option 1: Pip Installation (Recommended)

```bash
# Create a virtual environment
conda create -n isaaclab python=3.10
conda activate isaaclab

# Install Isaac Sim via pip
pip install isaacsim-rl isaacsim-replicator isaacsim-extscache-physics isaacsim-extscache-kit-sdk isaacsim-extscache-kit isaacsim-app --extra-index-url https://pypi.nvidia.com
```

### Option 2: Omniverse Launcher

1. Download NVIDIA Omniverse Launcher from https://www.nvidia.com/en-us/omniverse/
2. Install Isaac Sim 2023.1.1 or later through the Launcher
3. Set `ISAACSIM_PATH` to the installation directory

## Installing Isaac Lab

```bash
# Clone Isaac Lab
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# Create conda environment
conda create -n isaaclab python=3.10
conda activate isaaclab

# Install Isaac Lab
./isaaclab.sh --install  # For pip-installed Isaac Sim
# OR
./isaaclab.sh --install --isaacsim_path /path/to/isaac_sim  # For Omniverse installation
```

## Verifying Installation

```bash
# Test basic Isaac Lab functionality
cd /path/to/IsaacLab

# Run a simple environment
python source/standalone/tutorials/00_sim/create_empty.py

# Test a manipulation task
python source/standalone/workflows/rl_games/train.py --task Isaac-Cartpole-v0 --headless
```

## Available Environments for RECAP

Isaac Lab provides several manipulation environments suitable for RECAP training:

### Franka Manipulation Tasks
- `Isaac-Franka-Cabinet-Direct-v0` - Open a cabinet drawer
- `Isaac-Lift-Franka-v0` - Lift an object
- `Isaac-Reach-Franka-v0` - Reach to a target position
- `Isaac-Pick-Franka-v0` - Pick up an object

### UR5/UR10 Tasks
- `Isaac-Lift-UR10-v0` - Lift with UR10 robot
- `Isaac-Reach-UR5-v0` - Reach with UR5 robot

### Allegro Hand Tasks
- `Isaac-Allegro-Hand-v0` - Dexterous manipulation

## Setting Up for RECAP Data Collection

### Environment Variables

Set these environment variables for easier configuration:

```bash
export RECAP_DATA_DIR="/path/to/data/isaaclab"
export RECAP_CHECKPOINT_PATH="/path/to/checkpoint"
export RECAP_TASK="Isaac-Franka-Cabinet-Direct-v0"
export RECAP_NUM_ENVS=16
```

### Running Data Collection

```bash
# From the openpi directory
cd /path/to/openpi

# Collect data with random policy (initial exploration)
python scripts/isaaclab_data_collection.py \
    --task Isaac-Franka-Cabinet-Direct-v0 \
    --num_episodes 100 \
    --num_envs 16 \
    --headless

# Collect data with trained policy (for RECAP iterations)
python scripts/isaaclab_data_collection.py \
    --task Isaac-Franka-Cabinet-Direct-v0 \
    --num_episodes 100 \
    --policy checkpoint \
    --checkpoint_path checkpoints/recap_full/experiment_name/checkpoint_500 \
    --headless
```

## Troubleshooting

### Common Issues

**"Unable to initialize backend 'cuda'"**
- Ensure NVIDIA drivers are installed: `nvidia-smi`
- Check CUDA version compatibility with Isaac Sim

**"No display found"**
- Use `--headless` flag for server environments
- Or set `DISPLAY=:0` if using X11 forwarding

**"Out of memory"**
- Reduce `--num_envs` to decrease GPU memory usage
- Use `--image_width 128 --image_height 128` for smaller images

**"ModuleNotFoundError: No module named 'isaaclab'"**
- Ensure the Isaac Lab conda environment is activated
- Run `./isaaclab.sh --install` again if needed

### Camera Configuration

Isaac Lab uses TiledCamera for efficient batched rendering:

```python
# Camera is automatically added by the data collection script
# You can configure camera parameters via command line:
--image_width 224
--image_height 224
--cameras base_0_rgb,left_wrist_0_rgb
```

## Integration with RECAP Training

Once data is collected, train RECAP:

```bash
# Using Isaac Lab data
python scripts/train_recap_full.py \
    --config recap_aloha_sim \
    --experiment_name isaaclab_franka

# Resume from checkpoint
python scripts/train_recap_full.py \
    --config recap_aloha_sim \
    --resume_from checkpoints/recap_full/isaaclab_franka/checkpoint_500 \
    --skip_value_training
```

## Resources

- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/)
- [Isaac Sim Documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim.html)
- [NVIDIA Forums](https://forums.developer.nvidia.com/c/simulation-and-digital-twins/isaac-sim/)
