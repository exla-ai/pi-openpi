#!/bin/bash
# Monitor training completion and run evaluation automatically
#
# Usage:
#   ./scripts/monitor_and_evaluate.sh
#
# This script will:
# 1. Wait for training to complete (30000 step checkpoint)
# 2. Run comprehensive evaluation on all tasks
# 3. Update README.md with results
# 4. Commit and push the updated results

set -e

CHECKPOINT_BASE="./checkpoints/pi06_multi/pi06_multi_v1"
TARGET_STEP="30000"
CONFIG="pi06_multi"
NUM_EPISODES=50

echo "=========================================="
echo "Pi0.6 Training Monitor & Evaluation"
echo "=========================================="
echo "Checkpoint: ${CHECKPOINT_BASE}"
echo "Target step: ${TARGET_STEP}"
echo "Config: ${CONFIG}"
echo "Episodes per task: ${NUM_EPISODES}"
echo "=========================================="

# Wait for training to complete
echo ""
echo "Waiting for training to reach step ${TARGET_STEP}..."

while true; do
    if [ -d "${CHECKPOINT_BASE}/${TARGET_STEP}" ]; then
        echo ""
        echo "Checkpoint ${TARGET_STEP} found!"
        break
    fi

    # Show current status
    LATEST=$(ls -1 "${CHECKPOINT_BASE}" 2>/dev/null | sort -n | tail -1)
    if [ -n "$LATEST" ]; then
        echo -ne "\rCurrent: ${LATEST}/${TARGET_STEP} steps    "
    fi

    sleep 60  # Check every minute
done

# Wait a bit for checkpoint to be fully written
echo "Waiting for checkpoint to be fully written..."
sleep 30

# Check if training process is still running
if pgrep -f "train.py pi06_multi" > /dev/null; then
    echo "Training process still running, waiting for it to finish..."
    while pgrep -f "train.py pi06_multi" > /dev/null; do
        sleep 30
    done
    echo "Training process completed."
fi

# Run evaluation
echo ""
echo "=========================================="
echo "Running comprehensive evaluation..."
echo "=========================================="

# Set environment for headless rendering
export MUJOCO_GL=egl
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"

python scripts/run_full_evaluation.py \
    --checkpoint_dir "${CHECKPOINT_BASE}/${TARGET_STEP}" \
    --config "${CONFIG}" \
    --num_episodes "${NUM_EPISODES}" \
    --output "evaluation_results.json" \
    --update_readme \
    --readme_path "README.md"

# Commit and push results
echo ""
echo "=========================================="
echo "Committing evaluation results..."
echo "=========================================="

git add README.md evaluation_results.json
git commit -m "Add Pi0.6 evaluation results

- Evaluated on ALOHA sim benchmarks
- ${NUM_EPISODES} episodes per task
- Checkpoint: ${TARGET_STEP} steps

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "=========================================="
echo "Results saved to evaluation_results.json"
echo "README.md updated with benchmark results"
echo ""
echo "To push results, run:"
echo "  git push origin main"
