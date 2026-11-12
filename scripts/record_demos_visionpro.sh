#!/usr/bin/env bash
set -euo pipefail

# Default values may be edited here or overridden with environment variables.
VISIONPRO_IP="${VISIONPRO_IP:-192.168.3.37}"
VISIONPRO_YAW_OFFSET_DEG="${VISIONPRO_YAW_OFFSET_DEG:--90.0}"
TASK="${TASK:-Dexverse-PickCube-v0}"
DATASET_DIR="${DATASET_DIR:-visionpro_test}"
NUM_DEMOS="${NUM_DEMOS:-5}"
NUM_SUCCESS_STEPS="${NUM_SUCCESS_STEPS:-10}"
TELEOP_RETARGETER="${TELEOP_RETARGETER:-relative}"
RETARGETING_SCHEME="${RETARGETING_SCHEME:-dexpilot}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEXVERSE_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
CONDA_SETUP="${CONDA_SETUP:-$HOME/miniconda3/etc/profile.d/conda.sh}"

if [[ ! -f "$CONDA_SETUP" ]]; then
    echo "Error: Conda initialization script not found: $CONDA_SETUP" >&2
    exit 1
fi

if [[ ! -f "$DEXVERSE_DIR/scripts/record_demos.py" ]]; then
    echo "Error: DexVerse record_demos.py not found." >&2
    exit 1
fi

# shellcheck source=/dev/null
source "$CONDA_SETUP"
conda activate dexverse

export VISIONPRO_IP VISIONPRO_YAW_OFFSET_DEG

echo "Starting Vision Pro teleoperation data collection:"
echo "  Vision Pro IP: $VISIONPRO_IP"
echo "  Scene yaw offset: ${VISIONPRO_YAW_OFFSET_DEG} degrees"
echo "  Task: $TASK"
echo "  Dataset directory: $DATASET_DIR"
echo "  Target demonstrations: $NUM_DEMOS"
echo "  Consecutive success steps: $NUM_SUCCESS_STEPS"
echo ""
echo "Press START in Tracking Streamer before launching the simulation."
echo "After Isaac Sim opens, focus the main window: S=start/calibrate, P=pause, R=reset, Q=quit."

cd "$DEXVERSE_DIR"
exec python scripts/record_demos.py \
    --task "$TASK" \
    --teleop_device visionpro \
    --enable_cameras \
    --dataset_dir "$DATASET_DIR" \
    --num_demos "$NUM_DEMOS" \
    --num_success_steps "$NUM_SUCCESS_STEPS" \
    --teleop_retargeter "$TELEOP_RETARGETER" \
    --retargeting_scheme "$RETARGETING_SCHEME" \
    "$@"
