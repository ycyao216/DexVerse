#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEXVERSE_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
CONDA_SETUP="${CONDA_SETUP:-$HOME/miniconda3/etc/profile.d/conda.sh}"
source "$CONDA_SETUP"
conda activate lerobot

TASK_NAME="${TASK_NAME:-Dexverse-PickCube-v0}"
INPUT_DIR="${INPUT_DIR:-$DEXVERSE_DIR/demo/visionpro_test/$TASK_NAME}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DEXVERSE_DIR/lerobot_datasets/$TASK_NAME}"

exec python "$SCRIPT_DIR/dexverse_h5_to_lerobot.py" \
    --input-dir "$INPUT_DIR" \
    --output-root "$OUTPUT_ROOT" \
    --repo-id "${REPO_ID:-$TASK_NAME}" \
    --fps "${FPS:-60}" \
    --state-dim "${STATE_DIM:-28}" \
    --overwrite
