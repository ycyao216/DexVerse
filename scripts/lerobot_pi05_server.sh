#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEXVERSE_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PI05_DEVICE="${PI05_DEVICE:-cuda}"
CONDA_SETUP="${CONDA_SETUP:-$HOME/miniconda3/etc/profile.d/conda.sh}"
PI05_TASK="${PI05_TASK:-Dexverse-PickCube-v0}"
PI05_CHECKPOINT="${PI05_CHECKPOINT:-$DEXVERSE_DIR/hf_ckpts/finetune/pi05/$PI05_TASK-pi05/checkpoints/020000/pretrained_model}"
PI05_DATASET_ROOT="${PI05_DATASET_ROOT:-$DEXVERSE_DIR/lerobot_datasets/$PI05_TASK}"
PI05_SOCKET="${PI05_SOCKET:-/tmp/dexverse_pi05.sock}"
source "$CONDA_SETUP"
conda activate lerobot

exec python "$SCRIPT_DIR/lerobot_pi05_server.py" \
    --checkpoint "$PI05_CHECKPOINT" \
    --dataset-root "$PI05_DATASET_ROOT" \
    --dataset-repo-id "$PI05_TASK" \
    --socket-path "$PI05_SOCKET" \
    --device "${PI05_DEVICE}"
