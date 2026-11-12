#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEXVERSE_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
CONDA_SETUP="${CONDA_SETUP:-$HOME/miniconda3/etc/profile.d/conda.sh}"
TASK_NAME="${TASK_NAME:-Dexverse-PickCube-v0}"
DATASET_ROOT="${DATASET_ROOT:-$DEXVERSE_DIR/lerobot_datasets/$TASK_NAME}"
OUTPUT_DIR="${OUTPUT_DIR:-$DEXVERSE_DIR/hf_ckpts/finetune/pi05/$TASK_NAME-pi05}"
PI05_DEVICE="${PI05_DEVICE:-cuda}"

source "$CONDA_SETUP"
conda activate lerobot

cd "$DEXVERSE_DIR"
exec lerobot-train \
    --policy.push_to_hub=false \
    --dataset.repo_id="$TASK_NAME" \
    --dataset.root="$DATASET_ROOT" \
    --policy.type=pi05 \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.dtype=bfloat16 \
    --policy.device="$PI05_DEVICE" \
    --peft.method_type=LORA \
    --peft.r=16 \
    --peft.lora_alpha=16 \
    --output_dir="$OUTPUT_DIR" \
    --job_name="$TASK_NAME-pi05" \
    --batch_size=1 \
    --steps=20000 \
    --save_freq=5000 \
    --policy.n_obs_steps=3 \
    --policy.gradient_checkpointing=true \
    --policy.freeze_vision_encoder=true \
