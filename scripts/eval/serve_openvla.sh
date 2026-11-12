#!/usr/bin/env bash
# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Start an OpenVLA-OFT policy server for DexVerse online evaluation.
#
# Runs vla-scripts/deploy.py from an openvla-oft checkout, which exposes a
# FastAPI /act endpoint taking {full_image, wrist_image(s), state, instruction,
# unnorm_key} and returning an action chunk, already denormalized via the
# dataset_statistics.json under the checkpoint.
#
# This must be its own process in the openvla-oft environment: do NOT try to
# import isaaclab into the same interpreter. scripts/eval/eval_policy.py is the
# HTTP client.
#
# Required environment variables (no defaults — set them for your machine):
#   OPENVLA_OFT_ROOT   path to your openvla-oft checkout
#   PYTHON_BIN         python interpreter of the openvla-oft env
#   CKPT_DIR           checkpoint dir (must contain dataset_statistics.json)
#   UNNORM_KEY         key into dataset_statistics.json (see the warning below)
#
# Optional:
#   EMBODIMENT=single|bimanual   selects NUM_IMAGES_IN_INPUT (default: single)
#   PORT=8777
#   LORA_RANK=32
#   NUM_IMAGES_IN_INPUT          overrides the EMBODIMENT-derived value
#
# Example:
#   OPENVLA_OFT_ROOT=/path/to/openvla-oft \
#   PYTHON_BIN=/path/to/envs/openvla-oft/bin/python \
#   CKPT_DIR=/path/to/openvla-oft/runs/<run>--<step>_chkpt \
#   UNNORM_KEY=dexbench_rlds/single \
#       bash scripts/eval/serve_openvla.sh
set -euo pipefail

require_var() {
    local name="$1"
    if [[ -z "${!name:-}" ]]; then
        echo "[serve_openvla] required environment variable ${name} is not set." >&2
        echo "                See the header of $0 for the full list." >&2
        exit 1
    fi
}

require_var OPENVLA_OFT_ROOT
require_var PYTHON_BIN
require_var CKPT_DIR
require_var UNNORM_KEY

EMBODIMENT="${EMBODIMENT:-single}"
PORT="${PORT:-8777}"
LORA_RANK="${LORA_RANK:-32}"

case "${EMBODIMENT}" in
    single)   DEFAULT_NUM_IMAGES=2 ;;  # third_person + wrist
    bimanual) DEFAULT_NUM_IMAGES=3 ;;  # third_person + left wrist + right wrist
    *)
        echo "[serve_openvla] EMBODIMENT must be 'single' or 'bimanual', got '${EMBODIMENT}'." >&2
        exit 1
        ;;
esac
NUM_IMAGES_IN_INPUT="${NUM_IMAGES_IN_INPUT:-${DEFAULT_NUM_IMAGES}}"

if [[ ! -d "${OPENVLA_OFT_ROOT}" ]]; then
    echo "[serve_openvla] OPENVLA_OFT_ROOT is not a directory: ${OPENVLA_OFT_ROOT}" >&2
    exit 1
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "[serve_openvla] PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
    exit 1
fi
if [[ ! -d "${CKPT_DIR}" ]]; then
    echo "[serve_openvla] checkpoint dir not found: ${CKPT_DIR}" >&2
    exit 1
fi
if [[ ! -f "${CKPT_DIR}/dataset_statistics.json" ]]; then
    echo "[serve_openvla] no dataset_statistics.json under ${CKPT_DIR} — actions cannot be denormalized." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# openvla-oft picks its action/proprio dimensions by sniffing sys.argv at
# import time (prismatic/vla/constants.py::detect_robot_platform). It searches
# the joined argv for "dexbench", "libero", "aloha", "bridge" in that order and
# falls back to LIBERO — 7-dim actions — when nothing matches.
#
# The only argv token carrying that keyword here is --unnorm_key. So the key
# must contain "dexbench" (and additionally "bimanual" for the two-hand model)
# or the server will silently serve the wrong action dimension.
# ---------------------------------------------------------------------------
if [[ "${UNNORM_KEY}" != *dexbench* ]]; then
    echo "[serve_openvla] UNNORM_KEY='${UNNORM_KEY}' does not contain 'dexbench'." >&2
    echo "                openvla-oft's detect_robot_platform() would fall back to LIBERO" >&2
    echo "                constants (7-dim actions). Set OVERRIDE_ARGV_KEYWORD_CHECK=1 only" >&2
    echo "                if you know the platform is selected some other way." >&2
    [[ "${OVERRIDE_ARGV_KEYWORD_CHECK:-0}" == "1" ]] || exit 1
fi
if [[ "${EMBODIMENT}" == "bimanual" && "${UNNORM_KEY}" != *bimanual* ]]; then
    echo "[serve_openvla] EMBODIMENT=bimanual but UNNORM_KEY='${UNNORM_KEY}' lacks 'bimanual';" >&2
    echo "                detect_robot_platform() would select the 28-dim single-hand constants." >&2
    [[ "${OVERRIDE_ARGV_KEYWORD_CHECK:-0}" == "1" ]] || exit 1
fi

cd "${OPENVLA_OFT_ROOT}"

echo "[serve_openvla] root=${OPENVLA_OFT_ROOT}"
echo "[serve_openvla] python=${PYTHON_BIN}"
echo "[serve_openvla] ckpt=${CKPT_DIR}"
echo "[serve_openvla] unnorm_key=${UNNORM_KEY}"
echo "[serve_openvla] embodiment=${EMBODIMENT} num_images_in_input=${NUM_IMAGES_IN_INPUT}"
echo "[serve_openvla] port=${PORT}"

exec "${PYTHON_BIN}" vla-scripts/deploy.py \
    --host 0.0.0.0 --port "${PORT}" \
    --pretrained_checkpoint "${CKPT_DIR}" \
    --use_l1_regression True \
    --use_proprio True \
    --num_images_in_input "${NUM_IMAGES_IN_INPUT}" \
    --center_crop True \
    --lora_rank "${LORA_RANK}" \
    --unnorm_key "${UNNORM_KEY}"
