#!/usr/bin/env bash
# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Start an openpi (pi0 / pi0.5) policy server for DexVerse online evaluation.
#
# Runs scripts/serve_policy.py from an openpi checkout, which exposes the
# websocket protocol that openpi_client speaks. The DexVerse-side data config
# sets prompt_from_task=True, so the client-sent "prompt" string reaches the
# model directly and no task_index lookup happens server-side.
#
# This must be its own process in the openpi environment: do NOT try to import
# isaaclab into the same interpreter. scripts/eval/eval_policy.py is the client.
#
# Required environment variables (no defaults — set them for your machine):
#   OPENPI_ROOT    path to your openpi checkout (source of the config + policies)
#   PYTHON_BIN     python interpreter with openpi's torch/JAX deps
#   CKPT_DIR       checkpoint dir (must contain model.safetensors)
#   CONFIG_NAME    TrainConfig name registered in openpi's training/config.py,
#                  e.g. the single-hand or bimanual DexBench config
#
# Optional:
#   PORT=8000
#
# Example:
#   OPENPI_ROOT=/path/to/openpi \
#   PYTHON_BIN=/path/to/openpi/.venv/bin/python \
#   CKPT_DIR=/path/to/openpi/checkpoints/<config>/<run>/<step> \
#   CONFIG_NAME=pi05_dexbench \
#       bash scripts/eval/serve_pi0.sh
set -euo pipefail

require_var() {
    local name="$1"
    if [[ -z "${!name:-}" ]]; then
        echo "[serve_pi0] required environment variable ${name} is not set." >&2
        echo "            See the header of $0 for the full list." >&2
        exit 1
    fi
}

require_var OPENPI_ROOT
require_var PYTHON_BIN
require_var CKPT_DIR
require_var CONFIG_NAME

PORT="${PORT:-8000}"

if [[ ! -d "${OPENPI_ROOT}" ]]; then
    echo "[serve_pi0] OPENPI_ROOT is not a directory: ${OPENPI_ROOT}" >&2
    exit 1
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "[serve_pi0] PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
    exit 1
fi
if [[ ! -d "${CKPT_DIR}" ]]; then
    echo "[serve_pi0] checkpoint dir not found: ${CKPT_DIR}" >&2
    exit 1
fi
if [[ ! -f "${CKPT_DIR}/model.safetensors" ]]; then
    echo "[serve_pi0] no model.safetensors under ${CKPT_DIR}" >&2
    exit 1
fi

cd "${OPENPI_ROOT}"

# Prepend openpi's src so `import openpi.training.config` resolves to THIS
# checkout (where the DexBench config is registered) rather than to whatever a
# separate pip-installed openpi in the interpreter points at. openpi's own
# CLAUDE.md flags this shadowing as a recurring footgun.
export PYTHONPATH="${OPENPI_ROOT}/src:${PYTHONPATH:-}"

echo "[serve_pi0] root=${OPENPI_ROOT}"
echo "[serve_pi0] python=${PYTHON_BIN}"
echo "[serve_pi0] config=${CONFIG_NAME}"
echo "[serve_pi0] ckpt=${CKPT_DIR}"
echo "[serve_pi0] port=${PORT}"
echo "[serve_pi0] PYTHONPATH=${PYTHONPATH}"

exec "${PYTHON_BIN}" scripts/serve_policy.py \
    --port "${PORT}" \
    policy:checkpoint \
    --policy.config "${CONFIG_NAME}" \
    --policy.dir "${CKPT_DIR}"
