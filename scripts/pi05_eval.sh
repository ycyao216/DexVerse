#!/usr/bin/env bash

# Evaluate a PI05 policy in DexVerse. The policy server runs in a separate
# environment and communicates with this process through a local Unix socket.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

TASK="${TASK:-Dexverse-PickCube-v0}"
NUM_EPISODES="${NUM_EPISODES:-30}"
OUTPUT_DIR="${REPO_DIR}/outputs/eval"
RENDER_MODE="${RENDER_MODE:-rgb}"  # gui / rgb / headless
SIM_DEVICE="${SIM_DEVICE:-cpu}"    # Physics simulation device; inference is configured by the server.
DOMAIN_RANDOMIZATION_CONFIG="${SCRIPT_DIR}/eval_domain_randomization.yaml"

# Load the socket client as an external policy module and identify the LoRA adapter checkpoint.
POLICY="module"
POLICY_MODULE="${SCRIPT_DIR}/lerobot_pi05_client.py"
CHECKPOINT="${CHECKPOINT:-${REPO_DIR}/hf_ckpts/finetune/pi05/${TASK}-pi05/checkpoints/020000/pretrained_model}"
export LEROBOT_PI05_SOCKET="${LEROBOT_PI05_SOCKET:-/tmp/dexverse_pi05.sock}"
export LEROBOT_PI05_TASK="${LEROBOT_PI05_TASK:-${TASK}}"
export LEROBOT_PI05_KEEP_ALIVE_INTERVAL="${LEROBOT_PI05_KEEP_ALIVE_INTERVAL:-0.5}"

POLICY_ARGS=(--policy "${POLICY}")
if [[ "${POLICY}" == "module" ]]; then
  if [[ -z "${POLICY_MODULE}" || -z "${CHECKPOINT}" ]]; then
    echo "[ERROR] POLICY_MODULE and CHECKPOINT are required when POLICY=module." >&2
    echo "Example: POLICY_MODULE=/path/to/policy.py CHECKPOINT=/path/to/model.ckpt bash pi05_eval.sh" >&2
    echo "Environment-only check: POLICY=zero bash pi05_eval.sh" >&2
    exit 2
  fi
  POLICY_ARGS+=(--policy_module "${POLICY_MODULE}" --checkpoint "${CHECKPOINT}")
elif [[ "${POLICY}" == "checkpoint" ]]; then
  if [[ -z "${CHECKPOINT}" ]]; then
    echo "[ERROR] CHECKPOINT is required when POLICY=checkpoint." >&2
    exit 2
  fi
  POLICY_ARGS+=(--checkpoint "${CHECKPOINT}" --checkpoint_fallback_action zero)
fi

python "${SCRIPT_DIR}/eval.py" \
  --task "${TASK}" \
  --device "${SIM_DEVICE}" \
  "${POLICY_ARGS[@]}" \
  --num_episodes "${NUM_EPISODES}" \
  --observation_preset 3view_rgb \
  --eval_render_mode "${RENDER_MODE}" \
  --domain_randomization_config "${DOMAIN_RANDOMIZATION_CONFIG}" \
  --output_dir "${OUTPUT_DIR}" \
  --video_fps 30 \
  --video_stride 2 \
  "$@"
