#!/usr/bin/env bash
# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Evaluate a whole baseline suite, one task per Python process.
#
# eval_policy.py --suite can run every task in a single process, but Isaac Sim's
# USD stage teardown can hang after many sequential gym.make/env.close cycles.
# Giving each task its own process sidesteps that and makes one bad task cost
# only that task. Failures are logged and the sweep continues.
#
# The policy server must already be running (serve_openvla.sh / serve_pi0.sh).
#
# Required:
#   POLICY=openvla|pi0
#   UNNORM_KEY=<key>      (only when POLICY=openvla)
#
# Optional:
#   SUITE=single|bimanual   (default: single)
#   NUM_EPISODES=20  MAX_STEPS=500  SEED=0
#   HOST=localhost   PORT=<backend default>
#   OUTPUT_DIR=runs/<policy>_eval_online
#   OBSERVATION_PRESET=<preset|none>   (default: backend default in eval_policy.py)
#   CUDA_VISIBLE_DEVICES=0             (keep the sim off the policy server's GPU)
#
# Any extra args are forwarded verbatim to eval_policy.py, e.g.
#   POLICY=pi0 bash scripts/eval/run_all_tasks.sh --save_videos
set -euo pipefail

if [[ -z "${POLICY:-}" ]]; then
    echo "[run_all_tasks] POLICY is required (openvla or pi0)." >&2
    exit 1
fi
if [[ "${POLICY}" == "openvla" && -z "${UNNORM_KEY:-}" ]]; then
    echo "[run_all_tasks] POLICY=openvla requires UNNORM_KEY." >&2
    exit 1
fi

SUITE="${SUITE:-single}"
NUM_EPISODES="${NUM_EPISODES:-20}"
MAX_STEPS="${MAX_STEPS:-500}"
SEED="${SEED:-0}"
HOST="${HOST:-localhost}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/${POLICY}_eval_online}"

# Pin Isaac Sim to one GPU by default so it does not contend with the policy
# server, which should be started on a different device.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Single source of truth for the task list: the same table eval_policy.py uses.
mapfile -t TASKS < <(PYTHONPATH="${SCRIPT_DIR}" python -c "
import baseline_tasks
for task in baseline_tasks.get_suite('${SUITE}'):
    print(task.env_id)
")
if [[ ${#TASKS[@]} -eq 0 ]]; then
    echo "[run_all_tasks] no tasks found for suite '${SUITE}'." >&2
    exit 1
fi

EXTRA_ARGS=()
[[ -n "${PORT:-}" ]] && EXTRA_ARGS+=(--port "${PORT}")
[[ -n "${UNNORM_KEY:-}" ]] && EXTRA_ARGS+=(--unnorm_key "${UNNORM_KEY}")
[[ -n "${OBSERVATION_PRESET:-}" ]] && EXTRA_ARGS+=(--observation_preset "${OBSERVATION_PRESET}")

mkdir -p "${OUTPUT_DIR}"
echo "[run_all_tasks] policy=${POLICY} suite=${SUITE} tasks=${#TASKS[@]} output=${OUTPUT_DIR}"

failed=()
for task in "${TASKS[@]}"; do
    echo
    echo "========================================================================"
    echo "[run_all_tasks] ${task} (seed=${SEED}, episodes=${NUM_EPISODES})"
    echo "========================================================================"
    if ! DISPLAY= python scripts/eval/eval_policy.py \
        --policy "${POLICY}" \
        --enable_cameras --headless \
        --host "${HOST}" \
        --task "${task}" \
        --num_episodes "${NUM_EPISODES}" \
        --max_steps "${MAX_STEPS}" \
        --seed "${SEED}" \
        --output_dir "${OUTPUT_DIR}" \
        "${EXTRA_ARGS[@]}" \
        "$@"; then
        echo "[run_all_tasks] ${task} FAILED, continuing"
        failed+=("${task}")
    fi
done

echo
echo "[run_all_tasks] aggregating summary ..."
OUTPUT_DIR="${OUTPUT_DIR}" python -c "
import glob, json, os

out = os.environ['OUTPUT_DIR']
summary = {}
for path in sorted(glob.glob(os.path.join(out, '*', 'metrics.json'))):
    with open(path, encoding='utf-8') as handle:
        summary[os.path.basename(os.path.dirname(path))] = json.load(handle)['metrics']

summary_path = os.path.join(out, 'summary.json')
with open(summary_path, 'w', encoding='utf-8') as handle:
    json.dump(summary, handle, ensure_ascii=False, indent=2)
print(f'Wrote {summary_path}')

rates = [m['success_rate'] for m in summary.values() if m.get('success_rate') is not None]
print('Per-task success rates:')
for task, metrics in summary.items():
    rate = metrics.get('success_rate')
    print(f\"  {task:52s}  success_rate={'N/A' if rate is None else f'{rate:.3f}'}\")
if rates:
    print(f'Mean over {len(rates)} scored task(s): {sum(rates) / len(rates):.3f}')
"

if [[ ${#failed[@]} -gt 0 ]]; then
    echo
    echo "[run_all_tasks] ${#failed[@]} task(s) failed: ${failed[*]}" >&2
    exit 1
fi
