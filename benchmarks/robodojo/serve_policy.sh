#!/usr/bin/env bash
# Serve the pi05_rlt XPolicyLab adapter over the official websocket protocol.
#
# Thin wrapper that runs XPolicyLab/setup_policy_server.py exactly as the
# benchmark does (same CLI, same overrides), pointing at the installed
# policy/Pi05_RLT/deploy.yml. The normal path is RoboDojo's own
# `robodojo.sh eval|server` (it invokes the policy's
# setup_eval_policy_server.sh); use this script when you want the server
# under your own shell for debugging or split-machine runs.
#
# Usage: bash serve_policy.sh <robodojo_root> <conda_env_with_torch_vla_rl> \
#            <port> [host]
set -euo pipefail

ROBODOJO_ROOT="$(cd "$1" && pwd)"
POLICY_ENV="$2"
PORT="$3"
HOST="${4:-0.0.0.0}"

XPL_ROOT="${ROBODOJO_ROOT}/XPolicyLab"
POLICY_DIR="${XPL_ROOT}/policy/Pi05_RLT"
YAML_FILE="${POLICY_DIR}/deploy.yml"

if [[ ! -f "${YAML_FILE}" ]]; then
  echo "[ERROR] ${YAML_FILE} missing — run benchmarks/robodojo/install_adapter.sh first" >&2
  exit 1
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found; the VLA_RL stack must be installed in a conda env" >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${POLICY_ENV}"

# PYTHONPATH order mirrors RoboDojo's run_policy_server.sh: benchmark root
# first (XPolicyLab + client_server import from there), VLA_RL src for lerobot.
VLA_RL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${ROBODOJO_ROOT}:${ROBODOJO_ROOT}/XPolicyLab:${VLA_RL_ROOT}/src:${PYTHONPATH:-}"

exec env \
  PYTHONUNBUFFERED=1 \
  python "${XPL_ROOT}/setup_policy_server.py" \
    --config_path "${YAML_FILE}" \
    --overrides \
      port="${PORT}" \
      host="${HOST}" \
      bench_name=RoboDojo \
      task_name=null \
      ckpt_name=pi05_rlt \
      env_cfg_type=arx_x5 \
      seed=0 \
      policy_name=Pi05_RLT \
      action_type=joint
