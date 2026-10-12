#!/usr/bin/env bash
# Copy the pi05_rlt adapter into a RoboDojo/XPolicyLab checkout and verify
# that the user's checkpoints are configured. Idempotent; no vendoring of
# RoboDojo code — the adapter only lands inside the official clone.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADAPTER_SRC="${SCRIPT_DIR}/xpolicylab_adapter/pi05_rlt"

usage() {
  cat <<'EOF'
Usage: bash install_adapter.sh <robodojo_root> \
           [--pi05-path PATH] [--rlt-checkpoint PATH] [--actor-checkpoint PATH] \
           [--stats-path PATH]

<robodojo_root>  path to a clone of RoboDojo-Benchmark/RoboDojo whose
                 XPolicyLab/ submodule is checked out (policy/ present).

Checkpoint flags fill deploy.yml before copying; each path must exist.
EOF
}

if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
  usage; exit 2
fi
ROBODOJO_ROOT="$(cd "$1" && pwd)"; shift

XPL_POLICY_DIR="${ROBODOJO_ROOT}/XPolicyLab/policy/Pi05_RLT"
if [[ ! -d "${ROBODOJO_ROOT}/XPolicyLab/policy" ]]; then
  echo "[ERROR] ${ROBODOJO_ROOT}/XPolicyLab/policy not found." >&2
  echo "[ERROR] Clone RoboDojo and init its XPolicyLab submodule first:" >&2
  echo "[ERROR]   git clone https://github.com/RoboDojo-Benchmark/RoboDojo" >&2
  echo "[ERROR]   git -C RoboDojo submodule update --init XPolicyLab" >&2
  exit 1
fi

TMP_YML="$(mktemp)"
cp "${ADAPTER_SRC}/deploy.yml" "${TMP_YML}"
missing=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pi05-path|--rlt-checkpoint|--actor-checkpoint|--stats-path)
      key="$1"; shift
      [[ $# -ge 1 && "$1" != --* ]] || { usage; exit 2; }
      path="$1"; shift
      if [[ ! -e "${path}" ]]; then
        missing+=("${key} ${path}")
        continue
      fi
      ykey="${key#--}"; ykey="${ykey//-/_}"
      sed -i.bak -E "s|^${ykey}:.*|${ykey}: ${path}|" "${TMP_YML}" && rm -f "${TMP_YML}.bak"
      ;;
    *) usage; exit 2 ;;
  esac
done

if [[ ${#missing[@]} -gt 0 ]]; then
  echo "[ERROR] checkpoint path(s) do not exist:" >&2
  printf '       %s\n' "${missing[@]}" >&2
  exit 1
fi

mkdir -p "${XPL_POLICY_DIR}"
cp "${TMP_YML}" "${XPL_POLICY_DIR}/deploy.yml"
rm -f "${TMP_YML}"
cp "${ADAPTER_SRC}/model.py" "${ADAPTER_SRC}/deploy.py" "${ADAPTER_SRC}/__init__.py" \
   "${ADAPTER_SRC}/README.md" "${XPL_POLICY_DIR}/"
for f in eval.sh setup_eval_policy_server.sh setup_eval_env_client.sh install.sh; do
  if [[ ! -f "${XPL_POLICY_DIR}/${f}" ]]; then
    cp "${ROBODOJO_ROOT}/XPolicyLab/policy/demo_policy/${f}" "${XPL_POLICY_DIR}/${f}"
  fi
done

python3 -m py_compile "${XPL_POLICY_DIR}/model.py" "${XPL_POLICY_DIR}/deploy.py" \
  && echo "[OK] adapter installed at ${XPL_POLICY_DIR} (py_compile passed)"
echo "[NOTE] launch commands: docs/benchmark_robodojo.md"
