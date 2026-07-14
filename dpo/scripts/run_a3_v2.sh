#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${DPO_DIR}/.." && pwd)"
source "${PROJECT_ROOT}/scripts/common_env.sh"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

COMMAND="${1:-all}"
shift || true

DATA_ROOT="${A3_V2_DATA_ROOT:-${PROJECT_ROOT}/dpo/data/a3_v2_runs}"
OUTPUT_ROOT="${A3_V2_OUTPUT_ROOT:-${PROJECT_ROOT}/dpo/outputs/a3_v2_runs}"

if [[ -z "${RUN_ID:-}" ]]; then
  if [[ "${COMMAND}" == "prepare" || "${COMMAND}" == "all" ]]; then
    export RUN_ID="a3_v2_$(date +%Y%m%d_%H%M%S)"
  else
    if [[ -d "${DATA_ROOT}" ]] && compgen -G "${DATA_ROOT}/*" > /dev/null; then
      export RUN_ID="$(ls -1dt "${DATA_ROOT}"/*/ | head -1 | xargs -n1 basename)"
      echo "RUN_ID was not set; using latest A3 v2 run: ${RUN_ID}"
    else
      echo "RUN_ID is required for '${COMMAND}' when no previous A3 v2 run exists." >&2
      exit 2
    fi
  fi
fi

mkdir -p "${OUTPUT_ROOT}/${RUN_ID}/logs"
cd "${PROJECT_ROOT}"

echo "project_root=${PROJECT_ROOT}"
echo "run_id=${RUN_ID}"
echo "command=${COMMAND}"
echo "python=${PYTHON_BIN}"
echo "data_root=${DATA_ROOT}"
echo "output_root=${OUTPUT_ROOT}"

run_prepare() {
  "${PYTHON_BIN}" -m dpo.a3_v2.pipeline prepare \
    --run_id "${RUN_ID}" \
    --data_root "${DATA_ROOT}" \
    --output_root "${OUTPUT_ROOT}" \
    "$@"
}

run_validate() {
  "${PYTHON_BIN}" -m dpo.a3_v2.pipeline validate \
    --run_id "${RUN_ID}" \
    --data_root "${DATA_ROOT}" \
    "$@"
}

case "${COMMAND}" in
  prepare)
    run_prepare "$@"
    ;;
  validate)
    run_validate "$@"
    ;;
  probe)
    "${SCRIPT_DIR}/validate_a3_a4_bridge.sh" "$@"
    ;;
  all)
    run_prepare "$@"
    run_validate
    "${SCRIPT_DIR}/validate_a3_a4_bridge.sh"
    ;;
  *)
    echo "Usage: RUN_ID=<id> bash dpo/scripts/run_a3_v2.sh {prepare|validate|probe|all}" >&2
    exit 2
    ;;
esac

echo "A3 v2 ${COMMAND} finished for RUN_ID=${RUN_ID}"
echo "Data: ${DATA_ROOT}/${RUN_ID}"
echo "Logs: ${OUTPUT_ROOT}/${RUN_ID}/logs"
