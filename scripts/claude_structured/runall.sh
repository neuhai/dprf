#!/bin/bash
# Launch Variant A and Variant B experiments in parallel.
# Fill scripts/claude_structured/config.sh once, then run this script.
# Logs: results/runall_<timestamp>/logs/<script>.log

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# shellcheck source=config.sh
source "${SCRIPT_DIR}/config.sh"

if [ -z "${BEDROCK_CLAUDE_MODEL_ID}" ]; then
  echo "Error: set BEDROCK_CLAUDE_MODEL_ID in ${SCRIPT_DIR}/config.sh"
  exit 1
fi

export AWS_ACCOUNT_ID BEDROCK_CLAUDE_MODEL_ID BEDROCK_AWS_REGION MODEL_KWARGS_JSON
export DPRF_LENGTH DPRF_ITERATIONS DPRF_SEED
export ICL_LENGTH ICL_SEED ICL_MAX_CONCURRENCY
if [ -n "${WANDB_API_KEY:-}" ]; then
  export WANDB_API_KEY
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="results/runall_${TIMESTAMP}"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"

SCRIPTS=(
  variant_a.sh
  variant_b.sh
)

echo "Run-all: ${#SCRIPTS[@]} jobs in parallel"
echo "Model: ${BEDROCK_CLAUDE_MODEL_ID}  Region: ${BEDROCK_AWS_REGION}"
echo "Length: ${DPRF_LENGTH}  Iterations: ${DPRF_ITERATIONS}  Seed: ${DPRF_SEED}"
echo "Project root: ${PROJECT_ROOT}"
echo "Logs: ${LOG_DIR}"
echo ""

declare -a PIDS=()
declare -a NAMES=()

for script in "${SCRIPTS[@]}"; do
  name="${script%.sh}"
  log_file="${LOG_DIR}/${name}.log"
  echo "  -> ${script} (log: ${log_file})"
  (
    bash "${SCRIPT_DIR}/${script}" 2>&1 | tee "${log_file}"
    exit "${PIPESTATUS[0]}"
  ) &
  PIDS+=($!)
  NAMES+=("${name}")
done

echo ""
echo "Waiting for all jobs..."

failed=0
for i in "${!PIDS[@]}"; do
  pid=${PIDS[$i]}
  name=${NAMES[$i]}
  if wait "${pid}"; then
    echo "  OK   ${name}"
  else
    echo "  FAIL ${name} (exit $?; see ${LOG_DIR}/${name}.log)"
    failed=$((failed + 1))
  fi
done

echo ""
if [ "${failed}" -eq 0 ]; then
  echo "All ${#SCRIPTS[@]} jobs finished successfully."
  exit 0
else
  echo "${failed}/${#SCRIPTS[@]} jobs failed."
  exit 1
fi
