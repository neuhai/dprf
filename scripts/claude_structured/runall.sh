#!/bin/bash
# Launch rebuttal / ICL / few-shot experiments in parallel.
# Logs: results/runall_<timestamp>/logs/<script>.log

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="results/runall_${TIMESTAMP}"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"

SCRIPTS=(
  rebuttal_debate.sh
  rebuttal_interview.sh
  icl_baseline.sh
  fewshot_interview.sh
  fewshot_depression.sh
)

echo "Run-all: ${#SCRIPTS[@]} jobs in parallel"
echo "Project root: ${PROJECT_ROOT}"
echo "Logs: ${LOG_DIR}"
echo ""

declare -a PIDS=()
declare -a NAMES=()

for script in "${SCRIPTS[@]}"; do
  name="${script%.sh}"
  log_file="${LOG_DIR}/${name}.log"
  echo "  -> ${script} (log: ${log_file})"
  bash "${SCRIPT_DIR}/${script}" > "${log_file}" 2>&1 &
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
