#!/bin/bash
# ICL Baseline: persona rewrite on TRAIN, evaluate on TRAIN + VAL separately.
# Data: debate_variation.json, interview processed_val2
# Outputs per dataset: evaluation_results_train.csv, evaluation_results_val.csv

BEDROCK_CLAUDE_MODEL_ID=
BEDROCK_AWS_REGION=us-east-1
LENGTH=100
SEED=42
MAX_CONCURRENCY=20

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASE_DIR="results/icl_baseline_${TIMESTAMP}"
mkdir -p "${BASE_DIR}"

COMMON_ARGS=(
  --length "${LENGTH}"
  --seed "${SEED}"
  --model "${BEDROCK_CLAUDE_MODEL_ID}"
  --bedrock_region "${BEDROCK_AWS_REGION}"
  --max_concurrency "${MAX_CONCURRENCY}"
)

echo "ICL Baseline (debate + interview) -> ${BASE_DIR}"

echo "=== Debate (debate_variation.json) ==="
python Evaluation/icl_baseline/run.py \
  --dataset debate \
  --output_dir "${BASE_DIR}/debate" \
  --data_path Evaluation/debate/data/processed/debate_variation.json \
  "${COMMON_ARGS[@]}"

echo "=== Interview (processed_val2) ==="
python Evaluation/icl_baseline/run.py \
  --dataset interview \
  --output_dir "${BASE_DIR}/interview" \
  --data_path Evaluation/interview/data/processed_val2 \
  "${COMMON_ARGS[@]}"

echo "Done. Each subfolder has:"
echo "  evaluation_results_train.csv / evaluation_results_val.csv"
echo "  aggregate_metrics_train.json / aggregate_metrics_val.json"
