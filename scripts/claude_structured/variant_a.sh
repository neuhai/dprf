#!/bin/bash
# Variant A: combined analysis+refinement in one call (analysis dimensions as guidance, no separate analysis output)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
source "${SCRIPT_DIR}/config.sh"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
if [ -n "${WANDB_API_KEY:-}" ]; then export WANDB_API_KEY; fi

echo "=== Variant A: Analysis+Refinement combined (interview) ==="
OUTPUT_DIR="results/variant_a_interview_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"
python Evaluation/interview/interview.py \
  --task_model_type bedrock \
  --refiner_model_type bedrock \
  --task_model "${BEDROCK_CLAUDE_MODEL_ID}" \
  --refiner_model "${BEDROCK_CLAUDE_MODEL_ID}" \
  --output_dir "${OUTPUT_DIR}" \
  --length "${DPRF_LENGTH}" \
  --seed "${DPRF_SEED}" \
  --iterations "${DPRF_ITERATIONS}" \
  --direct_refinement_prompt_file prompts/analysis_refinement.txt \
  --instruction_prompt_file Evaluation/interview/prompts/instruction.txt \
  --initial_persona_file Evaluation/interview/prompts/initial_persona.txt \
  --data_dir Evaluation/interview/data/processed \
  --bedrock_region "${BEDROCK_AWS_REGION}" \
  --model_kwargs_json "${MODEL_KWARGS_JSON}" \
  --wandb_project dprf \
  --wandb_run_name variant_a

echo "=== Variant A: Analysis+Refinement combined (depression) ==="
OUTPUT_DIR="results/variant_a_depression_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"
python Evaluation/depression/depression.py \
  --task_model_type bedrock \
  --refiner_model_type bedrock \
  --task_model "${BEDROCK_CLAUDE_MODEL_ID}" \
  --refiner_model "${BEDROCK_CLAUDE_MODEL_ID}" \
  --output_dir "${OUTPUT_DIR}" \
  --length "${DPRF_LENGTH}" \
  --seed "${DPRF_SEED}" \
  --iterations "${DPRF_ITERATIONS}" \
  --direct_refinement_prompt_file prompts/analysis_refinement.txt \
  --instruction_prompt_file Evaluation/depression/prompts/instruction.txt \
  --initial_persona_file Evaluation/depression/prompts/initial_persona.txt \
  --data_dir Evaluation/depression/data/processed \
  --bedrock_region "${BEDROCK_AWS_REGION}" \
  --model_kwargs_json "${MODEL_KWARGS_JSON}" \
  --wandb_project dprf \
  --wandb_run_name variant_a
