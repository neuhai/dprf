#!/bin/bash
# Variant A (split): combined analysis+refinement in one call, 80/20 train/val split
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
source "${SCRIPT_DIR}/config.sh"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
if [ -n "${WANDB_API_KEY:-}" ]; then export WANDB_API_KEY; fi

echo "=== Variant A (split): Analysis+Refinement combined (interview, processed_val2) ==="
OUTPUT_DIR="results/variant_a_split_interview_${TIMESTAMP}"
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
  --data_dir Evaluation/interview/data/processed_val2 \
  --bedrock_region "${BEDROCK_AWS_REGION}" \
  --model_kwargs_json "${MODEL_KWARGS_JSON}" \
  --wandb_project dprf \
  --wandb_run_name variant_a_split

echo "=== Variant A (split): Analysis+Refinement combined (debate, debate_variation.json) ==="
OUTPUT_DIR="results/variant_a_split_debate_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"
python Evaluation/debate/debate.py \
  --task_model_type bedrock \
  --refiner_model_type bedrock \
  --task_model "${BEDROCK_CLAUDE_MODEL_ID}" \
  --refiner_model "${BEDROCK_CLAUDE_MODEL_ID}" \
  --output_dir "${OUTPUT_DIR}" \
  --length "${DPRF_LENGTH}" \
  --seed "${DPRF_SEED}" \
  --iterations "${DPRF_ITERATIONS}" \
  --direct_refinement_prompt_file prompts/analysis_refinement.txt \
  --instruction_prompt_file Evaluation/debate/prompts/instruction.txt \
  --initial_persona_file Evaluation/debate/prompts/initial_persona.txt \
  --debate_data_file Evaluation/debate/data/processed/debate_variation.json \
  --bedrock_region "${BEDROCK_AWS_REGION}" \
  --model_kwargs_json "${MODEL_KWARGS_JSON}" \
  --wandb_project dprf \
  --wandb_run_name variant_a_split
