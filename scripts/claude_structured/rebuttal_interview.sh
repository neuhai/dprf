#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
source "${SCRIPT_DIR}/config.sh"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="results/rebuttal_interview_claude_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"
if [ -n "${WANDB_API_KEY:-}" ]; then export WANDB_API_KEY; fi

echo "Starting Bedrock (Claude) on Interview task with train/val split (processed_val2)..."
python Evaluation/interview/interview.py \
  --task_model_type bedrock \
  --refiner_model_type bedrock \
  --task_model "${BEDROCK_CLAUDE_MODEL_ID}" \
  --refiner_model "${BEDROCK_CLAUDE_MODEL_ID}" \
  --output_dir "${OUTPUT_DIR}" \
  --length 100 \
  --iterations 20 \
  --analysis_prompt_file prompts/analysis_structured.txt \
  --refinement_prompt_file prompts/refinement.txt \
  --instruction_prompt_file Evaluation/interview/prompts/instruction.txt \
  --initial_persona_file Evaluation/interview/prompts/initial_persona.txt \
  --data_dir Evaluation/interview/data/processed_val2 \
  --bedrock_region "${BEDROCK_AWS_REGION}" \
  --model_kwargs_json "${MODEL_KWARGS_JSON}"
