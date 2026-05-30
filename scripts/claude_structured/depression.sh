#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
source "${SCRIPT_DIR}/config.sh"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="results/depression_claude_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"
if [ -n "${WANDB_API_KEY:-}" ]; then export WANDB_API_KEY; fi

echo "DPRF Depression (full processed, ${DPRF_LENGTH} posts, ${DPRF_ITERATIONS} iterations)..."
python Evaluation/depression/depression.py \
  --task_model_type bedrock \
  --refiner_model_type bedrock \
  --task_model "${BEDROCK_CLAUDE_MODEL_ID}" \
  --refiner_model "${BEDROCK_CLAUDE_MODEL_ID}" \
  --output_dir "${OUTPUT_DIR}" \
  --length "${DPRF_LENGTH}" \
  --seed "${DPRF_SEED}" \
  --iterations "${DPRF_ITERATIONS}" \
  --analysis_prompt_file prompts/analysis_structured.txt \
  --refinement_prompt_file prompts/refinement.txt \
  --instruction_prompt_file Evaluation/depression/prompts/instruction.txt \
  --initial_persona_file Evaluation/depression/prompts/initial_persona.txt \
  --data_dir Evaluation/depression/data/processed \
  --bedrock_region "${BEDROCK_AWS_REGION}" \
  --model_kwargs_json "${MODEL_KWARGS_JSON}"
