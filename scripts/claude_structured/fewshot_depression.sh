#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
source "${SCRIPT_DIR}/config.sh"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="results/fewshot_depression_claude_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"
if [ -n "${WANDB_API_KEY:-}" ]; then export WANDB_API_KEY; fi

echo "Few-shot depression: eval on FIRST 100 posts, shots from LAST 100 pool..."
python Evaluation/depression/depression.py \
  --task_model_type bedrock \
  --refiner_model_type bedrock \
  --task_model "${BEDROCK_CLAUDE_MODEL_ID}" \
  --refiner_model "${BEDROCK_CLAUDE_MODEL_ID}" \
  --output_dir "${OUTPUT_DIR}" \
  --length 100 \
  --example_select first \
  --seed 42 \
  --iterations 20 \
  --analysis_prompt_file prompts/analysis_structured.txt \
  --refinement_prompt_file prompts/refinement.txt \
  --instruction_prompt_file Evaluation/depression/prompts/instruction_few_shot.txt \
  --few_shot_examples_file Evaluation/depression/data/few_shot_examples.json \
  --initial_persona_file Evaluation/depression/prompts/initial_persona.txt \
  --data_dir Evaluation/depression/data/processed \
  --bedrock_region "${BEDROCK_AWS_REGION}" \
  --model_kwargs_json "${MODEL_KWARGS_JSON}"
