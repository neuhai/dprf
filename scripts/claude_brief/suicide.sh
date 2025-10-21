#!/bin/bash
AWS_ACCOUNT_ID=
BEDROCK_CLAUDE_MODEL_ID=
BEDROCK_AWS_REGION=
MODEL_KWARGS_JSON='{"claude_temperature": 0.6, "claude_max_tokens": 2000, "bedrock_max_attempts": 100, "claude_top_p": 0.95}'
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="results/suicide_claude_brief_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"
export WANDB_API_KEY="2c389cc27b41b9449eb11d8a0cd28723808e1df7" 
echo "Starting Python script for Bedrock (Claude 3.7) on Suicide task..."
python Evaluation/suicide/suicide.py \
  --task_model_type bedrock \
  --refiner_model_type bedrock \
  --task_model "${BEDROCK_CLAUDE_MODEL_ID}" \
  --refiner_model "${BEDROCK_CLAUDE_MODEL_ID}" \
  --output_dir "${OUTPUT_DIR}" \
  --iterations 20 \
  --analysis_prompt_file prompts/analysis_brief.txt \
  --refinement_prompt_file prompts/refinement.txt \
  --instruction_prompt_file Evaluation/suicide/prompts/instruction.txt \
  --initial_persona_file Evaluation/suicide/prompts/initial_persona.txt \
  --data_dir Evaluation/suicide/data/processed \
  --bedrock_region "${BEDROCK_AWS_REGION}" \
  --model_kwargs_json "${MODEL_KWARGS_JSON}"
