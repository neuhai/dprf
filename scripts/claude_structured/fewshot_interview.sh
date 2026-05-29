#!/bin/bash
AWS_ACCOUNT_ID=
BEDROCK_CLAUDE_MODEL_ID=
BEDROCK_AWS_REGION=
MODEL_KWARGS_JSON='{"claude_temperature": 0.6, "claude_max_tokens": 2000, "bedrock_max_attempts": 100, "claude_top_p": 0.95}'
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="results/fewshot_interview_claude_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"
export WANDB_API_KEY="2c389cc27b41b9449eb11d8a0cd28723808e1df7"
echo "Few-shot interview: eval on FIRST 100 speakers, shots from LAST 100 pool..."
python Evaluation/interview/interview.py \
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
  --instruction_prompt_file Evaluation/interview/prompts/instruction_few_shot.txt \
  --few_shot_examples_file Evaluation/interview/data/few_shot_examples.json \
  --initial_persona_file Evaluation/interview/prompts/initial_persona.txt \
  --data_dir Evaluation/interview/data/processed \
  --bedrock_region "${BEDROCK_AWS_REGION}" \
  --model_kwargs_json "${MODEL_KWARGS_JSON}"
