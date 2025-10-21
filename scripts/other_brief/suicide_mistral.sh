TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="results/suicide_mistral_brief_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"
echo "Starting Python script..."
python Evaluation/suicide/suicide.py \
  --task_model_type sglang \
  --task_model mistralai/Mistral-7B-Instruct-v0.3 \
  --refiner_model mistralai/Mistral-7B-Instruct-v0.3 \
  --refiner_model_type sglang \
  --output_dir "${OUTPUT_DIR}" \
  --iterations 15 \
  --analysis_prompt_file prompts/analysis_brief.txt \
  --refinement_prompt_file prompts/refinement.txt \
  --instruction_prompt_file Evaluation/suicide/prompts/instruction.txt \
  --initial_persona_file Evaluation/suicide/prompts/initial_persona.txt \
  --data_dir Evaluation/suicide/data/processed

echo "Slurm script finished."
