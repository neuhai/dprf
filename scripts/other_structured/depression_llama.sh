TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="results/depression_llama_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"
echo "Starting Python script..."
python Evaluation/depression/depression.py \
  --task_model_type sglang \
  --task_model meta-llama/Llama-3.2-3B-Instruct \
  --refiner_model meta-llama/Llama-3.2-3B-Instruct \
  --refiner_model_type sglang \
  --output_dir "${OUTPUT_DIR}" \
  --iterations 15 \
  --analysis_prompt_file prompts/analysis_structured.txt \
  --refinement_prompt_file prompts/refinement.txt \
  --instruction_prompt_file Evaluation/depression/prompts/instruction.txt \
  --initial_persona_file Evaluation/depression/prompts/initial_persona.txt \
  --data_dir Evaluation/depression/data/processed

echo "Slurm script finished."
