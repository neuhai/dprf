TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="results/interview_deepseek_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"
echo "Starting Python script..."
python Evaluation/interview/interview.py \
  --task_model_type sglang \
  --task_model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --refiner_model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --refiner_model_type sglang \
  --output_dir "${OUTPUT_DIR}" \
  --iterations 15 \
  --analysis_prompt_file prompts/analysis_structured.txt \
  --refinement_prompt_file prompts/refinement.txt \
  --instruction_prompt_file Evaluation/interview/prompts/instruction.txt \
  --initial_persona_file Evaluation/interview/prompts/initial_persona.txt \
  --data_dir Evaluation/interview/data/processed

echo "Slurm script finished."
