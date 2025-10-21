TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="results/imdb_qwen_brief_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"
echo "Starting Python script..."
python Evaluation/imdb/imdb.py \
  --task_model_type sglang \
  --task_model Qwen/Qwen2.5-7B-Instruct \
  --refiner_model Qwen/Qwen2.5-7B-Instruct \
  --refiner_model_type sglang \
  --output_dir "${OUTPUT_DIR}" \
  --iterations 15 \
  --analysis_prompt_file prompts/analysis_brief.txt \
  --refinement_prompt_file prompts/refinement.txt \
  --instruction_prompt_file Evaluation/imdb/prompts/instruction.txt \
  --initial_persona_file Evaluation/imdb/prompts/initial_persona.txt \
  --data_dir Evaluation/imdb/data/processed

echo "Slurm script finished."
