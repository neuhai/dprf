#!/usr/bin/env python3
"""
Debate DPRF Evaluation Script (DPRF2 Version)

This script evaluates the DPRF framework on the Intelligence Squared Debates dataset,
using the generic BaseEvaluator.
"""
import argparse
import json
import os
import sys
import asyncio
import wandb
import glob
import time
from datetime import datetime 
# Add project root and evaluate directory to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
evaluate_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(evaluate_dir)) # Add parent of debate dir (Evaluate dir)

from evaluate import BaseEvaluator
from core.utils import ensure_directory

class DebateEvaluator(BaseEvaluator):
    """
    Evaluator for DPRF on the Intelligence Squared Debates dataset.
    Inherits from BaseEvaluator and handles debate-specific data processing.
    """
    def __init__(self,
                 data_dir="Evaluation/debate/data/processed", # Default relative to project root
                 num_examples_to_process=None, # Process all by default
                 custom_initial_persona_file=None,
                 **kwargs): # Catches all generic args for BaseEvaluator
        
        if not os.path.isabs(data_dir):
            self.data_dir = os.path.join(project_root, data_dir)
        else:
            self.data_dir = data_dir

        self.num_examples_to_process = num_examples_to_process

        self.custom_initial_persona_content = None
        if custom_initial_persona_file:
            resolved_persona_file = custom_initial_persona_file
            if not os.path.isabs(resolved_persona_file):
                resolved_persona_file = os.path.join(project_root, resolved_persona_file)
            if os.path.exists(resolved_persona_file):
                try:
                    with open(resolved_persona_file, 'r') as f:
                        self.custom_initial_persona_content = f.read().strip()
                    print(f"Using custom initial persona from: {resolved_persona_file}")
                except Exception as e:
                    print(f"Warning: Failed to load initial persona from {resolved_persona_file}: {e}")
            else:
                print(f"Warning: Custom initial persona file not found: {resolved_persona_file}")

        super().__init__(**kwargs) # Pass all args to BaseEvaluator
        
        # Key used to extract a unique ID from each example for filename generation
        # and for linking in per_iteration_metrics CSV.
        self.example_id_key_for_filename = "speaker_id" 
        # For _log_iteration_improvements, to exclude from numeric calculation
        self.task_specific_id_keys_for_iteration_df = ["speaker_id", "debate_id", "speaker_type", "debate_title", "debate_topic", "speaker_position"]

    async def load_examples(self):
        import time
        import random
        import wandb
        
        examples_path = os.path.join(self.data_dir, "debate_examples.json")
        if not os.path.exists(examples_path):
            raise FileNotFoundError(f"Debate examples file not found: {examples_path}")

        with open(examples_path, "r") as f:
            all_examples = json.load(f)
        print(f"Loaded {len(all_examples)} total debate examples from {examples_path}")

        # If num_examples_to_process is None or 0, use all examples
        if self.num_examples_to_process is None or self.num_examples_to_process <= 0:
            print(f"Processing all {len(all_examples)} examples")
            examples_to_run = all_examples
        else:
            time_seed = int(time.time())
            random.seed(time_seed)
            
            wandb.log({"random_seed": time_seed})
            print(f"Using time seed: {time_seed}")
            
            num_to_select = min(self.num_examples_to_process, len(all_examples))
            examples_to_run = random.sample(all_examples, num_to_select)
            
            print(f"Randomly selected {len(examples_to_run)} examples using time seed {time_seed}")
        
        seen_speaker_ids = {}
        for example in examples_to_run:
            original_speaker_id = example["speaker_id"]
            
            if original_speaker_id in seen_speaker_ids:
                seen_speaker_ids[original_speaker_id] += 1
                new_speaker_id = f"{original_speaker_id}_{seen_speaker_ids[original_speaker_id]}"
                example["speaker_id"] = new_speaker_id
                print(f"Duplicate speaker_id detected: {original_speaker_id} -> {new_speaker_id}")
            else:
                seen_speaker_ids[original_speaker_id] = 0
        
        return examples_to_run

    async def create_task(self, example_data):
        """
        Process a raw debate example into a structured task for BaseEvaluator.
        """
        debate_topic = example_data["debate_topic"]
        position = example_data["speaker_position"]
        
        # Content for the task
        content = f"The debate topic is: {debate_topic}\n\n"
        content += f"Your position is: {position} the motion."

        # Initial persona
        if self.custom_initial_persona_content:
            initial_persona = self.custom_initial_persona_content
        else:
            initial_persona = "You are a speaker in a formal debate setting." # Default for debate
            print(f"Using default initial persona!")
        
        ground_truth = example_data["consolidated_statements"]
        bio_text = example_data.get("bio") # Will be None if not present

        task_specific_info = {
            "speaker_id": example_data["speaker_id"],
            "debate_id": example_data["debate_id"],
            "debate_title": example_data["debate_title"],
            "debate_topic": debate_topic,
            "speaker_position": position,
            "speaker_type": example_data.get("speaker_type", "N/A")
        }

        return {
            "content": content,
            "initial_persona": initial_persona,
            "ground_truth": ground_truth,
            "bio_text": bio_text,
            "task_specific_info": task_specific_info,
            "example_id_for_filename": str(example_data["speaker_id"]) # Must be a string
        }

def main():
    parser = argparse.ArgumentParser(description="DPRF Evaluation for Debate (DPRF2)")
    
    # Add generic arguments from BaseEvaluator
    BaseEvaluator.add_generic_args(parser)
    
    args = parser.parse_args()
    start_time = datetime.now()
    start_timestamp = time.time()
    print(f"Program started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Load model_kwargs from JSON string or file
    parsed_model_kwargs = {}
    if args.model_kwargs_json:
        try:
            if os.path.exists(args.model_kwargs_json):
                with open(args.model_kwargs_json, 'r') as f:
                    parsed_model_kwargs = json.load(f)
                print(f"Loaded model_kwargs from file: {args.model_kwargs_json}")
            else:
                parsed_model_kwargs = json.loads(args.model_kwargs_json)
                print(f"Loaded model_kwargs from JSON string: {args.model_kwargs_json}")
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse model_kwargs_json: {e}. Using empty dict.")
        except Exception as e:
            print(f"Warning: Error loading model_kwargs_json: {e}. Using empty dict.")

    # Initialize WandB at the beginning
    wandb.init(
        project=args.wandb_project,
        name=f"{args.wandb_run_name}_debate_{args.task_model.replace('/', '-')}",
        notes=args.wandb_notes or "Debate evaluation run (DPRF2 Structure)",
        config={
            "task_model_name": args.task_model,
            "task_model_type": args.task_model_type,
            "refiner_model_name": args.refiner_model,
            "refiner_model_type": args.refiner_model_type,
            "max_iterations": args.iterations,
            "output_dir": args.output_dir,
            "num_examples": args.length,
            "data_dir": args.data_dir
        }
    )

    evaluator = DebateEvaluator(
        data_dir=args.data_dir,
        num_examples_to_process=args.length,
        custom_initial_persona_file=args.initial_persona_file,
        output_dir=args.output_dir,
        task_model_name=args.task_model,
        task_model_type=args.task_model_type,
        refiner_model_name=args.refiner_model,
        refiner_model_type=args.refiner_model_type,
        openai_api_key=args.openai_api_key,
        max_iterations=args.iterations,
        seed=args.seed,
        instruction_prompt_file=args.instruction_prompt_file,
        analysis_prompt_file=args.analysis_prompt_file,
        refinement_prompt_file=args.refinement_prompt_file,
        bedrock_region_name=args.bedrock_region,
        model_kwargs=parsed_model_kwargs,
        wandb_project=args.wandb_project,
        wandb_run_name=f"debate_{args.task_model.replace('/', '-')}",
        wandb_notes=args.wandb_notes or "Debate evaluation run (DPRF2 Structure)"
    )

    async def run_evaluation():
        return await evaluator.evaluate()

    try:
        # Run evaluation
        asyncio.run(run_evaluation())
        
        end_time = datetime.now()
        end_timestamp = time.time()
        duration = end_timestamp - start_timestamp
        
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)
        duration_formatted = f"{hours}h {minutes}m {seconds}s"
        
        print(f"Program ended at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total execution time: {duration_formatted} ({duration:.2f} seconds)")
        
        wandb.log({
            "timing/end_time": end_time.strftime('%Y-%m-%d %H:%M:%S'),
            "timing/duration_seconds": duration,
            "timing/duration_formatted": duration_formatted,
            "timing/duration_hours": duration / 3600
        })
        
        # Upload artifacts to WandB at the end
        print("Uploading artifacts to WandB...")
        artifact = wandb.Artifact(f'{args.output_dir.replace("/", "_")}', type='experiment')
        
        # Add CSV and JSON files
        for pattern in ['*.csv', '*.json']:
            for file_path in glob.glob(os.path.join(args.output_dir, pattern)):
                artifact.add_file(file_path, name=os.path.basename(file_path))
        
        # Add details directory
        details_dir = os.path.join(args.output_dir, "details")
        if os.path.exists(details_dir):
            artifact.add_dir(details_dir, name='details')
            
        # Add logs directory  
        logs_dir = os.path.join(args.output_dir, "logs")
        if os.path.exists(logs_dir):
            artifact.add_dir(logs_dir, name='logs')
            
        wandb.log_artifact(artifact)
        print(f"Artifact uploaded to WandB successfully.")
        
    finally:
        # Ensure WandB run is finished properly
        wandb.finish()

if __name__ == '__main__':
    main() 