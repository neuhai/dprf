#!/usr/bin/env python3
"""
IMDB DPRF Evaluation Script (DPRF2 Version)

This script evaluates the DPRF framework on the IMDB movie review dataset,
using the generic BaseEvaluator.
"""
import argparse
import json
import os
import sys
import asyncio
import wandb
import glob

# Add project root and evaluate directory to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
evaluate_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(evaluate_dir)) # Add parent of imdb dir (Evaluate dir)

from evaluate import BaseEvaluator
from core.utils import ensure_directory

class IMDBEvaluator(BaseEvaluator):
    """
    Evaluator for DPRF on the IMDB movie review dataset.
    Inherits from BaseEvaluator and handles IMDB-specific data processing.
    """
    def __init__(self,
                 data_dir="Evaluation/imdb/data/processed", # Default relative to project root
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
        self.example_id_key_for_filename = "id" 
        # For _log_iteration_improvements, to exclude from numeric calculation
        self.task_specific_id_keys_for_iteration_df = ["id", "sentiment_label"]

    async def load_examples(self):
        """
        Load IMDB examples from the processed data.
        """
        import random
        import time
        
        examples_path = os.path.join(self.data_dir, "imdb_data.json")
        if not os.path.exists(examples_path):
            raise FileNotFoundError(f"IMDB examples file not found: {examples_path}")

        with open(examples_path, "r") as f:
            all_examples = json.load(f)
        print(f"Loaded {len(all_examples)} total IMDB examples from {examples_path}")

        # If num_examples_to_process is None or 0, use all examples
        if self.num_examples_to_process is None or self.num_examples_to_process <= 0:
            print(f"Processing all {len(all_examples)} examples")
            return all_examples
        
        # Randomly select examples
        time_seed = int(time.time())
        random.seed(time_seed)
        print(f"Using time seed: {time_seed}")
        
        num_to_select = min(self.num_examples_to_process, len(all_examples))
        examples_to_run = random.sample(all_examples, num_to_select)
        
        print(f"Randomly selected {len(examples_to_run)} examples using time seed {time_seed}")
        return examples_to_run

    async def create_task(self, example_data):
        """
        Process a raw IMDB example into a structured task for BaseEvaluator.
        """
        # Get the example data
        review_text = example_data["review"]
        sentiment_label = example_data["sentiment_label"]
        
        # Create content from the sentiment label
        if sentiment_label == 1:
            content = "The overall sentiment of this movie review is positive.\n"
        else:
            content = "The overall sentiment of this movie review is negative.\n"
        
        
        # Use custom initial persona if provided, otherwise use default
        if self.custom_initial_persona_content:
            initial_persona = self.custom_initial_persona_content
        else:
            initial_persona = "You are a viewer writing a movie review." # Default for IMDB
        
        ground_truth = review_text
        bio_text = None # IMDB examples don't have bio text

        task_specific_info = {
            "id": example_data["id"],
            "sentiment_label": sentiment_label
        }

        return {
            "content": content,
            "initial_persona": initial_persona,
            "ground_truth": ground_truth,
            "bio_text": bio_text,
            "task_specific_info": task_specific_info,
            "example_id_for_filename": str(example_data["id"]) # Must be a string
        }

def main():
    parser = argparse.ArgumentParser(description="DPRF Evaluation for IMDB (DPRF2)")
    
    # Add generic arguments from BaseEvaluator
    BaseEvaluator.add_generic_args(parser)
    
    args = parser.parse_args()

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
        name=f"{args.wandb_run_name}_imdb_{args.task_model.replace('/', '-')}",
        notes=args.wandb_notes or "IMDB evaluation run (DPRF2 Structure)",
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

    evaluator = IMDBEvaluator(
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
        direct_refinement_prompt_file=args.direct_refinement_prompt_file,
        bedrock_region_name=args.bedrock_region,
        model_kwargs=parsed_model_kwargs,
        wandb_project=args.wandb_project,
        wandb_run_name=f"imdb_{args.task_model.replace('/', '-')}",
        wandb_notes=args.wandb_notes or "IMDB evaluation run (DPRF2 Structure)"
    )

    async def run_evaluation():
        return await evaluator.evaluate()

    try:
        # Run evaluation
        asyncio.run(run_evaluation())
        
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
