#!/usr/bin/env python3
"""
Depression Severity DPRF Evaluation Script (DPRF2 Version)

This script evaluates the DPRF framework on the Reddit Depression Severity dataset,
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
sys.path.insert(0, os.path.dirname(evaluate_dir)) # Add parent of depression dir (Evaluate dir)

from evaluate import BaseEvaluator
from core.utils import ensure_directory

class DepressionSeverityEvaluator(BaseEvaluator):
    """
    Evaluator for DPRF on the Reddit Depression Severity dataset.
    Inherits from BaseEvaluator and handles depression-specific data processing.
    """
    def __init__(self,
                 data_dir="Evaluation/depression/data/processed", # Default relative to project root
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
        self.task_specific_id_keys_for_iteration_df = ["id", "depression_level"]

    async def load_examples(self):
        """
        Load depression examples from the processed data.
        """
        import random
        import time
        
        examples_path = os.path.join(self.data_dir, "depression_data.json")
        if not os.path.exists(examples_path):
            raise FileNotFoundError(f"Depression examples file not found: {examples_path}")

        with open(examples_path, "r") as f:
            all_examples = json.load(f)
        print(f"Loaded {len(all_examples)} total depression examples from {examples_path}")

        from few_shot import select_items

        if self.num_examples_to_process is None or self.num_examples_to_process <= 0:
            print(f"Processing all {len(all_examples)} examples")
            return all_examples

        examples_to_run = select_items(
            all_examples,
            self.num_examples_to_process,
            self.example_select,
            seed=self.seed,
        )
        print(
            f"Selected {len(examples_to_run)} examples "
            f"(mode={self.example_select}, length={self.num_examples_to_process})"
        )
        return examples_to_run

    async def create_task(self, example_data):
        """
        Process a raw depression example into a structured task for BaseEvaluator.
        """
        # Get the example data
        post = example_data["post"]
        depression_level = example_data["depression_level"]
        initial_persona = example_data["initial_persona"]
        
        # Create content from the depression level
        content = f"Your depression severity level is: {depression_level}"
        
        # Use custom initial persona if provided, otherwise use from example
        if self.custom_initial_persona_content:
            persona_to_use = self.custom_initial_persona_content
        else:
            persona_to_use = initial_persona
        
        ground_truth = post
        bio_text = None # Depression examples don't have bio text

        task_specific_info = {
            "id": example_data["id"],
            "depression_level": depression_level
        }

        return {
            "content": content,
            "initial_persona": persona_to_use,
            "ground_truth": ground_truth,
            "bio_text": bio_text,
            "task_specific_info": task_specific_info,
            "example_id_for_filename": str(example_data["id"]) # Must be a string
        }

def main():
    parser = argparse.ArgumentParser(description="DPRF Evaluation for Depression Severity (DPRF2)")
    
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
        name=f"{args.wandb_run_name}_depression_{args.task_model.replace('/', '-')}",
        notes=args.wandb_notes or "Depression severity evaluation run (DPRF2 Structure)",
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

    evaluator = DepressionSeverityEvaluator(
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
        few_shot_examples_file=args.few_shot_examples_file,
        example_select=args.example_select,
        wandb_project=args.wandb_project,
        wandb_run_name=f"depression_{args.task_model.replace('/', '-')}",
        wandb_notes=args.wandb_notes or "Depression severity evaluation run (DPRF2 Structure)"
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
