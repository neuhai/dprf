#!/usr/bin/env python3
"""
Interview DPRF Evaluation Script (DPRF2 Version)

This script evaluates the DPRF framework on an interview dataset,
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
sys.path.insert(0, os.path.dirname(evaluate_dir)) # Add parent of interview dir (Evaluate dir)

from evaluate import BaseEvaluator
from core.utils import ensure_directory

class InterviewEvaluator(BaseEvaluator):
    """
    Evaluator for DPRF on the interview dataset.
    Inherits from BaseEvaluator and handles interview-specific data processing.
    """
    def __init__(self,
                 data_dir="Evaluation/interview/data/processed", # Default relative to project root
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
        self.task_specific_id_keys_for_iteration_df = ["id", "speakername"]

    async def load_examples(self):
        """
        Load interview examples from the processed data.
        Randomly selects length JSON files and loads all 5 examples from each file.
        Total examples = length * 5
        """
        # Get all JSON files in the directory
        json_files = [f for f in os.listdir(self.data_dir) if f.endswith(".json")]
        
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {self.data_dir}")
        
        print(f"Found {len(json_files)} JSON files in {self.data_dir}")
        
        from few_shot import select_items

        json_files = sorted(json_files)

        if self.num_examples_to_process is None or self.num_examples_to_process <= 0:
            print(f"Processing all {len(json_files)} files")
            files_to_process = json_files
        else:
            files_to_process = select_items(
                json_files,
                self.num_examples_to_process,
                self.example_select,
                seed=self.seed,
            )
            print(
                f"Selected {len(files_to_process)} JSON files "
                f"(mode={self.example_select}, length={self.num_examples_to_process})"
            )
            print(f"Expected total examples: {len(files_to_process)} files × ~5 examples/file")
            if len(files_to_process) <= 10:
                print(f"Selected files: {', '.join(files_to_process)}")
        
        # Load examples from selected files
        all_examples = []
        for filename in files_to_process:
            try:
                with open(os.path.join(self.data_dir, filename), "r") as f:
                    examples_from_file = json.load(f)
                    all_examples.extend(examples_from_file)
                    print(f"Loaded {len(examples_from_file)} examples from {filename}")
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to decode JSON from {filename}: {e}")
            except Exception as e:
                print(f"Warning: Error loading {filename}: {e}")
        
        if not all_examples:
            raise FileNotFoundError(f"No interview examples found in selected files from {self.data_dir}")
        
        train_count = sum(1 for ex in all_examples if ex.get("split") == "train")
        val_count = sum(1 for ex in all_examples if ex.get("split") == "val")
        if train_count or val_count:
            print(
                f"Loaded split examples: {train_count} train, {val_count} val "
                f"({len(all_examples)} total from {len(files_to_process)} files)"
            )
        else:
            print(f"Total loaded: {len(all_examples)} interview examples from {len(files_to_process)} files")
        
        return all_examples

    async def create_task(self, example_data):
        """
        Process a raw interview example into a structured task for BaseEvaluator.
        """
        # Extract the background conversation
        background = example_data.get("background", [])
        speaker_name = example_data.get("speakername", "")
        bio = example_data.get("bio", "")
        
        # Create the conversation history
        conversation = []
        for turn in background:
            speaker = turn.get("speaker", "")
            text = turn.get("text", "")
            conversation.append(f"{speaker}: {text}")
        
        # Create content from the conversation history
        content = "Here is a conversation excerpt from an interview:\n\n"
        content += "\n".join(conversation)
        
        # Use custom initial persona if provided, otherwise use default
        if self.custom_initial_persona_content:
            initial_persona = self.custom_initial_persona_content
        else:
            initial_persona = f"You are an interviewee in an interview setting. You will respond based on the previous conversation, sharing your perspectives and opinions."
        
        # Ground truth is the expected response
        ground_truth = example_data.get("ground_truth", "")
        split = example_data.get("split")
        if split == "train":
            optimization_ground_truth = ground_truth
            evaluation_ground_truth = ground_truth
        elif split == "val":
            optimization_ground_truth = ""
            evaluation_ground_truth = ground_truth
        else:
            optimization_ground_truth = ground_truth
            evaluation_ground_truth = ground_truth

        bio_text = "You are a composed and well-informed interviewee participating in a interview. " + bio # Interview examples have bio text

        # Generate ID from index if not present
        example_id = example_data.get("id", getattr(self, '_current_example_index', 0))
        
        task_specific_info = {
            "id": example_id,
            "speakername": speaker_name,
            "split": split,
        }

        return {
            "content": content,
            "initial_persona": initial_persona,
            "ground_truth": evaluation_ground_truth or ground_truth,
            "optimization_ground_truth": optimization_ground_truth or ground_truth,
            "evaluation_ground_truth": evaluation_ground_truth or ground_truth,
            "split": split,
            "bio_text": bio_text,
            "task_specific_info": task_specific_info,
            "example_id_for_filename": str(example_id) # Must be a string
        }

def main():
    parser = argparse.ArgumentParser(description="DPRF Evaluation for Interview Dataset (DPRF2)")
    
    # Add generic arguments from BaseEvaluator
    BaseEvaluator.add_generic_args(parser)
    parser.set_defaults(data_dir="Evaluation/interview/data/processed")

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
    _run_prefix = f"{args.wandb_run_name}_" if args.wandb_run_name else ""
    wandb.init(
        project=args.wandb_project,
        name=f"{_run_prefix}interview_{args.task_model}",
        notes=args.wandb_notes or "Interview evaluation run (DPRF2 Structure)",
        config={
            "task_model_name": args.task_model,
            "task_model_type": args.task_model_type,
            "refiner_model_name": args.refiner_model,
            "refiner_model_type": args.refiner_model_type,
            "max_iterations": args.iterations,
            "output_dir": args.output_dir,
            "num_examples": args.length,
            "data_dir": args.data_dir,
            "direct_refinement_prompt_file": args.direct_refinement_prompt_file or "",
        }
    )

    evaluator = InterviewEvaluator(
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
        few_shot_examples_file=args.few_shot_examples_file,
        example_select=args.example_select,
        wandb_project=args.wandb_project,
        wandb_run_name=f"interview_{args.task_model.replace('/', '-')}",
        wandb_notes=args.wandb_notes or "Interview evaluation run (DPRF2 Structure)"
    )

    async def run_evaluation():
        return await evaluator.evaluate()

    try:
        # Use asyncio.run instead of get_event_loop to avoid deprecation warning
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
