#!/usr/bin/env python3
"""
Base DPRF Evaluation Script
"""
import traceback
from tqdm.asyncio import trange, tqdm
import argparse
import json
import os
import random
import sys
from pathlib import Path
import wandb
import glob
import asyncio
from bert_score import BERTScorer
import nltk
nltk.download('punkt_tab')
import numpy as np
import pandas as pd
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Add project root to sys.path
# Assuming this script is in DPRF2/Evaluate/ and core is in DPRF2/core
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
from core.dprf_agent import DPRFAgent
from core.utils import ensure_directory
from Evaluation.metrics.bart_score import BARTScorer

class BaseEvaluator:
    """
    Base evaluator for DPRF framework.
    Provides generic evaluation functionalities.
    """

    def __init__(
        self,
        output_dir="results",
        task_model_name="gpt-4o",
        task_model_type="openai",
        refiner_model_name="gpt-4o",
        refiner_model_type="openai",
        openai_api_key=None,
        max_iterations=3,
        seed=42,
        initial_persona_file=None,
        instruction_prompt_file=None, # Add file path for instruction template
        analysis_prompt_file=None, # Add file path for analysis template
        refinement_prompt_file=None, # Add file path for refinement template
        direct_refinement_prompt_file=None, # Variant A or B: skip analysis step
        bedrock_region_name: str = 'us-east-1',
        model_kwargs: dict = None,
        bert_score_device=None,
        bart_score_batch_size=8,
        wandb_project="dprf",
        wandb_run_name="evaluation_run",
        wandb_notes="",
        few_shot_examples_file=None,
        example_select="random",
    ):
        self.output_dir = output_dir
        ensure_directory(output_dir)
        self.details_dir = os.path.join(output_dir, "details")
        ensure_directory(self.details_dir)
        self.logs_dir = os.path.join(output_dir, "logs") # For DPRFAgent logs
        ensure_directory(self.logs_dir)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.seed = seed
        self.example_select = example_select
        self.initial_persona_file = initial_persona_file
        self.task_model_name = task_model_name
        self.task_model_type = task_model_type
        self.max_iterations = max_iterations
        
        self.bedrock_region_name = bedrock_region_name
        self.model_kwargs = model_kwargs if model_kwargs else {}

        # Load template strings from files if provided, otherwise use provided strings
        final_instruction_template_str = ''
        if instruction_prompt_file and os.path.exists(instruction_prompt_file):
            try:
                with open(instruction_prompt_file, 'r') as f:
                    final_instruction_template_str = f.read()
                print(f"Using custom instruction prompt from: {instruction_prompt_file}")
            except Exception as e:
                print(f"Warning: Failed to load instruction prompt template: {e}")

        final_analysis_template_str = ''
        if analysis_prompt_file and os.path.exists(analysis_prompt_file):
            try:
                with open(analysis_prompt_file, 'r') as f:
                    final_analysis_template_str = f.read()
                print(f"Using custom analysis prompt from: {analysis_prompt_file}")
            except Exception as e:
                print(f"Warning: Failed to load analysis prompt template: {e}")
        else:
            print(f"DEBUG: analysis_prompt_file = {analysis_prompt_file}")
            print(f"DEBUG: File exists? {os.path.exists(analysis_prompt_file) if analysis_prompt_file else 'N/A'}")

        final_refinement_template_str = ''
        if refinement_prompt_file and os.path.exists(refinement_prompt_file):
            try:
                with open(refinement_prompt_file, 'r') as f:
                    final_refinement_template_str = f.read()
                print(f"Using custom refinement prompt from: {refinement_prompt_file}")
            except Exception as e:
                print(f"Warning: Failed to load refinement prompt template: {e}")

        self.few_shot_examples_text = ""
        self.few_shot_task = None
        if few_shot_examples_file:
            from few_shot import format_few_shot_block

            few_shot_path = few_shot_examples_file
            if not os.path.isabs(few_shot_path):
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                candidate = os.path.join(project_root, few_shot_path)
                if os.path.exists(candidate):
                    few_shot_path = candidate
            if not os.path.exists(few_shot_path):
                raise FileNotFoundError(f"few_shot_examples_file not found: {few_shot_examples_file}")
            with open(few_shot_path, "r", encoding="utf-8") as f:
                few_shot_payload = json.load(f)
            if isinstance(few_shot_payload, dict):
                self.few_shot_task = few_shot_payload.get("task")
                records = few_shot_payload.get("examples", [])
            else:
                records = few_shot_payload
            task_name = self.few_shot_task or "generic"
            self.few_shot_examples_text = format_few_shot_block(records, task_name)
            print(
                f"Loaded {len(records)} few-shot examples from {few_shot_path} "
                f"(task={task_name})"
            )

        self.instruction_template_str = final_instruction_template_str
        self.active_instruction_formatter = None
        if self.instruction_template_str:
            few_shot_text = self.few_shot_examples_text

            def custom_formatter(content, persona=None):
                return self.instruction_template_str.format(
                    content=content,
                    persona=persona if persona else "A helpful assistant.",
                    few_shot_examples=few_shot_text,
                )

            self.active_instruction_formatter = custom_formatter
        else:
            # Default basic formatter if no template string is given
            def default_formatter(content, persona=None):
                return f"{persona if persona else ''}\\n\\n{content}".strip()
            self.active_instruction_formatter = default_formatter
            print("Using default basic formatter for prompts.")

        self.analysis_template_str = final_analysis_template_str
        self.active_analysis_formatter = None
        if self.analysis_template_str:
            def custom_analysis_formatter(persona, content, generated_response, ground_truth):
                return self.analysis_template_str.format(
                    persona=persona,
                    content=content,
                    generated_response=generated_response,
                    ground_truth=ground_truth
                )
            self.active_analysis_formatter = custom_analysis_formatter
            print(f"DEBUG: active_analysis_formatter created successfully: {self.active_analysis_formatter is not None}")
        else:
            self.active_analysis_formatter = None
            print("Using default analysis formatter.")
            print(f"DEBUG: No analysis template loaded, active_analysis_formatter = None")

        self.refinement_template_str = final_refinement_template_str
        self.active_refinement_formatter = None
        if self.refinement_template_str:
            def custom_refinement_formatter(persona, analysis):
                return self.refinement_template_str.format(
                    persona=persona,
                    analysis=analysis
                )
            self.active_refinement_formatter = custom_refinement_formatter
        else:
            self.active_refinement_formatter = None
            print("Using default refinement formatter.")

        final_direct_refinement_template_str = ''
        if direct_refinement_prompt_file and os.path.exists(direct_refinement_prompt_file):
            try:
                with open(direct_refinement_prompt_file, 'r') as f:
                    final_direct_refinement_template_str = f.read()
                print(f"Using direct refinement prompt from: {direct_refinement_prompt_file} (analysis step will be skipped)")
            except Exception as e:
                print(f"Warning: Failed to load direct refinement prompt template: {e}")

        self.active_direct_refinement_formatter = None
        if final_direct_refinement_template_str:
            def custom_direct_refinement_formatter(persona, content, generated_response, ground_truth):
                return final_direct_refinement_template_str.format(
                    persona=persona,
                    content=content,
                    generated_response=generated_response,
                    ground_truth=ground_truth
                )
            self.active_direct_refinement_formatter = custom_direct_refinement_formatter
            print("Direct refinement formatter created. Analysis step will be skipped.")

        # Store DPRFAgent parameters for later initialization
        self.agent_params = {
            'task_model_name': task_model_name,
            'task_model_type': task_model_type,
            'refiner_model_name': refiner_model_name,
            'refiner_model_type': refiner_model_type,
            'openai_api_key': openai_api_key,
            'max_iterations': max_iterations,
            'save_logs': True,
            'log_dir': self.logs_dir,
            'bedrock_region_name': bedrock_region_name,
            'model_kwargs': self.model_kwargs
        }
        
        # Agent will be initialized in context manager
        self.agent = None

        self.bert_score_device = bert_score_device if bert_score_device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.bart_score_batch_size = bart_score_batch_size
        self._initialize_scorers()

        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        self.wandb_notes = wandb_notes


    def _initialize_scorers(self):
        """Initializes all scoring models."""
        # BERTScorer
        try:
            print(f"Initializing BERTScorer on {self.bert_score_device}...")
            self.bert_scorer = BERTScorer(model_type="roberta-large", device=self.bert_score_device, lang="en")
            print("BERTScorer initialized successfully.")
        except Exception as e:
            print(f"Critical Error: Failed to initialize BERTScorer: {e}")
            raise RuntimeError(f"Failed to initialize BERTScorer: {e}") from e

        # SentenceTransformer
        try:
            print("Initializing SentenceTransformer ('all-MiniLM-L6-v2')...")
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("SentenceTransformer initialized successfully.")
        except Exception as e:
            print(f"Warning: Failed to initialize SentenceTransformer: {e}")
            self.sentence_model = None

        # ROUGE Scorer
        print("Initializing ROUGE Scorer...")
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        print("ROUGE Scorer initialized successfully.")

        # BARTScorer
        self.bart_scorer_instance = None
        if BARTScorer is not None:
            try:
                if torch.cuda.is_available(): device = 'cuda:0'
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = 'mps'
                else: device = 'cpu'
                
                print(f"Initializing BARTScorer on {device}...")
                self.bart_scorer_instance = BARTScorer(device=device, checkpoint='facebook/bart-large-cnn')
                
                # Optional: Load fine-tuned weights if available
                # bart_weights_path = os.path.join(metrics_dir, "bart_score.pth") # Or a configurable path
                # if os.path.exists(bart_weights_path):
                #     print(f"Loading fine-tuned BARTScore weights from {bart_weights_path}")
                #     self.bart_scorer_instance.load(path=bart_weights_path)
                # else:
                #     print("Using default CNN checkpoint for BARTScore.")
                print("BARTScorer initialized successfully.")
            except Exception as e:
                print(f"Critical Error: Failed to initialize BARTScorer: {e}")
                # Fallback or raise error
                # raise RuntimeError(f"Failed to initialize BARTScorer: {e}") from e
                self.bart_scorer_instance = None # Ensure it's None if init fails
        else:
            print("BARTScorer module not loaded, skipping BARTScore calculation.")


    def evaluate_similarity(self, generated_text, ground_truth):
        metrics = {}
        generated_text = str(generated_text) if generated_text is not None else ""
        ground_truth = str(ground_truth) if ground_truth is not None else ""

        # ROUGE
        rouge_scores = self.rouge_scorer.score(ground_truth, generated_text)
        for rouge_type in ['rouge1', 'rouge2', 'rougeL']:
            metrics[f'{rouge_type}_precision'] = float(rouge_scores[rouge_type].precision)
            metrics[f'{rouge_type}_recall'] = float(rouge_scores[rouge_type].recall)
            metrics[f'{rouge_type}_f1'] = float(rouge_scores[rouge_type].fmeasure)

        # BLEU
        try:
            reference_tokens = nltk.word_tokenize(ground_truth.lower())
            generated_tokens = nltk.word_tokenize(generated_text.lower())
            if not generated_tokens or not reference_tokens:
                metrics['bleu'] = 0.0
            else:
                smoother = SmoothingFunction().method1
                metrics['bleu'] = float(sentence_bleu([reference_tokens], generated_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoother))
        except Exception as e:
            print(f"BLEU calculation error: {e}")
            metrics['bleu'] = None

        # BERTScore
        try:
            P, R, F1 = self.bert_scorer.score([generated_text], [ground_truth])
            metrics['bertscore_precision'] = float(P[0])
            metrics['bertscore_recall'] = float(R[0])
            metrics['bertscore_f1'] = float(F1[0])
        except Exception as e:
            print(f"Warning: Error calculating BERTScore: {e}")
            metrics.update({'bertscore_precision': 0.0, 'bertscore_recall': 0.0, 'bertscore_f1': 0.0})
        
        # BARTScore
        if self.bart_scorer_instance:
            try:
                src_to_tgt = self.bart_scorer_instance.score([ground_truth], [generated_text], batch_size=self.bart_score_batch_size)[0]
                tgt_to_src = self.bart_scorer_instance.score([generated_text], [ground_truth], batch_size=self.bart_score_batch_size)[0]
                metrics.update({
                    'bartscore_src_to_tgt': float(src_to_tgt),
                    'bartscore_tgt_to_src': float(tgt_to_src),
                    'bartscore_avg': float((src_to_tgt + tgt_to_src) / 2)
                })
            except Exception as e:
                print(f"Warning: Error calculating BARTScore: {e}")
                metrics.update({'bartscore_src_to_tgt': 0.0, 'bartscore_tgt_to_src': 0.0, 'bartscore_avg': 0.0})

        # Embedding Similarity
        if self.sentence_model:
            try:
                embedding1 = self.sentence_model.encode(generated_text, convert_to_tensor=True)
                embedding2 = self.sentence_model.encode(ground_truth, convert_to_tensor=True)
                # Normalize embeddings
                embedding1 = embedding1 / torch.linalg.norm(embedding1)
                embedding2 = embedding2 / torch.linalg.norm(embedding2)
                similarity = torch.dot(embedding1, embedding2).item()
                metrics['embedding_similarity'] = float(similarity)
            except Exception as e:
                print(f"Warning: Error calculating embedding similarity: {e}")
                metrics['embedding_similarity'] = 0.0
        return metrics

    async def run_iterations_with_per_iter_eval(
        self, initial_persona, content, optimization_ground_truth, evaluation_ground_truth=None
    ):
        """
        Execute DPRF iterations and evaluate after each iteration.
        Agent handles all logic including initial response generation.
        """
        per_iteration_eval_results = []

        if evaluation_ground_truth is None:
            evaluation_ground_truth = optimization_ground_truth

        # Let agent handle everything - initial + refinement iterations
        if self.max_iterations > 0:
            refined_results_package = await self.agent.run_iterations(
                initial_persona=initial_persona,
                content=content,
                ground_truth=optimization_ground_truth,
                persona_formatter=self.active_instruction_formatter,
                analysis_formatter=self.active_analysis_formatter,
                refinement_formatter=self.active_refinement_formatter,
                direct_refinement_formatter=self.active_direct_refinement_formatter,
            )
            all_iteration_details = refined_results_package.get("iterations", [])
            final_persona_after_refinement = refined_results_package.get("final_persona", initial_persona)

            # Extract evaluation results from agent's iteration details
            # Iteration 1 is treated as "initial" (using initial_persona but first response)
            for i, iter_detail in enumerate(all_iteration_details):
                iteration_num = iter_detail.get("iteration", i + 1)
                current_persona = iter_detail.get("persona", initial_persona)  # Persona used for generation
                generated_response_in_iter = iter_detail.get("generated_response", "")
                
                # Evaluate the response generated within this iteration
                current_metrics = self.evaluate_similarity(generated_response_in_iter, evaluation_ground_truth)

                per_iteration_eval_results.append({
                    "iteration": iteration_num,
                    "persona": current_persona,
                    "response": generated_response_in_iter,
                    "metrics": current_metrics
                })
        else:
            # If no iterations, just generate with initial persona
            initial_response = await self.agent.generate_response(
                persona=initial_persona,
                content=content,
                custom_formatter=self.active_instruction_formatter
            )
            initial_metrics = self.evaluate_similarity(initial_response, evaluation_ground_truth)
            
            per_iteration_eval_results.append({
                "iteration": 1,  # Start from 1, not 0
                "persona": initial_persona,
                "response": initial_response,
                "metrics": initial_metrics
            })
            
            final_persona_after_refinement = initial_persona
            all_iteration_details = []
        
        return {
            "initial_persona": initial_persona,
            "final_persona": final_persona_after_refinement,
            "iterations_details_from_agent": all_iteration_details,
            "per_iteration_eval_results": per_iteration_eval_results
        }

    async def evaluate_example(self, task_data):
        """
        Evaluates a single example.
        """
        content = task_data["content"]
        initial_persona = task_data["initial_persona"]
        ground_truth = task_data["ground_truth"]
        optimization_ground_truth = task_data.get("optimization_ground_truth", ground_truth)
        evaluation_ground_truth = task_data.get("evaluation_ground_truth", ground_truth)
        bio_text = task_data.get("bio_text")
        
        if initial_persona:
            initial_persona = initial_persona
        elif self.initial_persona_file:
            with open(self.initial_persona_file, "r") as f:
                initial_persona = f.read()
                print(f"Loaded initial persona from {self.initial_persona_file}")
                print(initial_persona)

        iteration_package = await self.run_iterations_with_per_iter_eval(
            initial_persona=initial_persona,
            content=content,
            optimization_ground_truth=optimization_ground_truth,
            evaluation_ground_truth=evaluation_ground_truth
        )

        results_summary = {}

        # Get first iteration as "initial" (iteration 1 = initial persona response)
        if iteration_package["per_iteration_eval_results"]:
            initial_result = iteration_package["per_iteration_eval_results"][0]
            results_summary["initial"] = {
                "persona": initial_result["persona"],  # Should be initial_persona
                "response": initial_result["response"],
                "metrics": initial_result["metrics"]
            }
        
        # Get last iteration as "refined" 
        if len(iteration_package["per_iteration_eval_results"]) > 1:
            refined_result = iteration_package["per_iteration_eval_results"][-1]
            results_summary["refined"] = {
                "persona": refined_result["persona"],
                "response": refined_result["response"],
                "metrics": refined_result["metrics"],
                "refinement_iterations": len(iteration_package["iterations_details_from_agent"]),
                "refinement_analyses": [
                    iter_detail.get("analysis", "") for iter_detail in iteration_package["iterations_details_from_agent"]
                ]
            }
        else:
            # No refinement iterations, refined = initial
            initial_result = iteration_package["per_iteration_eval_results"][0]
            results_summary["refined"] = {
                "persona": initial_result["persona"],
                "response": initial_result["response"], 
                "metrics": initial_result["metrics"],
                "refinement_iterations": 0,
                "refinement_analyses": []
            }

        # Bio-informed persona evaluation (only this needs separate evaluation)
        if bio_text:
            bio_persona = bio_text
            bio_response = await self.agent.generate_response(
                persona=bio_persona,
                content=content,
                custom_formatter=self.active_instruction_formatter
            )
            bio_metrics = self.evaluate_similarity(bio_response, evaluation_ground_truth)
            
            results_summary["bio"] = {
                "persona": bio_persona,
                "response": bio_response,
                "metrics": bio_metrics
            }

        results_summary["task_info"] = task_data["task_specific_info"].copy()
        # Store content and ground_truth separately (not in task_info to keep CSV clean)
        results_summary["content"] = content
        results_summary["ground_truth"] = evaluation_ground_truth
        results_summary["optimization_ground_truth"] = optimization_ground_truth
        results_summary["per_iteration_evaluations"] = iteration_package["per_iteration_eval_results"]
        results_summary["dprf_iteration_details"] = iteration_package["iterations_details_from_agent"]
        
        return results_summary

    async def evaluate(self):
        """
        Main evaluation loop. Subclasses must implement load_examples and create_task.
        """
        if not hasattr(self, 'load_examples') or not hasattr(self, 'create_task'):
            raise NotImplementedError("Subclasses must implement load_examples and create_task methods.")

        # Use context manager to ensure DPRFAgent resources are cleaned up
        with DPRFAgent(**self.agent_params) as agent:
            self.agent = agent
            if hasattr(self.agent, "token_usage"):
                self.agent.token_usage.set_request_log(
                    os.path.join(self.output_dir, "token_usage_calls.jsonl")
                )

            examples_to_process = await self.load_examples()
            
            processed_results_list = []
            all_per_iteration_data_frames = []

            # Check if this is interview dataset based on evaluator class name
            is_interview_dataset = "Interview" in self.__class__.__name__
            
            if is_interview_dataset:
                print("Detected interview dataset - using speaker-grouped processing with agent interview mode")
                await self._evaluate_interview_with_agent(examples_to_process, processed_results_list, all_per_iteration_data_frames)
            else:
                print("Using standard evaluation processing")
                await self._evaluate_standard(examples_to_process, processed_results_list, all_per_iteration_data_frames)

            # Create DataFrames and save CSVs (common for both approaches)
            summary_results_df = pd.DataFrame(processed_results_list)
            summary_results_df.to_csv(os.path.join(self.output_dir, "evaluation_results.csv"), index=False)
            
            # Save iteration average metrics
            if all_per_iteration_data_frames:
                all_iterations_df = pd.concat(all_per_iteration_data_frames, ignore_index=True)
                
                try:
                    # Iteration average statistics (numeric_only=True for mean)
                    iteration_stats = all_iterations_df.groupby('iteration').mean(numeric_only=True).reset_index()
                    iteration_stats.to_csv(os.path.join(self.output_dir, "iteration_average_metrics.csv"), index=False)
                    self._log_iteration_improvements(iteration_stats)
                    
                    # Print iteration statistics like debate evaluation
                    print("\nIteration Evaluation Statistics:")
                    for _, row in iteration_stats.iterrows():
                        iter_num = int(row['iteration'])
                        print(f"\nIteration {iter_num}:")
                        for col in row.index:
                            if col not in ['iteration', 'example_id', 'speaker_id', 'debate_id', 'speaker_type'] and pd.api.types.is_numeric_dtype(row[col]) and not pd.isna(row[col]):
                                print(f"  {col}: {row[col]:.4f}")
                                
                except Exception as e:
                    print(f"Warning: Error calculating or logging iteration statistics: {e}")
                    traceback.print_exc()
            else:
                iteration_stats = pd.DataFrame() # Empty DF if no iteration data

            self.calculate_and_log_aggregate_metrics(summary_results_df)
            self._save_token_usage_report()
                
            return summary_results_df, iteration_stats

    def _save_token_usage_report(self):
        tracker = getattr(self.agent, "token_usage", None)
        if tracker is None:
            return

        summary_path = os.path.join(self.output_dir, "token_usage.json")
        tracker.save(summary_path)
        usage = tracker.to_dict()

        aggregate_path = os.path.join(self.output_dir, "aggregate_metrics.json")
        if os.path.exists(aggregate_path):
            try:
                with open(aggregate_path, "r", encoding="utf-8") as f:
                    aggregate = json.load(f)
                aggregate["token_usage"] = usage
                with open(aggregate_path, "w", encoding="utf-8") as f:
                    json.dump(aggregate, f, indent=2)
            except Exception as e:
                print(f"Warning: Could not merge token usage into aggregate_metrics.json: {e}")

        print("\n=== Token usage (for billing) ===")
        print(f"  API requests:      {usage['total_requests']:,}")
        print(f"  Input tokens:      {usage['total_input_tokens']:,}")
        print(f"  Output tokens:     {usage['total_output_tokens']:,}")
        print(f"  Total tokens:      {usage['total_tokens']:,}")
        if usage.get("by_source"):
            print("  By source:")
            for source, counts in sorted(usage["by_source"].items()):
                print(
                    f"    {source}: in={counts['input_tokens']:,} "
                    f"out={counts['output_tokens']:,} "
                    f"req={counts['requests']:,}"
                )
        print(f"  Saved: {summary_path}")
        calls_log = os.path.join(self.output_dir, "token_usage_calls.jsonl")
        if os.path.exists(calls_log):
            print(f"  Per-call log: {calls_log}")
    
    def _log_example_failure(self, i: int, example_obj, error: Exception) -> None:
        task_info = example_obj if isinstance(example_obj, dict) else {}
        speaker_id = task_info.get("speaker_id", "unknown")
        print(
            f"Skipped example idx={i}, speaker_id={speaker_id}: "
            f"{type(error).__name__}: {error}"
        )
        traceback.print_exception(type(error), error, error.__traceback__)

    async def _evaluate_standard(self, examples_to_process, processed_results_list, all_per_iteration_data_frames):
        """Standard evaluation approach - each example processed independently"""
        async def evaluate_one(i: int, example_obj):
            try:
                self._current_example_index = i
                task_data_for_eval = await self.create_task(example_obj)
                result = await self.evaluate_example(task_data_for_eval)
                return i, result
            except Exception as e:
                return i, e

        print(f"Evaluating {len(examples_to_process)} examples concurrently...")
        tasks = [
            asyncio.create_task(evaluate_one(i, example_obj))
            for i, example_obj in enumerate(examples_to_process)
        ]

        failed_count = 0
        saved_count = 0
        for finished in asyncio.as_completed(tasks):
            i, single_example_result_data = await finished
            if isinstance(single_example_result_data, Exception):
                failed_count += 1
                self._log_example_failure(i, examples_to_process[i], single_example_result_data)
                continue

            saved_count += 1
            example_id_for_file = single_example_result_data.get("task_info", {}).get(
                getattr(self, "example_id_key_for_filename", "example_idx"), f"example_idx_{i}"
            )
            await self._process_single_evaluation_result(
                i,
                single_example_result_data,
                processed_results_list,
                all_per_iteration_data_frames,
            )
            print(
                f"Saved detail ({saved_count}/{len(examples_to_process)}): "
                f"example_{example_id_for_file}.json"
            )

        if failed_count:
            print(
                f"Warning: {failed_count}/{len(examples_to_process)} examples failed and were skipped. "
                "Check logs for speaker_id (often Azure content_filter on debate text)."
            )

    async def _evaluate_interview_with_agent(self, examples_to_process, processed_results_list, all_per_iteration_data_frames):
        """Simplified interview evaluation using agent's interview mode"""
        
        print("Grouping examples by speaker...")
        
        # Group examples by speaker
        speaker_groups = {}
        task_data_list = []
        
        for i, example_obj in enumerate(examples_to_process):
            self._current_example_index = i
            task_data = await self.create_task(example_obj)
            task_data_list.append((task_data, i))
            
            speaker_name = task_data.get("task_specific_info", {}).get("speakername", "unknown")
            
            if speaker_name not in speaker_groups:
                speaker_groups[speaker_name] = []
            
            speaker_groups[speaker_name].append((task_data, i))
        
        print(f"Found {len(speaker_groups)} unique speakers")
        
        # Process each speaker group using agent's interview mode
        speaker_processing_tasks = []
        for speaker_name, speaker_examples in speaker_groups.items():
            speaker_processing_tasks.append(
                self._process_speaker_with_agent(speaker_name, speaker_examples)
            )
        
        # Wait for all speaker group processing to complete with progress bar
        speaker_results = await tqdm.gather(*speaker_processing_tasks, desc="Processing speaker groups")

        print("Collecting and formatting final results...")
        
        # Collect results and format for output
        for speaker_result in speaker_results:
            if isinstance(speaker_result, Exception):
                print(f"Error in speaker processing: {speaker_result}")
                continue
            
            for example_result in speaker_result["example_results"]:
                await self._process_single_interview_result(example_result, processed_results_list, all_per_iteration_data_frames)

    async def _process_speaker_with_agent(self, speaker_name, speaker_examples):
        """Process all examples for a single speaker using agent's interview mode"""
        print(f"Processing speaker: {speaker_name} ({len(speaker_examples)} examples)")
        
        if not speaker_examples:
            return {"speaker_name": speaker_name, "example_results": []}
        
        first_example = speaker_examples[0]
        initial_persona = first_example[0]["initial_persona"]

        has_val_split = any(ex[0].get("split") == "val" for ex in speaker_examples)
        if has_val_split:
            optimization_examples = [
                ex for ex in speaker_examples if ex[0].get("split", "train") == "train"
            ]
            evaluation_examples = [
                ex for ex in speaker_examples if ex[0].get("split") == "val"
            ]
            print(
                f"  Train/val split: {len(optimization_examples)} train, "
                f"{len(evaluation_examples)} val"
            )
        else:
            optimization_examples = speaker_examples
            evaluation_examples = speaker_examples

        opt_content_list = [task_data["content"] for task_data, _ in optimization_examples]
        opt_ground_truth_list = [
            task_data.get("optimization_ground_truth", task_data["ground_truth"])
            for task_data, _ in optimization_examples
        ]

        eval_content_list = [task_data["content"] for task_data, _ in evaluation_examples]
        eval_ground_truth_list = [
            task_data.get("evaluation_ground_truth", task_data["ground_truth"])
            for task_data, _ in evaluation_examples
        ]

        async def generate_responses_for_content(persona, contents):
            if not contents:
                return []
            response_tasks = [
                self.agent.generate_response(
                    persona=persona,
                    content=content,
                    custom_formatter=self.active_instruction_formatter,
                )
                for content in contents
            ]
            return await asyncio.gather(*response_tasks)

        if self.max_iterations > 0:
            refined_results_package = await self.agent.run_iterations(
                initial_persona=initial_persona,
                content=opt_content_list,
                ground_truth=opt_ground_truth_list,
                persona_formatter=self.active_instruction_formatter,
                analysis_formatter=self.active_analysis_formatter,
                refinement_formatter=self.active_refinement_formatter,
                interview_mode=True,
                direct_refinement_formatter=self.active_direct_refinement_formatter,
            )
            
            final_persona = refined_results_package.get("final_persona", initial_persona)
            iteration_details = refined_results_package.get("iterations", [])
            opt_final_responses = refined_results_package.get("final_responses", [])
            
            if len(opt_final_responses) != len(opt_content_list):
                print(
                    f"WARNING: Response count mismatch! Expected {len(opt_content_list)}, "
                    f"got {len(opt_final_responses)}"
                )
                opt_final_responses = await generate_responses_for_content(
                    final_persona, opt_content_list
                )
        else:
            opt_final_responses = await generate_responses_for_content(
                initial_persona, opt_content_list
            )
            final_persona = initial_persona
            iteration_details = []

        if has_val_split:
            initial_responses = await generate_responses_for_content(
                initial_persona, eval_content_list
            )
            final_responses = await generate_responses_for_content(
                final_persona, eval_content_list
            )
        else:
            if iteration_details and iteration_details[0].get("individual_responses"):
                initial_responses = iteration_details[0]["individual_responses"]
            else:
                initial_responses = opt_final_responses
            final_responses = opt_final_responses
        
        bio_responses = []
        bio_persona = None
        if speaker_examples and speaker_examples[0][0].get("bio_text"):
            bio_persona = speaker_examples[0][0]["bio_text"]
            print(f"Generating bio responses for speaker {speaker_name}")
            bio_responses = await generate_responses_for_content(
                bio_persona, eval_content_list
            )
        
        initial_metrics_list = []
        refined_metrics_list = []
        bio_metrics_list = []
        
        for i, (gt, initial_resp, final_resp) in enumerate(
            zip(eval_ground_truth_list, initial_responses, final_responses)
        ):
            initial_metrics = self.evaluate_similarity(initial_resp, gt)
            refined_metrics = self.evaluate_similarity(final_resp, gt)
            initial_metrics_list.append(initial_metrics)
            refined_metrics_list.append(refined_metrics)
            
            if bio_responses and i < len(bio_responses):
                bio_metrics = self.evaluate_similarity(bio_responses[i], gt)
                bio_metrics_list.append(bio_metrics)
        
        def average_metrics(metrics_list):
            if not metrics_list:
                return {}
            avg_metrics = {}
            for key in metrics_list[0].keys():
                values = [m[key] for m in metrics_list if m[key] is not None]
                avg_metrics[key] = sum(values) / len(values) if values else 0.0
            return avg_metrics
        
        avg_initial_metrics = average_metrics(initial_metrics_list)
        avg_refined_metrics = average_metrics(refined_metrics_list)
        avg_bio_metrics = average_metrics(bio_metrics_list) if bio_metrics_list else {}
        
        aggregated_result = {
            "speaker_name": speaker_name,
            "num_examples": len(evaluation_examples),
            "num_train_examples": len(optimization_examples),
            "num_val_examples": len(evaluation_examples),
            "has_train_val_split": has_val_split,
            "initial": {
                "persona": initial_persona,
                "responses": initial_responses,
                "metrics": avg_initial_metrics
            },
            "refined": {
                "persona": final_persona,
                "responses": final_responses,
                "metrics": avg_refined_metrics,
                "refinement_iterations": len(iteration_details),
                "refinement_analyses": [iter_detail.get("analysis", "") for iter_detail in iteration_details]
            },
            "content_list": eval_content_list,
            "ground_truth_list": eval_ground_truth_list,
            "optimization_content_list": opt_content_list,
            "optimization_ground_truth_list": opt_ground_truth_list,
            "per_iteration_evaluations": [],
            "dprf_iteration_details": iteration_details
        }
        
        if bio_responses:
            aggregated_result["bio"] = {
                "persona": bio_persona,
                "responses": bio_responses,
                "metrics": avg_bio_metrics
            }
        
        if iteration_details:
            for iter_detail in iteration_details:
                iteration_num = iter_detail.get("iteration", 1)
                persona_used = iter_detail.get("persona", initial_persona)
                if has_val_split:
                    individual_responses = await generate_responses_for_content(
                        persona_used, eval_content_list
                    )
                else:
                    individual_responses = iter_detail.get("individual_responses", [])
                
                for i, (gt, resp) in enumerate(zip(eval_ground_truth_list, individual_responses)):
                    if i < len(individual_responses):
                        iter_metrics = self.evaluate_similarity(resp, gt)
                        aggregated_result["per_iteration_evaluations"].append({
                            "example_index": i,
                            "iteration": iteration_num,
                            "persona": persona_used,
                            "response": resp,
                            "metrics": iter_metrics
                        })
        else:
            for i, (gt, resp) in enumerate(zip(eval_ground_truth_list, final_responses)):
                final_metrics = self.evaluate_similarity(resp, gt)
                aggregated_result["per_iteration_evaluations"].append({
                    "example_index": i,
                    "iteration": 1,
                    "persona": initial_persona,
                    "response": resp,
                    "metrics": final_metrics
                })
        
        return {
            "speaker_name": speaker_name,
            "example_results": [aggregated_result]
        }

    async def _process_single_interview_result(self, example_result, processed_results_list, all_per_iteration_data_frames):
        """Process a single interview example result for output formatting"""
        # Now handling aggregated results for a speaker
        speaker_name = example_result.get("speaker_name", "unknown")
        num_examples = example_result.get("num_examples", 0)
        
        # Use speaker name as the file ID
        example_id_for_file = speaker_name.replace(" ", "_").replace(".", "_")
        
        try:
            # Create result summary matching standard format but with lists
            results_summary = {
                "speaker_name": speaker_name,
                "num_examples": num_examples,
                "initial": example_result["initial"],
                "refined": example_result["refined"],
                "content_list": example_result["content_list"],
                "ground_truth_list": example_result["ground_truth_list"],
                "per_iteration_evaluations": example_result["per_iteration_evaluations"],
                "dprf_iteration_details": example_result["dprf_iteration_details"]
            }
            
            if "bio" in example_result:
                results_summary["bio"] = example_result["bio"]
            
            # Save detailed JSON for this speaker (with all examples aggregated)
            detail_file_path = os.path.join(self.details_dir, f"speaker_{example_id_for_file}.json")
            with open(detail_file_path, "w") as f:
                json.dump(results_summary, f, indent=2)

            # Prepare data for summary CSV - one row per speaker (averaged metrics)
            summary_row = {
                "speaker_name": speaker_name,
                "num_examples": num_examples,
            }

            # Add averaged metrics for each persona type
            for persona_key in ["initial", "bio", "refined"]:
                if persona_key in results_summary:
                    metrics = results_summary[persona_key]["metrics"]
                    for metric_n, metric_v in metrics.items():
                        summary_row[f"{persona_key}_{metric_n}"] = metric_v

            processed_results_list.append(summary_row)

            # Per-iteration data for averaging - create rows for each example in each iteration
            if "per_iteration_evaluations" in results_summary:
                iter_summary_for_df = []
                for iter_eval_res in results_summary["per_iteration_evaluations"]:
                    iter_row = {
                        "speaker_name": speaker_name,
                        "example_index": iter_eval_res["example_index"],
                        "iteration": iter_eval_res["iteration"],
                    }
                    
                    # Add all metrics
                    iter_row.update(iter_eval_res["metrics"])
                    iter_summary_for_df.append(iter_row)
                
                if iter_summary_for_df:
                    all_per_iteration_data_frames.append(pd.DataFrame(iter_summary_for_df))

        except Exception as e:
            print(f"Error processing aggregated interview result for speaker {speaker_name}: {e}")
            traceback.print_exc()

    async def _process_single_evaluation_result(
        self,
        i: int,
        single_example_result_data,
        processed_results_list,
        all_per_iteration_data_frames,
    ):
        """Save one example's detail JSON and append summary rows (called as each example finishes)."""
        example_id_for_file = single_example_result_data.get("task_info", {}).get(
            getattr(self, "example_id_key_for_filename", "example_idx"), f"example_idx_{i}"
        )
        try:
            detail_file_path = os.path.join(self.details_dir, f"example_{example_id_for_file}.json")
            with open(detail_file_path, "w") as f:
                json.dump(single_example_result_data, f, indent=2)

            summary_row = {"example_id": i + 1}
            task_info = single_example_result_data.get("task_info", {})
            for key, value in task_info.items():
                if key not in ["content", "ground_truth"]:
                    summary_row[key] = value

            for persona_key in ["initial", "bio", "refined"]:
                if persona_key in single_example_result_data:
                    for metric_n, metric_v in single_example_result_data[persona_key]["metrics"].items():
                        summary_row[f"{persona_key}_{metric_n}"] = metric_v

            processed_results_list.append(summary_row)

            if "per_iteration_evaluations" in single_example_result_data:
                iter_summary_for_df = []
                for iter_eval_res in single_example_result_data["per_iteration_evaluations"]:
                    iter_row = {
                        "example_id": i + 1,
                        "iteration": iter_eval_res["iteration"],
                    }
                    for key in ["speaker_id", "debate_id", "speaker_type"]:
                        if key in task_info:
                            iter_row[key] = task_info[key]
                    iter_row.update(iter_eval_res["metrics"])
                    iter_summary_for_df.append(iter_row)

                if iter_summary_for_df:
                    all_per_iteration_data_frames.append(pd.DataFrame(iter_summary_for_df))
        except Exception as e:
            print(f"Error processing result for example (ID: {example_id_for_file}): {e}")
            traceback.print_exc()

    async def _process_evaluation_results(self, raw_results_from_gather, examples_to_process, processed_results_list, all_per_iteration_data_frames):
        """Process evaluation results for standard (non-interview) datasets"""
        for i, single_example_result_data in enumerate(raw_results_from_gather):
            original_example_obj = examples_to_process[i]

            if isinstance(single_example_result_data, Exception):
                task_info = original_example_obj if isinstance(original_example_obj, dict) else {}
                speaker_id = task_info.get("speaker_id", "unknown")
                print(
                    f"Error evaluating example idx={i}, speaker_id={speaker_id}: "
                    f"{type(single_example_result_data).__name__}: {single_example_result_data}"
                )
                if hasattr(single_example_result_data, "__traceback__"):
                    traceback.print_exception(
                        type(single_example_result_data),
                        single_example_result_data,
                        single_example_result_data.__traceback__,
                    )
                continue

            await self._process_single_evaluation_result(
                i,
                single_example_result_data,
                processed_results_list,
                all_per_iteration_data_frames,
            )

    def _log_iteration_improvements(self, iteration_stats_df):
        print("\nImprovement Percentage Between Iterations:")
        # Define columns to exclude from improvement calculation
        cols_to_exclude = ['iteration', 'example_id', 'speaker_id', 'debate_id', 'speaker_type'] # Add any other non-metric ID columns from task_info
        if hasattr(self, 'task_specific_keys_in_iteration_df'):
             cols_to_exclude.extend(self.task_specific_keys_in_iteration_df())

        for col in iteration_stats_df.columns:
            if col not in cols_to_exclude and pd.api.types.is_numeric_dtype(iteration_stats_df[col]):
                sorted_stats = iteration_stats_df.sort_values('iteration')
                for i in range(1, len(sorted_stats)):
                    prev_iter_num = int(sorted_stats.iloc[i-1]['iteration'])
                    curr_iter_num = int(sorted_stats.iloc[i]['iteration'])
                    prev_val = sorted_stats.iloc[i-1][col]
                    curr_val = sorted_stats.iloc[i][col]
                    
                    if not pd.isna(prev_val) and not pd.isna(curr_val):
                        improvement_text = ""
                        if prev_val != 0:
                            improvement_pct = (curr_val - prev_val) / abs(prev_val) * 100
                            improvement_text = f"Iter {prev_iter_num} \u2192 Iter {curr_iter_num}, {col}: {improvement_pct:.2f}%"
                        elif curr_val != 0:
                            improvement_text = f"Iter {prev_iter_num} \u2192 Iter {curr_iter_num}, {col}: Change from 0 to {curr_val:.4f}"
                        if improvement_text:
                             print(f"  {improvement_text}")


    def calculate_and_log_aggregate_metrics(self, results_df):
        if results_df.empty:
            print("Warning: No evaluation results to aggregate.")
            return

        aggregate_metrics = {}
        persona_types_present = []
        if any(col.startswith("initial_") for col in results_df.columns): persona_types_present.append("initial")
        if any(col.startswith("refined_") for col in results_df.columns): persona_types_present.append("refined")
        if any(col.startswith("bio_") for col in results_df.columns): persona_types_present.append("bio")

        # Determine common metric bases (e.g., rouge1_f1, bleu)
        metric_bases = set()
        for p_type in persona_types_present:
            for col in results_df.columns:
                if col.startswith(f"{p_type}_"):
                    metric_bases.add(col[len(p_type)+1:])
        
        metric_bases = sorted(list(metric_bases))
        if not metric_bases:
            print("Warning: No common metric bases found to aggregate.")
            return

        print("\nAggregate Metrics:")

        # Mean and std for each persona type and metric
        print("\nMean Performance:")
        for p_type in persona_types_present:
            print(f"\n{p_type.capitalize()} Persona:")
            for base in metric_bases:
                col_name = f"{p_type}_{base}"
                if col_name in results_df.columns and results_df[col_name].notna().any():
                    mean_val = results_df[col_name].mean()
                    std_val = results_df[col_name].std()
                    aggregate_metrics[f"mean_{col_name}"] = mean_val
                    aggregate_metrics[f"std_{col_name}"] = std_val
                    print(f"  {base}: {mean_val:.4f} (±{std_val:.4f})")
                else:
                    print(f"  {base}: Not available")
        
        # Improvement percentages
        improvements = {}
        
        print("\nImprovement Percentages:")
        if "initial" in persona_types_present and "refined" in persona_types_present:
            print("Refined vs Initial Improvement (%):")
            for base in metric_bases:
                initial_col, refined_col = f"initial_{base}", f"refined_{base}"
                if initial_col in results_df.columns and refined_col in results_df.columns and \
                   results_df[initial_col].notna().any() and results_df[refined_col].notna().any():
                    # Handle division by zero or NaN initial values gracefully
                    improvement_series = (results_df[refined_col] - results_df[initial_col]) / results_df[initial_col].replace(0, float('nan')) * 100
                    mean_improvement = improvement_series.mean() # Mean of percentages
                    if not pd.isna(mean_improvement):
                        key = f"refined_vs_initial_{base}_improvement_%"
                        improvements[key] = mean_improvement
                        print(f"  {base}: {mean_improvement:.2f}%")
        
        if "bio" in persona_types_present:
            if "initial" in persona_types_present:
                print("Bio vs Initial Improvement (%):")
                for base in metric_bases:
                    initial_col, bio_col = f"initial_{base}", f"bio_{base}"
                    if initial_col in results_df.columns and bio_col in results_df.columns and \
                       results_df[initial_col].notna().any() and results_df[bio_col].notna().any():
                        improvement_series = (results_df[bio_col] - results_df[initial_col]) / results_df[initial_col].replace(0, float('nan')) * 100
                        mean_improvement = improvement_series.mean()
                        if not pd.isna(mean_improvement):
                            key = f"bio_vs_initial_{base}_improvement_%"
                            improvements[key] = mean_improvement
                            print(f"  {base}: {mean_improvement:.2f}%")
            
            if "refined" in persona_types_present:
                print("Refined vs Bio Improvement (%):")
                for base in metric_bases:
                    refined_col, bio_col = f"refined_{base}", f"bio_{base}"
                    if refined_col in results_df.columns and bio_col in results_df.columns and \
                       results_df[refined_col].notna().any() and results_df[bio_col].notna().any():
                        improvement_series = (results_df[refined_col] - results_df[bio_col]) / results_df[bio_col].replace(0, float('nan')) * 100
                        mean_improvement = improvement_series.mean()
                        if not pd.isna(mean_improvement):
                            key = f"refined_vs_bio_{base}_improvement_%"
                            improvements[key] = mean_improvement
                            print(f"  {base}: {mean_improvement:.2f}%")

        # Save aggregate metrics to JSON
        with open(os.path.join(self.output_dir, "aggregate_metrics.json"), "w") as f:
            json.dump({**aggregate_metrics, **improvements}, f, indent=2)

    @staticmethod
    def add_generic_args(parser):
        """Adds generic evaluation arguments to an argparse parser."""
        group = parser.add_argument_group('Generic Evaluation Parameters')
        group.add_argument("--output_dir", default="results", help="Directory to save evaluation results")
        group.add_argument("--task_model", default="gpt-4o", help="Model for task performance")
        group.add_argument("--task_model_type", default="openai", choices=["openai", "sglang", "vllm", "llama.cpp", "hf", "hf_8bit", "bedrock"], help="Type of task model")
        group.add_argument("--refiner_model", default="gpt-4o", help="Model for persona refinement")
        group.add_argument("--refiner_model_type", default="openai", choices=["openai", "vllm", "sglang", "llama.cpp", "hf", "hf_8bit", "bedrock"], help="Type of refiner model")
        group.add_argument("--openai_api_key", default=os.environ.get("OPENAI_API_KEY"), help="OpenAI API key (defaults to env var OPENAI_API_KEY)")
        group.add_argument("--iterations", type=int, default=3, help="Maximum number of refinement iterations for DPRF")
        group.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

        group.add_argument("--bedrock_region", default="us-east-1", help="AWS Bedrock region name (e.g., us-east-1)")
        group.add_argument("--model_kwargs_json", type=str, default=None, help="JSON string or path to a JSON file for model_kwargs (e.g., '{\"claude_temperature\": 0.6}')")

        group.add_argument("--wandb_project", type=str, default="dprf", help="WandB project name")
        group.add_argument("--wandb_run_name", type=str, help="WandB run name prefix")
        group.add_argument("--wandb_notes", type=str, default="", help="WandB run notes")
        group.add_argument("--data_dir", default="DPRF/Evaluation/debate/data/processed",                         help="Directory with processed debate data (relative to project root or absolute)")
        group.add_argument("--length", type=int, default=None, help="Number of examples to select (default: all). For interview, this is number of speaker JSON files.")
        group.add_argument(
            "--example_select",
            choices=["random", "first", "last"],
            default="random",
            help="How to pick examples when --length is set: random, first N, or last N (non-overlapping with few-shot pool if you use last-100 pool + first-100 eval).",
        )
        group.add_argument(
            "--few_shot_examples_file",
            default=None,
            help="JSON file with few-shot examples (injected into instruction prompt as {few_shot_examples}).",
        )
        group.add_argument("--initial_persona_file", default=None,
                                help="Path to a custom initial persona template file (optional, relative or absolute)")
        group.add_argument("--instruction_prompt_file", default=None, help="Path to a generic instruction prompt template file (optional). Use instruction_few_shot.txt with {few_shot_examples}.")
        
        group.add_argument("--analysis_prompt_file", help="Path to custom analysis prompt template")
        group.add_argument("--refinement_prompt_file", help="Path to custom refinement prompt template")
        group.add_argument("--direct_refinement_prompt_file", help="Path to direct refinement prompt (skips analysis step; use prompts/analysis_refinement.txt or prompts/direct_refinement.txt)")

        return parser

    # Methods to be implemented by subclasses
    async def load_examples(self):
        raise NotImplementedError("Subclasses must implement load_examples.")

    async def create_task(self, example):
        """
        Process a raw example from load_examples into a structured task.
        Should return a dictionary with keys:
        'content', 'initial_persona', 'ground_truth',
        'bio_text' (optional), 'task_specific_info' (dict for logging),
        'example_id_for_filename' (str for unique filename).
        """
        raise NotImplementedError("Subclasses must implement create_task.")
