#!/usr/bin/env python3
"""
ICL Baseline (debate + interview only, train/val split data).

Per unit (speaker / interviewee):
  1) Rewrite persona using P0 + TRAIN (x, y_train) — one API call
  2) Generate + score on TRAIN
  3) Generate + score on VAL (held-out)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Evaluation"))

from core.token_usage import TokenUsageTracker, record_bedrock_usage
from core.utils import build_bedrock_inference_config
from icl_baseline.data_loaders import (
    DATASET_REGISTRY,
    load_dataset_units,
    load_initial_persona,
)

import aioboto3
from botocore.config import Config
import nltk

nltk.download("punkt_tab", quiet=True)
import torch
from bert_score import BERTScorer
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from tqdm.asyncio import tqdm


class BedrockGenerator:
    def __init__(
        self,
        model_id: str,
        region: str,
        max_tokens: int = 2000,
        temperature: float = 0.6,
        top_p: float = 0.95,
        max_concurrency: int = 20,
        token_usage: Optional[TokenUsageTracker] = None,
    ):
        self.model_id = model_id
        self.region = region
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.token_usage = token_usage or TokenUsageTracker()
        self.config = Config(
            region_name=region,
            retries={"max_attempts": 100, "mode": "adaptive"},
        )
        self.session = aioboto3.Session()
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def generate(self, prompt: str, source: str) -> str:
        async with self.semaphore:
            async with self.session.client(
                "bedrock-runtime", config=self.config
            ) as client:
                request = {
                    "modelId": self.model_id,
                    "messages": [{"role": "user", "content": [{"text": prompt}]}],
                    "inferenceConfig": build_bedrock_inference_config(
                        self.max_tokens,
                        model_kwargs={
                            "claude_max_tokens": self.max_tokens,
                            "claude_temperature": self.temperature,
                        },
                        temperature=self.temperature,
                    ),
                }
                response = await client.converse(**request)
                record_bedrock_usage(self.token_usage, response, source=source)
                content = response["output"]["message"]["content"]
                if content and "text" in content[0]:
                    return content[0]["text"].strip()
                return ""


class MetricsCalculator:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rouge = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        self.bert_scorer = BERTScorer(
            model_type="roberta-large", device=device, lang="en"
        )
        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

    def score(self, generated: str, ground_truth: str) -> Dict[str, float]:
        generated = str(generated or "")
        ground_truth = str(ground_truth or "")

        rouge = self.rouge.score(ground_truth, generated)
        metrics = {
            "rougeL_f1": float(rouge["rougeL"].fmeasure),
            "rougeL_precision": float(rouge["rougeL"].precision),
            "rougeL_recall": float(rouge["rougeL"].recall),
        }

        P, R, F1 = self.bert_scorer.score([generated], [ground_truth])
        metrics["bertscore_f1"] = float(F1[0])
        metrics["bertscore_precision"] = float(P[0])
        metrics["bertscore_recall"] = float(R[0])

        emb1 = self.sentence_model.encode(generated, convert_to_tensor=True)
        emb2 = self.sentence_model.encode(ground_truth, convert_to_tensor=True)
        emb1 = emb1 / torch.linalg.norm(emb1)
        emb2 = emb2 / torch.linalg.norm(emb2)
        metrics["embedding_similarity"] = float(torch.dot(emb1, emb2).item())
        return metrics


def average_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    return {k: sum(m[k] for m in metrics_list) / len(metrics_list) for k in keys}


def format_instruction(template: str, persona: str, content: str) -> str:
    kwargs = {"persona": persona, "content": content}
    if "{few_shot_examples}" in template:
        kwargs["few_shot_examples"] = ""
    return template.format(**kwargs)


def format_persona_rewrite(
    template: str, persona: str, content: str, ground_truth: str
) -> str:
    return template.format(
        persona=persona,
        content=content,
        ground_truth=ground_truth,
    )


def build_interview_train_rewrite_fields(train_segments: List[Dict[str, str]]) -> tuple:
    content_parts = []
    gt_parts = []
    for i, seg in enumerate(train_segments, start=1):
        content_parts.append(f"Train segment {i}:\n{seg['content']}")
        gt_parts.append(f"Train segment {i} response:\n{seg['ground_truth']}")
    return "\n\n".join(content_parts), "\n\n".join(gt_parts)


async def generate_and_score(
    generator: BedrockGenerator,
    instruction_template: str,
    persona: str,
    content: str,
    ground_truth: str,
    metrics_calc: MetricsCalculator,
    source_tag: str,
) -> tuple:
    gen_prompt = format_instruction(instruction_template, persona, content)
    response = await generator.generate(gen_prompt, source=f"icl_gen_{source_tag}")
    metrics = metrics_calc.score(response, ground_truth)
    return response, metrics


async def process_debate_unit(
    unit: Dict[str, Any],
    p0: str,
    generator: BedrockGenerator,
    rewrite_template: str,
    instruction_template: str,
    metrics_calc: MetricsCalculator,
) -> Dict[str, Any]:
    content = unit["content"]
    train_gt = unit["train_ground_truth"]
    val_gt = unit["val_ground_truth"]

    rewrite_prompt = format_persona_rewrite(rewrite_template, p0, content, train_gt)
    refined_persona = await generator.generate(
        rewrite_prompt, source="icl_persona_rewrite_train"
    )

    train_response, train_metrics = await generate_and_score(
        generator,
        instruction_template,
        refined_persona,
        content,
        train_gt,
        metrics_calc,
        "train",
    )
    val_response, val_metrics = await generate_and_score(
        generator,
        instruction_template,
        refined_persona,
        content,
        val_gt,
        metrics_calc,
        "val",
    )

    return {
        "unit_id": unit["unit_id"],
        "task_info": unit["task_info"],
        "initial_persona": p0,
        "refined_persona_icl": refined_persona,
        "train": {
            "content": content,
            "ground_truth": train_gt,
            "response": train_response,
            "metrics": train_metrics,
        },
        "val": {
            "content": content,
            "ground_truth": val_gt,
            "response": val_response,
            "metrics": val_metrics,
        },
    }


async def process_interview_unit(
    unit: Dict[str, Any],
    p0: str,
    generator: BedrockGenerator,
    rewrite_template: str,
    instruction_template: str,
    metrics_calc: MetricsCalculator,
) -> Dict[str, Any]:
    train_segments = unit["train_segments"]
    val_segments = unit["val_segments"]

    rewrite_content, rewrite_gt = build_interview_train_rewrite_fields(train_segments)
    rewrite_prompt = format_persona_rewrite(
        rewrite_template, p0, rewrite_content, rewrite_gt
    )
    refined_persona = await generator.generate(
        rewrite_prompt, source="icl_persona_rewrite_train"
    )

    train_metrics_list = []
    train_responses = []
    for seg in train_segments:
        resp, m = await generate_and_score(
            generator,
            instruction_template,
            refined_persona,
            seg["content"],
            seg["ground_truth"],
            metrics_calc,
            "train",
        )
        train_responses.append(resp)
        train_metrics_list.append(m)

    val_metrics_list = []
    val_responses = []
    for seg in val_segments:
        resp, m = await generate_and_score(
            generator,
            instruction_template,
            refined_persona,
            seg["content"],
            seg["ground_truth"],
            metrics_calc,
            "val",
        )
        val_responses.append(resp)
        val_metrics_list.append(m)

    return {
        "unit_id": unit["unit_id"],
        "task_info": unit["task_info"],
        "initial_persona": p0,
        "refined_persona_icl": refined_persona,
        "train": {
            "segments": train_segments,
            "responses": train_responses,
            "metrics": average_metrics(train_metrics_list),
            "per_segment_metrics": train_metrics_list,
        },
        "val": {
            "segments": val_segments,
            "responses": val_responses,
            "metrics": average_metrics(val_metrics_list),
            "per_segment_metrics": val_metrics_list,
        },
    }


def _metrics_to_row(prefix: str, metrics: Dict[str, float]) -> Dict[str, float]:
    return {f"{prefix}_{k}": v for k, v in metrics.items()}


def save_outputs(
    output_dir: Path,
    dataset: str,
    results: List[Dict[str, Any]],
    token_usage: TokenUsageTracker,
    model_id: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    details_dir = output_dir / "details"
    details_dir.mkdir(exist_ok=True)

    train_rows = []
    val_rows = []
    for i, res in enumerate(results):
        base = {"example_id": i + 1, "unit_id": res["unit_id"]}
        base.update(res.get("task_info", {}))
        train_row = {**base, **_metrics_to_row("icl", res["train"]["metrics"])}
        val_row = {**base, **_metrics_to_row("icl", res["val"]["metrics"])}
        train_rows.append(train_row)
        val_rows.append(val_row)

    train_df = pd.DataFrame(train_rows)
    val_df = pd.DataFrame(val_rows)
    train_df.to_csv(output_dir / "evaluation_results_train.csv", index=False)
    val_df.to_csv(output_dir / "evaluation_results_val.csv", index=False)

    def build_aggregate(df: pd.DataFrame, split_name: str) -> Dict[str, Any]:
        metric_cols = [c for c in df.columns if c.startswith("icl_")]
        agg = {
            "dataset": dataset,
            "method": "icl_baseline",
            "split": split_name,
            "num_units": len(df),
            "model_id": model_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        for col in metric_cols:
            key = col.replace("icl_", "")
            agg[f"mean_icl_{key}"] = float(df[col].mean())
            agg[f"std_icl_{key}"] = float(df[col].std())
        return agg

    agg_train = build_aggregate(train_df, "train")
    agg_val = build_aggregate(val_df, "val")
    agg_train["token_usage"] = token_usage.to_dict()
    agg_val["token_usage"] = token_usage.to_dict()

    with (output_dir / "aggregate_metrics_train.json").open("w", encoding="utf-8") as f:
        json.dump(agg_train, f, indent=2)
    with (output_dir / "aggregate_metrics_val.json").open("w", encoding="utf-8") as f:
        json.dump(agg_val, f, indent=2)

    combined = {
        "train": agg_train,
        "val": agg_val,
    }
    with (output_dir / "aggregate_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    token_usage.save(str(output_dir / "token_usage.json"))

    for res in results:
        detail_path = details_dir / f"unit_{res['unit_id']}.json"
        with detail_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "initial_persona": res["initial_persona"],
                    "refined_persona_icl": res["refined_persona_icl"],
                    "task_info": res.get("task_info", {}),
                    "train": res["train"],
                    "val": res["val"],
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    print(f"\n=== ICL Baseline [{dataset}] ({len(results)} units) ===")
    print("TRAIN split:")
    print(f"  embedding_similarity: {agg_train.get('mean_icl_embedding_similarity', 0):.4f}")
    print(f"  rougeL_f1:            {agg_train.get('mean_icl_rougeL_f1', 0):.4f}")
    print(f"  bertscore_f1:         {agg_train.get('mean_icl_bertscore_f1', 0):.4f}")
    print("VAL split:")
    print(f"  embedding_similarity: {agg_val.get('mean_icl_embedding_similarity', 0):.4f}")
    print(f"  rougeL_f1:            {agg_val.get('mean_icl_rougeL_f1', 0):.4f}")
    print(f"  bertscore_f1:         {agg_val.get('mean_icl_bertscore_f1', 0):.4f}")
    print(f"  Saved: {output_dir}")


async def run_dataset(args) -> None:
    dataset = args.dataset
    if dataset not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset}")

    cfg = DATASET_REGISTRY[dataset]
    project_root = PROJECT_ROOT

    rewrite_path = Path(args.persona_rewrite_file)
    if not rewrite_path.is_absolute():
        rewrite_path = project_root / rewrite_path
    instruction_path = Path(args.instruction_prompt_file or cfg["instruction_prompt_file"])
    if not instruction_path.is_absolute():
        instruction_path = project_root / instruction_path
    persona_path = Path(args.initial_persona_file or cfg["initial_persona_file"])
    if not persona_path.is_absolute():
        persona_path = project_root / persona_path

    with rewrite_path.open("r", encoding="utf-8") as f:
        rewrite_template = f.read()
    with instruction_path.open("r", encoding="utf-8") as f:
        instruction_template = f.read()
    p0 = load_initial_persona(str(persona_path))

    units = load_dataset_units(
        dataset,
        project_root,
        n=args.length,
        seed=args.seed,
        data_path_override=args.data_path,
    )
    print(
        f"Loaded {len(units)} units for ICL on {dataset} "
        f"(seed={args.seed}, train/val split data)"
    )

    token_usage = TokenUsageTracker()
    generator = BedrockGenerator(
        model_id=args.model,
        region=args.bedrock_region,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        max_concurrency=args.max_concurrency,
        token_usage=token_usage,
    )
    token_usage.set_request_log(str(Path(args.output_dir) / "token_usage_calls.jsonl"))

    metrics_calc = MetricsCalculator()
    process_fn = process_debate_unit if dataset == "debate" else process_interview_unit

    async def run_unit(unit: Dict[str, Any]):
        try:
            return await process_fn(
                unit, p0, generator, rewrite_template, instruction_template, metrics_calc
            )
        except Exception as e:
            return e

    raw_results = await tqdm.gather(
        *[run_unit(u) for u in units], desc=f"ICL {dataset}"
    )

    results = []
    failed = 0
    for unit, outcome in zip(units, raw_results):
        if isinstance(outcome, Exception):
            failed += 1
            print(f"Failed unit {unit['unit_id']}: {type(outcome).__name__}: {outcome}")
            traceback.print_exception(type(outcome), outcome, outcome.__traceback__)
        else:
            results.append(outcome)

    if failed:
        print(f"Warning: {failed}/{len(units)} units failed.")

    if not results:
        raise RuntimeError("No successful ICL units.")

    save_outputs(
        Path(args.output_dir),
        dataset,
        results,
        token_usage,
        args.model,
    )


def main():
    parser = argparse.ArgumentParser(
        description="ICL Baseline (debate/interview, train rewrite + train/val eval)"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=list(DATASET_REGISTRY.keys()),
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--length", type=int, default=100, help="Number of speakers/people")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--bedrock_region", default="us-east-1")
    parser.add_argument("--max_tokens", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_concurrency", type=int, default=20)
    parser.add_argument(
        "--persona_rewrite_file",
        default="Evaluation/icl_baseline/prompts/persona_rewrite.txt",
    )
    parser.add_argument("--instruction_prompt_file", default=None)
    parser.add_argument("--initial_persona_file", default=None)
    args = parser.parse_args()

    if not args.model:
        args.model = os.environ.get("BEDROCK_CLAUDE_MODEL_ID", "")
    if not args.model:
        raise ValueError("Set --model or BEDROCK_CLAUDE_MODEL_ID")

    asyncio.run(run_dataset(args))


if __name__ == "__main__":
    main()
