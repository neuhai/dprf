#!/usr/bin/env python3
"""
Build few-shot example JSON files from the LAST N items in each dataset pool,
so evaluation can use the FIRST N items without overlap.
"""

import argparse
import json
import random
from pathlib import Path

from few_shot import (
    build_depression_few_shot_example,
    build_interview_few_shot_example,
    select_items,
)


def create_interview_few_shot(
    data_dir: Path,
    pool_size: int,
    num_shots: int,
    seed: int,
) -> dict:
    json_files = sorted(data_dir.glob("*.json"))
    if len(json_files) < pool_size:
        raise ValueError(
            f"Interview pool needs {pool_size} files, found {len(json_files)} in {data_dir}"
        )

    pool_files = json_files[-pool_size:]
    rng = random.Random(seed)
    selected_files = rng.sample(pool_files, num_shots)

    examples = []
    for file_path in selected_files:
        with file_path.open("r", encoding="utf-8") as f:
            records = json.load(f)
        if not records:
            continue
        record = records[0]
        record["_source_file"] = file_path.name
        examples.append(build_interview_few_shot_example(record))

    return {
        "task": "interview",
        "seed": seed,
        "pool_size": pool_size,
        "num_shots": num_shots,
        "pool_slice": f"last_{pool_size}_files_sorted_by_name",
        "selected_source_files": [ex.get("source_file") for ex in examples],
        "examples": examples,
    }


def create_depression_few_shot(
    data_file: Path,
    pool_size: int,
    num_shots: int,
    seed: int,
) -> dict:
    with data_file.open("r", encoding="utf-8") as f:
        all_examples = json.load(f)

    if len(all_examples) < pool_size:
        raise ValueError(
            f"Depression pool needs {pool_size} examples, found {len(all_examples)}"
        )

    pool = all_examples[-pool_size:]
    rng = random.Random(seed)
    selected = rng.sample(pool, num_shots)
    examples = [build_depression_few_shot_example(ex) for ex in selected]

    return {
        "task": "depression",
        "seed": seed,
        "pool_size": pool_size,
        "num_shots": num_shots,
        "pool_slice": f"last_{pool_size}_items_in_json_array",
        "selected_ids": [ex.get("id") for ex in examples],
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create few-shot example JSON files.")
    parser.add_argument(
        "--task",
        choices=["interview", "depression", "both"],
        default="both",
    )
    parser.add_argument(
        "--interview_data_dir",
        default="Evaluation/interview/data/processed",
    )
    parser.add_argument(
        "--depression_data_file",
        default="Evaluation/depression/data/processed/depression_data.json",
    )
    parser.add_argument("--pool_size", type=int, default=100)
    parser.add_argument("--num_shots", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--interview_output",
        default="Evaluation/interview/data/few_shot_examples.json",
    )
    parser.add_argument(
        "--depression_output",
        default="Evaluation/depression/data/few_shot_examples.json",
    )
    args = parser.parse_args()

    if args.task in ("interview", "both"):
        payload = create_interview_few_shot(
            Path(args.interview_data_dir),
            args.pool_size,
            args.num_shots,
            args.seed,
        )
        out = Path(args.interview_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"Wrote {out} ({len(payload['examples'])} interview examples)")

    if args.task in ("depression", "both"):
        payload = create_depression_few_shot(
            Path(args.depression_data_file),
            args.pool_size,
            args.num_shots,
            args.seed,
        )
        out = Path(args.depression_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"Wrote {out} ({len(payload['examples'])} depression examples)")


if __name__ == "__main__":
    main()
