#!/usr/bin/env python3
"""Utilities for few-shot role-playing prompts and non-overlapping data slices."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, List, Optional, Sequence


def select_items(
    items: Sequence[Any],
    count: Optional[int],
    mode: str = "random",
    seed: int = 42,
) -> List[Any]:
    """Select a subset of items by count and mode (random | first | last)."""
    if not items:
        return []
    if count is None or count <= 0 or count >= len(items):
        return list(items)

    if mode == "first":
        return list(items[:count])
    if mode == "last":
        return list(items[-count:])

    rng = random.Random(seed)
    return rng.sample(list(items), count)


def load_few_shot_records(path: str | Path) -> List[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("examples", [])
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported few-shot JSON format: {path}")


def format_few_shot_block(examples: List[dict], task: str) -> str:
    if not examples:
        return ""

    parts: List[str] = []
    for i, ex in enumerate(examples, start=1):
        if task == "interview":
            parts.append(
                f"Example {i}:\n"
                f"Context:\n{ex['input']}\n\n"
                f"Response:\n{ex['output']}"
            )
        elif task == "depression":
            parts.append(
                f"Example {i}:\n"
                f"Depression level: {ex.get('depression_level', 'unknown')}\n"
                f"Post:\n{ex['output']}"
            )
        else:
            parts.append(
                f"Example {i}:\n"
                f"Input:\n{ex['input']}\n\n"
                f"Output:\n{ex['output']}"
            )

    return "\n\n---\n\n".join(parts)


def build_interview_few_shot_example(example: dict) -> dict:
    background = example.get("background", [])
    lines = []
    for turn in background:
        speaker = turn.get("speaker", "")
        text = turn.get("text", "")
        lines.append(f"{speaker}: {text}")
    content = "Here is a conversation excerpt from an interview:\n\n" + "\n".join(lines)
    return {
        "id": example.get("id"),
        "speakername": example.get("speakername"),
        "source_file": example.get("_source_file"),
        "input": content,
        "output": example.get("ground_truth", ""),
    }


def build_depression_few_shot_example(example: dict) -> dict:
    level = example.get("depression_level", "unknown")
    content = f"Your depression severity level is: {level}"
    return {
        "id": example.get("id"),
        "depression_level": level,
        "input": content,
        "output": example.get("post", ""),
    }


def interview_background_to_content(background: list) -> str:
    lines = [f"{t.get('speaker', '')}: {t.get('text', '')}" for t in background]
    return "Here is a conversation excerpt from an interview:\n\n" + "\n".join(lines)
