"""Load train/val split units for ICL baseline (debate_variation + interview processed_val2)."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_initial_persona(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def sample_units(units: List[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    if n <= 0 or n >= len(units):
        return list(units)
    rng = random.Random(seed)
    return rng.sample(units, n)


def _interview_segment_to_fields(record: Dict[str, Any]) -> Dict[str, str]:
    background = record.get("background", [])
    lines = [f"{t.get('speaker', '')}: {t.get('text', '')}" for t in background]
    content = "Here is a conversation excerpt from an interview:\n\n" + "\n".join(lines)
    return {
        "content": content,
        "ground_truth": record.get("ground_truth", ""),
    }


def load_debate_units(data_file: Path, n: int, seed: int) -> List[Dict[str, Any]]:
    with data_file.open("r", encoding="utf-8") as f:
        records = json.load(f)

    units = []
    for ex in records:
        train_refs = ex.get("individual_utterances_train", [])
        val_refs = ex.get("individual_utterances_val", [])
        if not train_refs or not val_refs:
            continue
        content = (
            f"The debate topic is: {ex['debate_topic']}\n\n"
            f"Your position is: {ex['speaker_position']} the motion."
        )
        units.append(
            {
                "unit_id": str(ex.get("speaker_id", "unknown")).replace(" ", "_"),
                "content": content,
                "train_ground_truth": "\n\n".join(train_refs),
                "val_ground_truth": "\n\n".join(val_refs),
                "task_info": {
                    "speaker_id": ex.get("speaker_id"),
                    "debate_id": ex.get("debate_id"),
                    "debate_topic": ex.get("debate_topic"),
                    "speaker_position": ex.get("speaker_position"),
                },
            }
        )
    return sample_units(units, n, seed)


def load_interview_units(data_dir: Path, n: int, seed: int) -> List[Dict[str, Any]]:
    units = []
    for file_path in sorted(data_dir.glob("*.json")):
        with file_path.open("r", encoding="utf-8") as f:
            records = json.load(f)
        if not isinstance(records, list):
            continue

        train_segs = []
        val_segs = []
        for idx, record in enumerate(records):
            fields = _interview_segment_to_fields(record)
            seg = {
                "segment_index": idx,
                "content": fields["content"],
                "ground_truth": fields["ground_truth"],
            }
            if record.get("split") == "val":
                val_segs.append(seg)
            else:
                train_segs.append(seg)

        if not train_segs or not val_segs:
            continue

        speaker = records[0].get("speakername", file_path.stem)
        units.append(
            {
                "unit_id": speaker.replace(" ", "_").replace("/", "_"),
                "train_segments": train_segs,
                "val_segments": val_segs,
                "task_info": {
                    "speakername": speaker,
                    "source_file": file_path.name,
                    "num_train_segments": len(train_segs),
                    "num_val_segments": len(val_segs),
                },
            }
        )
    return sample_units(units, n, seed)


DATASET_REGISTRY = {
    "debate": {
        "data_path": "Evaluation/debate/data/processed/debate_variation.json",
        "initial_persona_file": "Evaluation/debate/prompts/initial_persona.txt",
        "instruction_prompt_file": "Evaluation/debate/prompts/instruction.txt",
    },
    "interview": {
        "data_path": "Evaluation/interview/data/processed_val2",
        "initial_persona_file": "Evaluation/interview/prompts/initial_persona.txt",
        "instruction_prompt_file": "Evaluation/interview/prompts/instruction.txt",
    },
}


def load_dataset_units(
    dataset: str,
    project_root: Path,
    n: int,
    seed: int,
    data_path_override: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if dataset not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from {list(DATASET_REGISTRY)}")

    cfg = DATASET_REGISTRY[dataset]
    rel = data_path_override or cfg["data_path"]
    path = Path(rel)
    if not path.is_absolute():
        path = project_root / path

    if dataset == "debate":
        return load_debate_units(path, n, seed)
    return load_interview_units(path, n, seed)
