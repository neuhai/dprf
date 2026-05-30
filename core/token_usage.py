#!/usr/bin/env python3
"""Aggregate LLM API token usage across an evaluation run."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class TokenUsageTracker:
    """Thread-safe accumulator for input/output tokens across API calls."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_requests: int = 0
    by_source: Dict[str, Dict[str, int]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _request_log_path: Optional[str] = None

    def set_request_log(self, path: str) -> None:
        self._request_log_path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        input_tokens: int,
        output_tokens: int,
        source: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        input_tokens = int(input_tokens or 0)
        output_tokens = int(output_tokens or 0)
        with self._lock:
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_requests += 1

            bucket = self.by_source.setdefault(
                source,
                {"input_tokens": 0, "output_tokens": 0, "requests": 0},
            )
            bucket["input_tokens"] += input_tokens
            bucket["output_tokens"] += output_tokens
            bucket["requests"] += 1

            if self._request_log_path:
                entry = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "source": source,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                }
                if metadata:
                    entry["metadata"] = metadata
                with open(self._request_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "total_tokens": self.total_tokens,
                "total_requests": self.total_requests,
                "by_source": dict(self.by_source),
            }

    def save(self, path: str) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


def record_openai_usage(tracker: Optional[TokenUsageTracker], response: Any, source: str) -> None:
    if tracker is None or response is None:
        return
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    if prompt_tokens is None and completion_tokens is None:
        return
    tracker.record(
        prompt_tokens or 0,
        completion_tokens or 0,
        source=source,
        metadata={"provider": "openai"},
    )


def record_bedrock_usage(tracker: Optional[TokenUsageTracker], response: Any, source: str) -> None:
    if tracker is None or not isinstance(response, dict):
        return
    usage = response.get("usage") or {}
    input_tokens = (
        usage.get("inputTokens")
        or usage.get("input_tokens")
        or 0
    )
    output_tokens = (
        usage.get("outputTokens")
        or usage.get("output_tokens")
        or 0
    )
    if not input_tokens and not output_tokens:
        return
    tracker.record(
        int(input_tokens),
        int(output_tokens),
        source=source,
        metadata={"provider": "bedrock"},
    )


def record_estimated_usage(
    tracker: Optional[TokenUsageTracker],
    input_tokens: int,
    output_tokens: int,
    source: str,
) -> None:
    if tracker is None:
        return
    tracker.record(
        input_tokens,
        output_tokens,
        source=f"{source}_estimated",
        metadata={"provider": "estimated"},
    )
