"""Helpers to read prompts from the RULER benchmark JSONL files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence


DEFAULT_RULER_FIELDS: Sequence[str] = ("input", "question", "prompt")


def load_ruler_prompts(
    jsonl_path: Path,
    max_examples: int | None = None,
    candidate_fields: Sequence[str] = DEFAULT_RULER_FIELDS,
) -> List[str]:
    """Parses a subset of RULER JSONL records and extracts textual prompts."""
    records = []
    limit = max_examples if max_examples is None or max_examples > 0 else None
    with Path(jsonl_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            text = _pick_prompt_text(payload, candidate_fields)
            if text is None:
                continue
            records.append(text)
            if limit is not None and len(records) >= limit:
                break
    return records


def _pick_prompt_text(entry: dict, candidate_fields: Sequence[str]) -> str | None:
    for field in candidate_fields:
        value = entry.get(field)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                return stripped
    return None
