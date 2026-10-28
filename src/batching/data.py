"""Loading unified JSONL records."""
from __future__ import annotations

import json
import random
from pathlib import Path

PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"

# default evaluation file per task
EVAL_FILES = {
    "gec": "bea19_dev.jsonl",
    "gsm8k": "test.jsonl",
    "sst5": "test.jsonl",
    "tweeteval": "test.jsonl",
    "absa_e2e": "rest14_test.jsonl",
    "absa_atsc": "rest14_test.jsonl",
    "strategyqa": "test.jsonl",
    "wikitq": "test.jsonl",
}


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_task(task: str, file: str | None = None, n: int | None = None, seed: int = 0,
              root: Path = PROCESSED) -> list[dict]:
    """Load records for a task; optionally a fixed random subset of size n (seeded)."""
    rows = load_jsonl(root / task / (file or EVAL_FILES[task]))
    if n is not None and n < len(rows):
        rng = random.Random(seed)
        rows = rng.sample(rows, n)
    return rows
