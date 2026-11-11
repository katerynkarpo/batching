"""Experiment runner: records -> batches (seeded permutations) -> prompts -> generations (cached)
-> per-item outputs -> metrics. Writes results/<name>/<task>/<model>/b<k>_p<perm>/.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import metrics, parsing, prompts
from .backends import Backend, Generation
from .data import EVAL_FILES, PROCESSED, load_jsonl

ROOT = Path(__file__).resolve().parents[2]

TRAIN_FILES = {
    "gec": "bea19_train.jsonl", "gsm8k": "train.jsonl", "sst5": "train.jsonl", "tweeteval": "train.jsonl",
    "absa_e2e": "{domain}_train.jsonl", "absa_atsc": "{domain}_train.jsonl",
    "strategyqa": "train.jsonl", "wikitq": "train.jsonl",
}


@dataclass
class RunConfig:
    name: str
    task: str
    model: str
    backend: str = "vllm"
    backend_kwargs: dict = field(default_factory=dict)
    data_file: str | None = None        # default: EVAL_FILES[task]
    n_items: int | None = None          # None = full eval file
    item_seed: int = 0
    batch_sizes: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    permutations: int = 3               # shuffles per batch size (b=1 always 1)
    strategy: str = "random"            # random | length_sorted
    few_shot: int = 0
    demo_seed: int = 0
    reasoning: bool = False
    results_dir: str = "results"
    gec_ref_mode: str = "reannotate"
    prompt: str | None = None           # gec only: paper prompt variant (see gec_prompts.py)
    label: str | None = None            # condition name when one task runs under several prompts

    @property
    def model_slug(self) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", self.model)

    @property
    def condition(self) -> str:
        return self.label or self.task

    @property
    def out_dir(self) -> Path:
        return ROOT / self.results_dir / self.name / self.condition / self.model_slug


# ----------------------------------------------------------------------------- cache
class GenCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=300)    # several sweep processes share the cache
        for attempt in range(60):   # switching to WAL ignores the busy timeout: retry while others start
            try:
                self.db.execute("PRAGMA journal_mode=WAL")
                break
            except sqlite3.OperationalError:
                if attempt == 59:
                    raise
                time.sleep(0.5)
        self.db.execute("CREATE TABLE IF NOT EXISTS gen (key TEXT PRIMARY KEY, value TEXT)")

    @staticmethod
    def key(backend: Backend, model: str, messages: list[dict], max_new_tokens: int) -> str:
        blob = json.dumps({"b": backend.name, "m": model, "msgs": messages, "max": max_new_tokens},
                          sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, key: str) -> Generation | None:
        row = self.db.execute("SELECT value FROM gen WHERE key=?", (key,)).fetchone()
        return Generation(**json.loads(row[0])) if row else None

    def put(self, key: str, g: Generation) -> None:
        self.db.execute("INSERT OR REPLACE INTO gen VALUES (?, ?)", (key, json.dumps(asdict(g), ensure_ascii=False)))
        self.db.commit()


# ----------------------------------------------------------------------------- batching
def make_batches(records: list[dict], b: int, seed: int, strategy: str = "random") -> list[list[dict]]:
    rng = random.Random(seed)
    recs = records[:]
    rng.shuffle(recs)
    if strategy == "length_sorted":
        recs.sort(key=lambda r: len(r["input"]))
        batches = [recs[i:i + b] for i in range(0, len(recs), b)]
        rng.shuffle(batches)
        return batches
    if strategy != "random":
        raise ValueError(strategy)
    return [recs[i:i + b] for i in range(0, len(recs), b)]


def load_eval_records(cfg: RunConfig) -> list[dict]:
    rows = load_jsonl(PROCESSED / cfg.task / (cfg.data_file or EVAL_FILES[cfg.task]))
    if cfg.n_items is not None and cfg.n_items < len(rows):
        rows = random.Random(cfg.item_seed).sample(rows, cfg.n_items)
    return rows


def load_demos(cfg: RunConfig) -> list[dict]:
    if cfg.few_shot == 0 or cfg.prompt is not None:   # paper prompts: demos are in the system prompt
        return []
    name = TRAIN_FILES[cfg.task]
    if "{domain}" in name:
        domain = (cfg.data_file or EVAL_FILES[cfg.task]).split("_")[0]
        name = name.format(domain=domain)
    rows = load_jsonl(PROCESSED / cfg.task / name)
    if cfg.task == "wikitq":  # keep demo tables small
        rows = [r for r in rows if r["meta"]["n_rows"] <= 8 and r["meta"]["n_cols"] <= 5]
    return random.Random(cfg.demo_seed).sample(rows, cfg.few_shot)


# ----------------------------------------------------------------------------- run
def run_one_setting(cfg: RunConfig, backend: Backend, cache: GenCache, records: list[dict],
                    demos: list[dict], b: int, perm: int) -> dict:
    out_dir = cfg.out_dir / f"b{b}_p{perm}"
    if (out_dir / "summary.json").exists():
        return json.loads((out_dir / "summary.json").read_text())
    out_dir.mkdir(parents=True, exist_ok=True)

    batches = make_batches(records, b, seed=1000 * perm + b, strategy=cfg.strategy)
    if cfg.prompt is not None:
        max_new = max(prompts.gec_max_new_tokens(batch) for batch in batches)
    else:
        max_new = prompts.max_new_tokens(cfg.task, b, cfg.reasoning)
    convs = [prompts.build_messages(cfg.task, batch, demos, cfg.reasoning, prompt=cfg.prompt,
                                    few_shot=cfg.few_shot) for batch in batches]
    keys = [GenCache.key(backend, cfg.model, c, max_new) for c in convs]
    gens: list[Generation | None] = [cache.get(k) for k in keys]
    todo = [i for i, g in enumerate(gens) if g is None]
    t0 = time.time()
    if todo:
        new = backend.generate([convs[i] for i in todo], max_new, batch_records=[batches[i] for i in todo])
        for i, g in zip(todo, new):
            cache.put(keys[i], g)
            gens[i] = g
    wall = time.time() - t0

    raw: dict[str, str | None] = {}
    item_rows, gen_rows = [], []
    n_missing = n_extra = n_dup = n_trunc = 0
    for bi, (batch, conv, g) in enumerate(zip(batches, convs, gens)):
        split = parsing.split_batch_output(g.text, len(batch))
        n_missing += split.n_missing
        n_extra += len(split.extra_keys)
        n_dup += len(split.duplicate_keys)
        n_trunc += int(g.extra.get("finish_reason") == "length")
        gen_rows.append({
            "batch_id": bi, "b": b, "perm": perm, "ids": [r["id"] for r in batch], "completion": g.text,
            "prompt_tokens": g.prompt_tokens, "completion_tokens": g.completion_tokens, "seconds": g.seconds,
            "finish_reason": g.extra.get("finish_reason"), "missing": split.n_missing,
            "extra_keys": split.extra_keys, "duplicate_keys": split.duplicate_keys,
            "messages": conv if bi == 0 else None, "cache_key": keys[bi],
        })
        for pos, (r, txt) in enumerate(zip(batch, split.items)):
            raw[r["id"]] = txt
            item_rows.append({"id": r["id"], "batch_id": bi, "position": pos, "batch_size": len(batch)})

    summary, items = metrics.evaluate(cfg.task, records, raw)
    pos_by_id = {x["id"]: x for x in item_rows}
    for it in items:
        it.update(pos_by_id[it["id"]])
        it["b"], it["perm"] = b, perm

    n = len(records)
    ptoks = [g.prompt_tokens for g in gens if g.prompt_tokens is not None]
    ctoks = [g.completion_tokens for g in gens if g.completion_tokens is not None]
    summary.update({
        "task": cfg.task, "condition": cfg.condition, "prompt": cfg.prompt, "few_shot": cfg.few_shot,
        "model": cfg.model, "b": b, "perm": perm, "n_batches": len(batches), "max_new_tokens": max_new,
        "primary_metric": metrics.PRIMARY_METRIC[cfg.task],
        "primary": summary[metrics.PRIMARY_METRIC[cfg.task]],
        "missing_rate": n_missing / n, "extra_keys": n_extra, "duplicate_keys": n_dup,
        "truncated_batches": n_trunc,
        "prompt_tokens_per_item": sum(ptoks) / n if ptoks else None,
        "completion_tokens_per_item": sum(ctoks) / n if ctoks else None,
        "seconds_per_item": sum(g.seconds or 0 for g in gens) / n,
        "wall_seconds": wall, "cached_batches": len(batches) - len(todo),
    })
    _write_jsonl(out_dir / "generations.jsonl", gen_rows)
    _write_jsonl(out_dir / "items.jsonl", items)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def run(cfg: RunConfig, backend: Backend, cache: GenCache | None = None) -> list[dict]:
    cache = cache or GenCache(ROOT / cfg.results_dir / "cache" / "generations.sqlite")
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    (cfg.out_dir / "run.json").write_text(json.dumps(asdict(cfg), indent=2))
    records = load_eval_records(cfg)
    demos = load_demos(cfg)
    summaries = []
    for b in cfg.batch_sizes:
        for perm in range(1 if b == 1 else cfg.permutations):
            s = run_one_setting(cfg, backend, cache, records, demos, b, perm)
            summaries.append(s)
            print(f"[{cfg.condition} | {cfg.model} | b={b} p={perm}] {s['primary_metric']}={s['primary']:.4f} "
                  f"missing={s['missing_rate']:.3f} parse_fail={s['parse_failure_rate']:.3f} "
                  f"trunc={s['truncated_batches']} cached={s['cached_batches']}/{s['n_batches']}", flush=True)
    return summaries


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
