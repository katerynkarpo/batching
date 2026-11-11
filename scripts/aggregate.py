#!/usr/bin/env python
"""Aggregate results/<name>/**/summary.json into tables.

    python scripts/aggregate.py --name rq1_v1 [--out results/rq1_v1/tables]

Produces:
  settings.csv   one row per (condition, model, b, perm)
  by_b.csv       mean over permutations per (condition, model, b), delta vs b=1, paired-bootstrap
                 95% CI on the per-item primary signal (for tasks with per-item `correct`)

`condition` is the task name, or the label when one task runs under several prompts.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from batching.runner import ROOT  # noqa: E402


def paired_bootstrap(a: dict[str, float], b: dict[str, float], n_boot: int = 2000, seed: int = 0):
    """95% CI of mean(b) - mean(a) over items paired by id."""
    ids = sorted(set(a) & set(b))
    diffs = [b[i] - a[i] for i in ids]
    rng = random.Random(seed)
    n = len(diffs)
    stats = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return stats[int(0.025 * n_boot)], stats[int(0.975 * n_boot)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out")
    a = ap.parse_args()
    base = ROOT / a.results_dir / a.name
    out = Path(a.out) if a.out else base / "tables"
    out.mkdir(parents=True, exist_ok=True)

    rows, items = [], defaultdict(dict)   # items[(condition, model, b)][id] -> list of correct over perms
    for sp in base.glob("*/*/b*_p*/summary.json"):
        s = json.loads(sp.read_text())
        s.setdefault("condition", s["task"])
        rows.append({k: v for k, v in s.items() if not isinstance(v, (dict, list))})
        ip = sp.with_name("items.jsonl")
        if ip.exists():
            key = (s["condition"], s["model"], s["b"])
            for line in open(ip, encoding="utf-8"):
                it = json.loads(line)
                if "correct" in it:
                    items[key].setdefault(it["id"], []).append(float(it["correct"]))
    if not rows:
        print("no summaries found under", base)
        return 1
    df = pd.DataFrame(rows).sort_values(["condition", "model", "b", "perm"])
    df.to_csv(out / "settings.csv", index=False)

    num = [c for c in df.columns if df[c].dtype.kind in "fi" and c not in ("b", "perm")]
    agg = df.groupby(["condition", "model", "b"])[num].mean().reset_index()
    agg["n_perms"] = df.groupby(["condition", "model", "b"]).size().values
    first = df.groupby(["condition", "model", "b"])[["task", "primary_metric"]].first().reset_index()
    agg["task"], agg["primary_metric"] = first["task"].values, first["primary_metric"].values
    base1 = agg[agg.b == 1].set_index(["condition", "model"])["primary"]
    agg["delta_vs_b1"] = [p - base1.get((t, m), float("nan")) for t, m, p in zip(agg.condition, agg.model, agg.primary)]
    lo, hi = [], []
    for t, m, b in zip(agg.condition, agg.model, agg.b):
        a1, ab = items.get((t, m, 1)), items.get((t, m, b))
        if b == 1 or not a1 or not ab:
            lo.append(float("nan")); hi.append(float("nan")); continue
        ma = {i: sum(v) / len(v) for i, v in a1.items()}
        mb = {i: sum(v) / len(v) for i, v in ab.items()}
        l, h = paired_bootstrap(ma, mb)
        lo.append(l); hi.append(h)
    agg["delta_ci_lo"], agg["delta_ci_hi"] = lo, hi
    cols = ["condition", "model", "b", "n_perms", "primary_metric", "primary", "delta_vs_b1", "delta_ci_lo", "delta_ci_hi",
            "parse_failure_rate", "missing_rate", "truncated_batches", "prompt_tokens_per_item",
            "completion_tokens_per_item", "seconds_per_item"]
    agg = agg[[c for c in cols if c in agg.columns] + [c for c in agg.columns if c not in cols]]
    agg.to_csv(out / "by_b.csv", index=False)
    pd.set_option("display.width", 200)
    print(agg[[c for c in cols[:12] if c in agg.columns]].to_string(index=False))
    print(f"\nwrote {out}/settings.csv and {out}/by_b.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
