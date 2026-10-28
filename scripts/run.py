#!/usr/bin/env python
"""Run a sweep from a YAML config.

    python scripts/run.py --config configs/rq1.yaml [--tasks sst5 gsm8k] [--models Qwen/Qwen3-8B]
                          [--batch-sizes 1 4] [--n-items 64] [--backend dummy]

Config schema (see configs/*.yaml): name, backend, backend_kwargs, models: [{id, ...backend overrides}],
tasks: {task: {n_items, few_shot, reasoning, data_file, ...}}, batch_sizes, permutations, strategy.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from batching.backends import make_backend  # noqa: E402
from batching.runner import GenCache, ROOT, RunConfig, run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--batch-sizes", nargs="*", type=int)
    ap.add_argument("--n-items", type=int)
    ap.add_argument("--permutations", type=int)
    ap.add_argument("--backend")
    ap.add_argument("--name")
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text())
    name = a.name or cfg["name"]
    backend_name = a.backend or cfg.get("backend", "vllm")
    models = [m for m in cfg["models"] if not a.models or m["id"] in a.models]
    tasks = {t: v or {} for t, v in cfg["tasks"].items() if not a.tasks or t in a.tasks}
    cache = GenCache(ROOT / cfg.get("results_dir", "results") / "cache" / "generations.sqlite")

    for m in models:
        model_id = m["id"]
        bk = dict(cfg.get("backend_kwargs", {}))
        bk.update({k: v for k, v in m.items() if k != "id"})
        print(f"== loading {backend_name}: {model_id}", flush=True)
        for task, tcfg in tasks.items():
            backend_kwargs = dict(bk)
            if backend_name == "dummy":
                backend_kwargs = {"mode": bk.get("mode", "oracle"), "task": task}
            elif backend_name in ("hf", "vllm", "openai"):
                backend_kwargs["model"] = model_id
            if task == list(tasks)[0] or backend_name == "dummy":
                backend = make_backend(backend_name, **backend_kwargs)
            rc = RunConfig(
                name=name, task=task, model=model_id, backend=backend_name, backend_kwargs=backend_kwargs,
                data_file=tcfg.get("data_file"),
                n_items=a.n_items if a.n_items is not None else tcfg.get("n_items"),
                item_seed=tcfg.get("item_seed", cfg.get("item_seed", 0)),
                batch_sizes=a.batch_sizes or tcfg.get("batch_sizes") or cfg.get("batch_sizes", [1, 2, 4, 8, 16, 32]),
                permutations=a.permutations or tcfg.get("permutations") or cfg.get("permutations", 3),
                strategy=tcfg.get("strategy", cfg.get("strategy", "random")),
                few_shot=tcfg.get("few_shot", cfg.get("few_shot", 0)),
                demo_seed=tcfg.get("demo_seed", 0),
                reasoning=tcfg.get("reasoning", False),
                results_dir=cfg.get("results_dir", "results"),
            )
            run(rc, backend, cache)
    return 0


if __name__ == "__main__":
    sys.exit(main())
