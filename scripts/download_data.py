#!/usr/bin/env python
"""Download and normalise all benchmark datasets into data/processed/<task>/<split>.jsonl.

Unified record format:
    {"id": str, "task": str, "input": str, "target": Any, "meta": dict}

Tasks: gec, gsm8k, sst5, tweeteval, absa, strategyqa, wikitq.
Usage:
    .venv/bin/python -I scripts/download_data.py --root . [--tasks gec gsm8k ...]
"""
from __future__ import annotations

import argparse
import ast
import csv
import io
import json
import re
import sys
import tarfile
import zipfile
from pathlib import Path

import requests

ROOT = Path(".")
TIMEOUT = 300


# ----------------------------------------------------------------------------- helpers
def raw(task: str) -> Path:
    p = ROOT / "data" / "raw" / task
    p.mkdir(parents=True, exist_ok=True)
    return p


def processed(task: str) -> Path:
    p = ROOT / "data" / "processed" / task
    p.mkdir(parents=True, exist_ok=True)
    return p


def download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  cached   {dest}")
        return dest
    print(f"  download {url}")
    r = requests.get(url, timeout=TIMEOUT, stream=True)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    return dest


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote    {path}  ({len(rows)} rows)")


def rec(task: str, id_: str, input_: str, target, **meta) -> dict:
    return {"id": f"{task}:{id_}", "task": task, "input": input_, "target": target, "meta": meta}


# ----------------------------------------------------------------------------- GEC
WI_URL = "https://www.cl.cam.ac.uk/research/nl/bea2019st/data/wi+locness_v2.1.bea19.tar.gz"
CONLL_URL = "https://www.comp.nus.edu.sg/~nlp/conll14st/conll14st-test-data.tar.gz"


def parse_m2(text: str, annotator: int = 0) -> list[dict]:
    """Parse an M2 file. Returns [{source, edits_by_annotator: {id: [(s,e,type,corr)]}}]."""
    out = []
    for block in text.strip().split("\n\n"):
        lines = [l for l in block.split("\n") if l.strip()]
        if not lines or not lines[0].startswith("S "):
            continue
        src = lines[0][2:]
        edits: dict[int, list] = {}
        for l in lines[1:]:
            if not l.startswith("A "):
                continue
            parts = l[2:].split("|||")
            s, e = map(int, parts[0].split())
            etype, corr, ann = parts[1], parts[2], int(parts[5])
            edits.setdefault(ann, []).append((s, e, etype, corr))
        out.append({"source": src, "edits": edits})
    return out


def apply_edits(source: str, edits: list[tuple]) -> str:
    toks = source.split(" ")
    offset = 0
    for s, e, etype, corr in sorted(edits, key=lambda x: (x[0], x[1])):
        if etype in ("noop", "Um", "UNK") and s == -1:
            continue
        corr_toks = corr.split(" ") if corr else []
        toks[s + offset : e + offset] = corr_toks
        offset += len(corr_toks) - (e - s)
    return " ".join(toks)


def m2_to_records(task: str, m2_path: Path, prefix: str, split_meta: dict) -> list[dict]:
    items = parse_m2(m2_path.read_text(encoding="utf-8"))
    # an annotator with no lines for a sentence made no changes -> reference = source
    all_annotators = sorted({a for it in items for a in it["edits"]}) or [0]
    rows = []
    for i, it in enumerate(items):
        refs = {str(a): apply_edits(it["source"], it["edits"].get(a, [])) for a in all_annotators}
        rows.append(
            rec(
                task, f"{prefix}-{i}", it["source"], refs.get("0", next(iter(refs.values()))),
                references=refs, n_annotators=len(refs), m2_file=str(m2_path.relative_to(ROOT)),
                m2_index=i, **split_meta,
            )
        )
    return rows


def do_gec() -> None:
    task = "gec"
    r = raw(task)
    wi = download(WI_URL, r / "wi+locness_v2.1.bea19.tar.gz")
    conll = download(CONLL_URL, r / "conll14st-test-data.tar.gz")
    for tgz in (wi, conll):
        with tarfile.open(tgz) as t:
            t.extractall(r, filter="data")
    # BEA-2019 dev = W&I+LOCNESS dev, gold M2 (single annotator)
    dev_m2 = r / "wi+locness" / "m2" / "ABCN.dev.gold.bea19.m2"
    rows = m2_to_records(task, dev_m2, "bea19dev", {"source_dataset": "wi+locness", "split": "dev"})
    write_jsonl(processed(task) / "bea19_dev.jsonl", rows)
    # small train sample for few-shot demonstrations (never used for eval)
    train_m2 = r / "wi+locness" / "m2" / "ABC.train.gold.bea19.m2"
    rows = m2_to_records(task, train_m2, "bea19train", {"source_dataset": "wi+locness", "split": "train"})
    write_jsonl(processed(task) / "bea19_train.jsonl", rows[:2000])
    # CoNLL-2014 test, two annotators -> references {"0","1"}
    conll_m2 = r / "conll14st-test-data" / "noalt" / "official-2014.combined.m2"
    rows = m2_to_records(task, conll_m2, "conll14", {"source_dataset": "conll2014", "split": "test"})
    write_jsonl(processed(task) / "conll14_test.jsonl", rows)


# ----------------------------------------------------------------------------- GSM8K
def do_gsm8k() -> None:
    from datasets import load_dataset

    task = "gsm8k"
    ds = load_dataset("openai/gsm8k", "main")
    for split in ("train", "test"):
        rows = []
        for i, ex in enumerate(ds[split]):
            sol, ans = ex["answer"].rsplit("####", 1)
            ans = ans.strip().replace(",", "")
            rows.append(rec(task, f"{split}-{i}", ex["question"], ans, solution=sol.strip(), split=split))
        write_jsonl(processed(task) / f"{split}.jsonl", rows)


# ----------------------------------------------------------------------------- SST-5
def do_sst5() -> None:
    from datasets import load_dataset

    task = "sst5"
    ds = load_dataset("SetFit/sst5")
    for split in ("train", "validation", "test"):
        rows = [
            rec(task, f"{split}-{i}", ex["text"], ex["label_text"], label_id=ex["label"], split=split)
            for i, ex in enumerate(ds[split])
        ]
        write_jsonl(processed(task) / f"{split}.jsonl", rows)


# ----------------------------------------------------------------------------- TweetEval sentiment
TWEET_LABELS = {0: "negative", 1: "neutral", 2: "positive"}


def do_tweeteval() -> None:
    from datasets import load_dataset

    task = "tweeteval"
    ds = load_dataset("cardiffnlp/tweet_eval", "sentiment")
    for split in ("train", "validation", "test"):
        rows = [
            rec(task, f"{split}-{i}", ex["text"], TWEET_LABELS[ex["label"]], label_id=ex["label"], split=split)
            for i, ex in enumerate(ds[split])
        ]
        write_jsonl(processed(task) / f"{split}.jsonl", rows)


# ----------------------------------------------------------------------------- ABSA (SemEval-2014 Task 4)
ABSA_BASE = "https://raw.githubusercontent.com/kevinscaria/InstructABSA/main/Dataset/SemEval14"
ABSA_FILES = {
    ("rest14", "train"): "Train/Restaurants_Train.csv",
    ("rest14", "test"): "Test/Restaurants_Test.csv",
    ("laptop14", "train"): "Train/Laptops_Train.csv",
    ("laptop14", "test"): "Test/Laptops_Test.csv",
}


def do_absa() -> None:
    r = raw("absa")
    for (domain, split), rel in ABSA_FILES.items():
        path = download(f"{ABSA_BASE}/{rel}", r / Path(rel).name)
        e2e_rows, atsc_rows = [], []
        with open(path, encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f)):
                text = row["raw_text"]
                aspects = ast.literal_eval(row["aspectTerms"])
                # InstructABSA marks sentences with no aspect terms as a placeholder
                aspects = [a for a in aspects if a["term"].lower() != "noaspectterm"]
                pairs = [{"term": a["term"], "polarity": a["polarity"]} for a in aspects]
                sid = row["sentenceId"]
                e2e_rows.append(
                    rec("absa_e2e", f"{domain}-{split}-{i}", text, pairs, sentence_id=sid,
                        domain=domain, split=split, n_aspects=len(pairs),
                        has_conflict=any(p["polarity"] == "conflict" for p in pairs))
                )
                for j, p in enumerate(pairs):
                    if p["polarity"] == "conflict":  # standard ATSC setting drops conflict
                        continue
                    atsc_rows.append(
                        rec("absa_atsc", f"{domain}-{split}-{i}-{j}", text, p["polarity"],
                            aspect=p["term"], sentence_id=sid, domain=domain, split=split,
                            n_aspects_in_sentence=len(pairs))
                    )
        write_jsonl(processed("absa_e2e") / f"{domain}_{split}.jsonl", e2e_rows)
        write_jsonl(processed("absa_atsc") / f"{domain}_{split}.jsonl", atsc_rows)


# ----------------------------------------------------------------------------- StrategyQA
def do_strategyqa() -> None:
    from datasets import load_dataset

    task = "strategyqa"
    ds = load_dataset("ChilleD/StrategyQA")  # labelled split of the original train set
    for split in ("train", "test"):
        rows = [
            rec(task, f"{split}-{i}", ex["question"], "yes" if ex["answer"] else "no",
                answer_bool=bool(ex["answer"]), qid=ex["qid"], facts=ex["facts"], split=split)
            for i, ex in enumerate(ds[split])
        ]
        write_jsonl(processed(task) / f"{split}.jsonl", rows)


# ----------------------------------------------------------------------------- WikiTQ
WTQ_URL = "https://github.com/ppasupat/WikiTableQuestions/releases/download/v1.0.2/WikiTableQuestions-1.0.2-compact.zip"


def read_table(csv_path: Path) -> dict:
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = list(csv.reader(f))
    if not reader:
        return {"header": [], "rows": []}
    return {"header": reader[0], "rows": reader[1:]}


def do_wikitq() -> None:
    task = "wikitq"
    r = raw(task)
    z = download(WTQ_URL, r / "wtq.zip")
    base = r / "WikiTableQuestions"
    if not base.exists():
        with zipfile.ZipFile(z) as zf:
            zf.extractall(r)
    files = {"test": "pristine-unseen-tables.tsv", "dev": "random-split-1-dev.tsv", "train": "random-split-1-train.tsv"}
    for split, name in files.items():
        rows = []
        with open(base / "data" / name, encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)):
                table = read_table(base / row["context"])
                answers = row["targetValue"].split("|")
                rows.append(
                    rec(task, row["id"], row["utterance"], answers, table=table, table_path=row["context"],
                        n_rows=len(table["rows"]), n_cols=len(table["header"]), split=split)
                )
        write_jsonl(processed(task) / f"{split}.jsonl", rows)


# ----------------------------------------------------------------------------- main
TASKS = {
    "gec": do_gec,
    "gsm8k": do_gsm8k,
    "sst5": do_sst5,
    "tweeteval": do_tweeteval,
    "absa": do_absa,
    "strategyqa": do_strategyqa,
    "wikitq": do_wikitq,
}


def main() -> int:
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--tasks", nargs="*", default=list(TASKS))
    a = ap.parse_args()
    ROOT = Path(a.root).resolve()
    failed = []
    for t in a.tasks:
        print(f"== {t}")
        try:
            TASKS[t]()
        except Exception as e:  # keep going, report at the end
            print(f"  FAILED {t}: {type(e).__name__}: {e}", file=sys.stderr)
            failed.append(t)
    if failed:
        print(f"FAILED: {failed}", file=sys.stderr)
        return 1
    print("all done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
