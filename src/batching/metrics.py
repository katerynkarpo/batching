"""Per-task scoring. Entry point: evaluate(task, records, raw_outputs) -> (summary, per_item).

raw_outputs: id -> raw per-item model text (already split out of the batch), None = missing.
Unparsable / missing predictions count as wrong and are reported in parse_failure_rate.
"""
from __future__ import annotations

from collections import Counter

from sklearn.metrics import f1_score

from . import parsing
from .wtq_eval import denotation_match


def _classification(task: str, records: list[dict], raw: dict[str, str | None], labels: list[str]):
    items, y_true, y_pred = [], [], []
    for r in records:
        pred = parsing.parse_item(task, raw.get(r["id"]))
        ok = pred == r["target"]
        items.append({"id": r["id"], "raw": raw.get(r["id"]), "pred": pred, "target": r["target"], "correct": ok})
        y_true.append(r["target"])
        y_pred.append(pred if pred is not None else "__none__")
    n = len(records)
    summary = {
        "n": n,
        "accuracy": sum(i["correct"] for i in items) / n,
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "parse_failure_rate": sum(i["pred"] is None for i in items) / n,
        "pred_distribution": dict(Counter(y_pred)),
    }
    return summary, items


def _gsm8k(records, raw):
    items = []
    for r in records:
        pred = parsing.parse_number(raw.get(r["id"]))
        tgt = parsing.normalize_number(r["target"])
        ok = pred is not None and abs(float(pred) - float(tgt)) < 1e-6
        items.append({"id": r["id"], "raw": raw.get(r["id"]), "pred": pred, "target": tgt, "correct": ok})
    n = len(records)
    return {
        "n": n,
        "accuracy": sum(i["correct"] for i in items) / n,
        "parse_failure_rate": sum(i["pred"] is None for i in items) / n,
        "mean_output_chars": sum(len(i["raw"] or "") for i in items) / n,
    }, items


def _absa_e2e(records, raw):
    """Micro P/R/F1 over (term, polarity) pairs and over terms alone.
    hallucinated_term_rate: predicted terms that do not occur in the input sentence
    (the behavioural leakage signal). empty_accuracy: sentences with no gold aspects
    for which the model correctly returned an empty list."""
    tp = fp = fn = 0
    tp_t = fp_t = fn_t = 0
    n_pred_terms = n_halluc = 0
    empty_total = empty_ok = 0
    items = []
    for r in records:
        pred = parsing.parse_absa_pairs(raw.get(r["id"]))
        gold = [(p["term"].lower(), p["polarity"]) for p in r["target"]]
        gold_terms = Counter(t for t, _ in gold)
        if pred is None:
            pp, pt = [], Counter()
        else:
            pp = [(p["term"].lower(), p["polarity"]) for p in pred]
            pt = Counter(t for t, _ in pp)
        g, p_ = Counter(gold), Counter(pp)
        tp += sum((g & p_).values()); fp += sum((p_ - g).values()); fn += sum((g - p_).values())
        tp_t += sum((gold_terms & pt).values()); fp_t += sum((pt - gold_terms).values()); fn_t += sum((gold_terms - pt).values())
        text = r["input"].lower()
        halluc = [t for t in pt if t not in text]
        n_pred_terms += sum(pt.values()); n_halluc += len(halluc)
        if not gold:
            empty_total += 1
            empty_ok += int(pred == [])
        items.append({"id": r["id"], "raw": raw.get(r["id"]), "pred": pred, "target": r["target"],
                      "correct": pred is not None and g == p_, "hallucinated_terms": halluc})

    def prf(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * rc / (p + rc) if p + rc else 0.0
        return p, rc, f

    p, rc, f = prf(tp, fp, fn)
    pt_, rt_, ft_ = prf(tp_t, fp_t, fn_t)
    n = len(records)
    return {
        "n": n,
        "pair_precision": p, "pair_recall": rc, "pair_f1": f,
        "term_precision": pt_, "term_recall": rt_, "term_f1": ft_,
        "sentence_exact_match": sum(i["correct"] for i in items) / n,
        "parse_failure_rate": sum(i["pred"] is None for i in items) / n,
        "hallucinated_term_rate": n_halluc / n_pred_terms if n_pred_terms else 0.0,
        "empty_accuracy": empty_ok / empty_total if empty_total else None,
        "pred_terms_per_sentence": n_pred_terms / n,
    }, items


def _wikitq(records, raw):
    items = []
    for r in records:
        pred = parsing.parse_wtq(raw.get(r["id"]))
        ok = pred is not None and denotation_match(r["target"], pred)
        items.append({"id": r["id"], "raw": raw.get(r["id"]), "pred": pred, "target": r["target"], "correct": ok})
    n = len(records)
    return {
        "n": n,
        "denotation_accuracy": sum(i["correct"] for i in items) / n,
        "parse_failure_rate": sum(i["pred"] is None for i in items) / n,
    }, items


def _gec(records, raw):
    from .gec_eval import score_gec

    preds = {r["id"]: parsing.parse_gec(raw.get(r["id"])) for r in records}
    summary = score_gec(records, preds)
    items = [{"id": r["id"], "raw": raw.get(r["id"]), "pred": preds[r["id"]], "target": r["target"],
              "exact_match": preds[r["id"]] is not None and preds[r["id"]].split() == r["target"].split()}
             for r in records]
    summary["exact_match"] = sum(i["exact_match"] for i in items) / len(items)
    return summary, items


PRIMARY_METRIC = {
    "gec": "f05", "gsm8k": "accuracy", "sst5": "accuracy", "tweeteval": "accuracy",
    "absa_atsc": "accuracy", "absa_e2e": "pair_f1", "strategyqa": "accuracy", "wikitq": "denotation_accuracy",
}


def evaluate(task: str, records: list[dict], raw: dict[str, str | None]) -> tuple[dict, list[dict]]:
    if task == "sst5":
        return _classification(task, records, raw, parsing.SST5_LABELS)
    if task == "tweeteval":
        return _classification(task, records, raw, parsing.TWEET_LABELS)
    if task == "absa_atsc":
        return _classification(task, records, raw, parsing.ATSC_LABELS)
    if task == "strategyqa":
        return _classification(task, records, raw, ["yes", "no"])
    if task == "gsm8k":
        return _gsm8k(records, raw)
    if task == "absa_e2e":
        return _absa_e2e(records, raw)
    if task == "wikitq":
        return _wikitq(records, raw)
    if task == "gec":
        return _gec(records, raw)
    raise ValueError(f"unknown task {task}")
