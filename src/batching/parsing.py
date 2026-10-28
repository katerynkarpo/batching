"""Parsers: raw model text -> per-item predictions.

Two layers:
1. split_batch_output: one batched completion -> list of per-item raw strings (None = missing).
2. per-task item parsers: raw string -> prediction (None = unparsable).

Batch output convention (used by all prompts): every answer starts on its own line with the
item number in square brackets, e.g.

    [1] positive
    [2] negative

Continuation lines (chain-of-thought, multi-line answers) are attached to the last seen item.
Bare "1." / "1)" / "1:" markers are accepted as a fallback.
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field

_MARKER = re.compile(r"^\s*(?:\[(\d+)\]|(?:answer|output|item)?\s*(\d+)\s*[\.\):])\s*", re.I)
_FENCE = re.compile(r"^```[a-z]*\s*$", re.I)


@dataclass
class BatchSplit:
    items: list[str | None]            # aligned to expected batch, None = missing
    extra_keys: list[int] = field(default_factory=list)   # numbers outside 1..n
    duplicate_keys: list[int] = field(default_factory=list)
    unnumbered_prefix: str = ""        # text before the first marker

    @property
    def n_missing(self) -> int:
        return sum(x is None for x in self.items)


def split_batch_output(text: str, n: int) -> BatchSplit:
    """Split a batched completion into n per-item strings."""
    found: dict[int, list[str]] = {}
    order: list[int] = []
    current: int | None = None
    prefix: list[str] = []
    duplicates: list[int] = []
    for line in text.splitlines():
        if _FENCE.match(line):
            continue
        m = _MARKER.match(line)
        if m:
            k = int(m.group(1) or m.group(2))
            rest = line[m.end():]
            if k in found:
                duplicates.append(k)
                found[k] = [rest]       # later answer overrides earlier (model self-corrected)
            else:
                found[k] = [rest]
                order.append(k)
            current = k
        elif current is None:
            prefix.append(line)
        else:
            found[current].append(line)
    if not found:
        # no markers at all: for a single item, take the whole text
        whole = text.strip()
        items: list[str | None] = [whole if (n == 1 and whole) else None] + [None] * (n - 1)
        return BatchSplit(items=items, unnumbered_prefix="" if n == 1 else whole)
    items = []
    for i in range(1, n + 1):
        if i in found:
            items.append("\n".join(found[i]).strip() or None)
        else:
            items.append(None)
    extra = sorted(k for k in found if k < 1 or k > n)
    return BatchSplit(items=items, extra_keys=extra, duplicate_keys=sorted(set(duplicates)),
                      unnumbered_prefix="\n".join(prefix).strip())


# ----------------------------------------------------------------------------- item parsers
_WORD = r"(?<![a-z])({})(?![a-z])"


def parse_label(text: str | None, labels: list[str]) -> str | None:
    """Find one of `labels` in text. Longer labels first so 'very negative' beats 'negative'.
    If several distinct labels appear, take the one that appears last after an 'answer' cue,
    otherwise the first one in the text."""
    if not text:
        return None
    t = text.lower().replace("_", " ")
    ordered = sorted(labels, key=len, reverse=True)
    pat = re.compile(_WORD.format("|".join(re.escape(l.lower()) for l in ordered)))
    cue = re.search(r"(answer|label|sentiment|polarity|verdict)\s*(is|:)\s*", t)
    if cue:
        m = pat.search(t, cue.end())
        if m:
            return m.group(1)
    m = pat.search(t)
    return m.group(1) if m else None


_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def normalize_number(s: str) -> str | None:
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        v = float(s)
    except ValueError:
        return None
    return str(int(v)) if v == int(v) else str(v)


def parse_number(text: str | None) -> str | None:
    """Final numeric answer. Prefers '#### x', 'answer is x', 'answer: x', then the last number."""
    if not text:
        return None
    t = text.replace("\u2212", "-")
    for pat in (r"####\s*\$?(-?[\d,]*\.?\d+)", r"answer\s*(?:is|:)\s*\$?(-?[\d,]*\.?\d+)",
                r"\\boxed\{\$?(-?[\d,]*\.?\d+)\}"):
        ms = re.findall(pat, t, flags=re.I)
        if ms:
            return normalize_number(ms[-1])
    ms = _NUM.findall(t)
    return normalize_number(ms[-1]) if ms else None


def parse_yes_no(text: str | None) -> str | None:
    lab = parse_label(text, ["yes", "no", "true", "false"])
    return {"true": "yes", "false": "no"}.get(lab, lab) if lab else None


_POLARITIES = {"positive", "negative", "neutral", "conflict"}


def parse_absa_pairs(text: str | None) -> list[dict] | None:
    """Parse '[{"term": ..., "polarity": ...}, ...]'. Accepts JSON, Python literals,
    [[term, polarity], ...], or 'none'/'[]' for no aspects. Returns None if unparsable."""
    if text is None:
        return None
    t = text.strip()
    if not t:
        return None
    if t.lower() in {"none", "[]", "no aspects", "no aspect terms", "n/a", "-"}:
        return []
    start, end = t.find("["), t.rfind("]")
    if start == -1 or end == -1 or end < start:
        return None
    blob = t[start:end + 1]
    obj = None
    for loader in (json.loads, ast.literal_eval):
        try:
            obj = loader(blob)
            break
        except Exception:
            continue
    if not isinstance(obj, list):
        return None
    pairs = []
    for it in obj:
        if isinstance(it, dict):
            term = it.get("term") or it.get("aspect") or it.get("aspect_term")
            pol = it.get("polarity") or it.get("sentiment")
        elif isinstance(it, (list, tuple)) and len(it) == 2:
            term, pol = it
        else:
            return None
        if not isinstance(term, str) or not isinstance(pol, str):
            return None
        pol = pol.strip().lower()
        if pol not in _POLARITIES:
            return None
        pairs.append({"term": term.strip(), "polarity": pol})
    return pairs


def parse_gec(text: str | None) -> str | None:
    if text is None:
        return None
    t = text.strip()
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'":
        t = t[1:-1].strip()
    return t or None


def parse_wtq(text: str | None) -> list[str] | None:
    """Answers separated by ' | '. 'none' is not a valid answer in WikiTQ."""
    if text is None:
        return None
    t = text.strip().strip("\"'")
    if not t:
        return None
    parts = [p.strip() for p in t.split("|")]
    parts = [p for p in parts if p]
    return parts or None


SST5_LABELS = ["very negative", "negative", "neutral", "positive", "very positive"]
TWEET_LABELS = ["negative", "neutral", "positive"]
ATSC_LABELS = ["positive", "negative", "neutral"]


def parse_item(task: str, text: str | None):
    """Dispatch by task name as used in data/processed."""
    if task == "gsm8k":
        return parse_number(text)
    if task == "sst5":
        return parse_label(text, SST5_LABELS)
    if task == "tweeteval":
        return parse_label(text, TWEET_LABELS)
    if task == "absa_atsc":
        return parse_label(text, ATSC_LABELS)
    if task == "absa_e2e":
        return parse_absa_pairs(text)
    if task == "strategyqa":
        return parse_yes_no(text)
    if task == "gec":
        return parse_gec(text)
    if task == "wikitq":
        return parse_wtq(text)
    raise ValueError(f"unknown task {task}")
