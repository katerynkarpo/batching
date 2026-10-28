"""GEC scoring with ERRANT against the original M2 reference files.

Pipeline: hypotheses -> spaCy-retokenised -> ERRANT annotate(orig, hyp) -> hyp.m2
          reference blocks picked from the original M2 by m2_index -> ref.m2
          errant_compare hyp.m2 ref.m2 -> P / R / F0.5 (+ TP / FP / FN).

FP (edits the model made that no annotator made) is the overcorrection count.
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def _annotator():
    import errant

    return errant.load("en")


def retokenize(text: str) -> str:
    """Tokenise model output the way the W&I / CoNLL references are tokenised (spaCy)."""
    ann = _annotator()
    return " ".join(t.text for t in ann.nlp.tokenizer(text))


def m2_blocks(m2_path: Path) -> list[str]:
    return [b for b in m2_path.read_text(encoding="utf-8").strip().split("\n\n") if b.strip()]


def _edit_lines(orig_doc, cor: str, annotator_id: int) -> list[str]:
    ann = _annotator()
    edits = ann.annotate(orig_doc, ann.parse(cor))
    if not edits:
        return [f"A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||{annotator_id}"]
    return [f"A {e.o_start} {e.o_end}|||{e.type}|||{e.c_str}|||REQUIRED|||-NONE-|||{annotator_id}" for e in edits]


def hyp_to_m2(orig: str, hyp: str) -> str:
    ann = _annotator()
    return "\n".join(["S " + orig] + _edit_lines(ann.parse(orig), hyp, 0))


def ref_to_m2(orig: str, references: dict[str, str]) -> str:
    """Re-annotate (source, reference) pairs with ERRANT so reference edit spans use the same
    alignment as the hypotheses. Keeps one annotator id per reference."""
    ann = _annotator()
    o = ann.parse(orig)
    lines = ["S " + orig]
    for k, ref in sorted(references.items(), key=lambda kv: int(kv[0])):
        lines += _edit_lines(o, ref, int(k))
    return "\n".join(lines)


def errant_compare(hyp_m2: Path, ref_m2: Path) -> dict:
    cmd = [str(Path(sys.executable).with_name("errant_compare")), "-hyp", str(hyp_m2), "-ref", str(ref_m2)]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    # table: TP  FP  FN  Prec  Rec  F0.5
    m = re.search(r"TP\s+FP\s+FN\s+Prec\s+Rec\s+F0\.5\s*\n\s*(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", out)
    if not m:
        raise RuntimeError(f"could not parse errant_compare output:\n{out}")
    tp, fp, fn = map(int, m.groups()[:3])
    p, r, f = map(float, m.groups()[3:])
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f05": f}


def score_gec(records: list[dict], predictions: dict[str, str | None], workdir: Path | None = None,
              ref_mode: str = "reannotate") -> dict:
    """records: unified GEC records (same m2_file). predictions: id -> corrected text (None = failed).
    Failed / missing predictions fall back to the source (i.e. 'no correction') and are counted.

    ref_mode:
      "reannotate" (default): references re-aligned with ERRANT from (source, reference) pairs,
                              so gold-as-hypothesis scores exactly 1.0 (consistent alignment).
      "original":             edit spans taken verbatim from the original M2 file (human spans);
                              alignment differences cost ~0.09 F0.5 even for gold hypotheses.
    """
    m2_files = {r["meta"]["m2_file"] for r in records}
    assert len(m2_files) == 1, f"records must come from one M2 file, got {m2_files}"
    blocks = m2_blocks(ROOT / m2_files.pop())
    recs = sorted(records, key=lambda r: r["meta"]["m2_index"])

    hyp_lines, ref_lines = [], []
    n_failed = n_unchanged = 0
    for r in recs:
        pred = predictions.get(r["id"])
        if pred is None:
            n_failed += 1
            hyp = r["input"]
        else:
            hyp = retokenize(pred)
        if hyp == r["input"]:
            n_unchanged += 1
        hyp_lines.append(hyp_to_m2(r["input"], hyp))
        if ref_mode == "reannotate":
            ref_lines.append(ref_to_m2(r["input"], r["meta"]["references"]))
        elif ref_mode == "original":
            ref_lines.append(blocks[r["meta"]["m2_index"]])
        else:
            raise ValueError(ref_mode)

    tmp = Path(tempfile.mkdtemp(dir=workdir))
    hyp_path, ref_path = tmp / "hyp.m2", tmp / "ref.m2"
    hyp_path.write_text("\n\n".join(hyp_lines) + "\n", encoding="utf-8")
    ref_path.write_text("\n\n".join(ref_lines) + "\n", encoding="utf-8")
    res = errant_compare(hyp_path, ref_path)
    res.update({
        "n": len(recs),
        "parse_failure_rate": n_failed / len(recs),
        "unchanged_rate": n_unchanged / len(recs),
        "edits_per_sentence": (res["tp"] + res["fp"]) / len(recs),
        "ref_mode": ref_mode,
        "m2_dir": str(tmp),
    })
    return res
