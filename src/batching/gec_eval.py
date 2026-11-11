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


# Clitics that are single tokens in the W&I / CoNLL tokenisation. spaCy splits them correctly
# inside a word ("I'm" -> "I 'm") but breaks them when they already stand alone ("'m" -> "' m").
_CLITICS = frozenset("'m 's 're 've 'd 'll n't 'S 'M 'RE 'VE 'D 'LL N'T n’t ’m ’s ’re ’ve ’d ’ll".split())


def retokenize(text: str, keep: frozenset[str] = frozenset()) -> str:
    """Tokenise model output the way the W&I / CoNLL references are tokenised (spaCy).

    spaCy tokenises each whitespace-separated chunk on its own, so the output is processed chunk
    by chunk and a chunk is left alone when it is already a token: a clitic, or any token of the
    source sentence (`keep`). Without this a model that copies the tokenised input ("I 'm") is
    charged a spurious edit ("I ' m")."""
    tok = _annotator().nlp.tokenizer
    out: list[str] = []
    for chunk in text.split():
        if chunk in keep or chunk in _CLITICS:
            out.append(chunk)
        else:
            out.extend(t.text for t in tok(chunk))
    return " ".join(out)


def m2_blocks(m2_path: Path) -> list[str]:
    return [b for b in m2_path.read_text(encoding="utf-8").strip().split("\n\n") if b.strip()]


def _edit_lines(orig_doc, cor: str, annotator_id: int) -> list[str]:
    ann = _annotator()
    edits = ann.annotate(orig_doc, ann.parse(cor))
    if not edits:
        return [f"A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||{annotator_id}"]
    return [f"A {e.o_start} {e.o_end}|||{e.type}|||{e.c_str}|||REQUIRED|||-NONE-|||{annotator_id}" for e in edits]


# A sweep scores the same sentences once per (model, prompt, batch size): source parses and
# reference annotations never change, and most hypotheses repeat across batch sizes.
@lru_cache(maxsize=20_000)
def _parse(text: str):
    return _annotator().parse(text)


@lru_cache(maxsize=500_000)
def hyp_to_m2(orig: str, hyp: str) -> str:
    return "\n".join(["S " + orig] + _edit_lines(_parse(orig), hyp, 0))


@lru_cache(maxsize=20_000)
def _ref_to_m2(orig: str, refs: tuple[tuple[int, str], ...]) -> str:
    o = _parse(orig)
    lines = ["S " + orig]
    for k, ref in refs:
        lines += _edit_lines(o, ref, k)
    return "\n".join(lines)


def ref_to_m2(orig: str, references: dict[str, str]) -> str:
    """Re-annotate (source, reference) pairs with ERRANT so reference edit spans use the same
    alignment as the hypotheses. Keeps one annotator id per reference."""
    return _ref_to_m2(orig, tuple(sorted((int(k), ref) for k, ref in references.items())))


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
            hyp = retokenize(pred, keep=frozenset(r["input"].split()))
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
