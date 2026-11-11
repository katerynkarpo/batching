import pytest

from batching import metrics as M
from batching.data import PROCESSED, load_jsonl
from batching.wtq_eval import denotation_match


def rec(task, i, inp, tgt, **meta):
    return {"id": f"{task}:{i}", "task": task, "input": inp, "target": tgt, "meta": meta}


def test_classification_summary():
    recs = [rec("sst5", 0, "a", "positive"), rec("sst5", 1, "b", "negative"), rec("sst5", 2, "c", "neutral")]
    raw = {"sst5:0": "positive", "sst5:1": "very positive", "sst5:2": "I don't know"}
    s, items = M.evaluate("sst5", recs, raw)
    assert s["accuracy"] == pytest.approx(1 / 3)
    assert s["parse_failure_rate"] == pytest.approx(1 / 3)
    assert items[1]["pred"] == "very positive" and not items[1]["correct"]


def test_gsm8k_numeric_compare():
    recs = [rec("gsm8k", 0, "q", "18"), rec("gsm8k", 1, "q", "1,250"), rec("gsm8k", 2, "q", "3")]
    raw = {"gsm8k:0": "... #### 18.0", "gsm8k:1": "The answer is $1250", "gsm8k:2": None}
    s, _ = M.evaluate("gsm8k", recs, raw)
    assert s["accuracy"] == pytest.approx(2 / 3)
    assert s["parse_failure_rate"] == pytest.approx(1 / 3)


def test_absa_e2e_f1_and_hallucination():
    recs = [
        rec("absa_e2e", 0, "The bread is top notch but the service was slow.",
            [{"term": "bread", "polarity": "positive"}, {"term": "service", "polarity": "negative"}]),
        rec("absa_e2e", 1, "Nice place.", []),
        rec("absa_e2e", 2, "Great sushi.", [{"term": "sushi", "polarity": "positive"}]),
    ]
    raw = {
        # one correct pair, one wrong polarity, one leaked term from another sentence
        "absa_e2e:0": '[{"term": "bread", "polarity": "positive"}, {"term": "service", "polarity": "neutral"}, {"term": "sushi", "polarity": "positive"}]',
        "absa_e2e:1": "[]",
        "absa_e2e:2": '[{"term": "sushi", "polarity": "positive"}]',
    }
    s, items = M.evaluate("absa_e2e", recs, raw)
    assert s["pair_precision"] == pytest.approx(2 / 4)
    assert s["pair_recall"] == pytest.approx(2 / 3)
    assert s["term_precision"] == pytest.approx(3 / 4)
    assert s["hallucinated_term_rate"] == pytest.approx(1 / 4)
    assert items[0]["hallucinated_terms"] == ["sushi"]
    assert s["empty_accuracy"] == 1.0
    assert s["sentence_exact_match"] == pytest.approx(2 / 3)


def test_wtq_denotation():
    assert denotation_match(["Italy"], ["italy"])
    assert denotation_match(["100,000"], ["100000"]) is False  # official evaluator: strings, not numbers w/ commas
    assert denotation_match(["5"], ["5.0"])
    assert denotation_match(["a", "b"], ["b", "a"])
    assert not denotation_match(["a", "b"], ["a"])
    recs = [rec("wikitq", 0, "q", ["Italy"]), rec("wikitq", 1, "q", ["1", "2"])]
    s, _ = M.evaluate("wikitq", recs, {"wikitq:0": "Italy", "wikitq:1": "2 | 1"})
    assert s["denotation_accuracy"] == 1.0


@pytest.mark.skipif(not (PROCESSED / "gec" / "bea19_dev.jsonl").exists(), reason="data not downloaded")
def test_gec_errant_smoke():
    rows = load_jsonl(PROCESSED / "gec" / "bea19_dev.jsonl")[:20]
    # gold references as predictions -> perfect score; unchanged -> zero recall
    s_gold, items = M.evaluate("gec", rows, {r["id"]: r["target"] for r in rows})
    assert s_gold["f05"] == pytest.approx(1.0)
    assert s_gold["exact_match"] == 1.0
    s_copy, _ = M.evaluate("gec", rows, {r["id"]: r["input"] for r in rows})
    assert s_copy["tp"] == 0 and s_copy["fp"] == 0 and s_copy["unchanged_rate"] == 1.0
    # missing predictions fall back to the source and are counted
    s_none, _ = M.evaluate("gec", rows, {})
    assert s_none["parse_failure_rate"] == 1.0
    # original human spans: runs, but alignment differences cost recall even for gold
    from batching.gec_eval import score_gec
    s_orig = score_gec(rows, {r["id"]: r["target"] for r in rows}, ref_mode="original")
    assert 0.8 < s_orig["f05"] < 1.0


@pytest.mark.skipif(not (PROCESSED / "gec" / "conll14_test.jsonl").exists(), reason="data not downloaded")
def test_gec_conll_two_annotators():
    from batching.gec_eval import retokenize
    rows = load_jsonl(PROCESSED / "gec" / "conll14_test.jsonl")[:40]
    # predicting annotator 1's version must score 1.0 (best match over annotators) whenever
    # spaCy retokenisation leaves the text unchanged; quote tokenisation differs for a few sentences
    s, _ = M.evaluate("gec", rows, {r["id"]: r["meta"]["references"]["1"] for r in rows})
    assert s["f05"] > 0.97
    stable = [r for r in rows if retokenize(r["meta"]["references"]["1"]) == r["meta"]["references"]["1"]]
    s, _ = M.evaluate("gec", stable, {r["id"]: r["meta"]["references"]["1"] for r in stable})
    assert s["f05"] == pytest.approx(1.0)


def test_gec_retokenize_keeps_existing_tokens():
    from batching.gec_eval import retokenize
    # detokenised model output is tokenised the W&I way
    assert retokenize("I'm sure he hasn't seen it.") == "I 'm sure he has n't seen it ."
    # already tokenised clitics are not split again (plain spaCy gives "I ' m")
    assert retokenize("I 'm sure he has n't seen it .") == "I 'm sure he has n't seen it ."
    # a source token that spaCy would split is kept when the model copies it
    assert retokenize("my 11-year - old son", keep=frozenset("my 11-year - old son".split())) == "my 11-year - old son"
