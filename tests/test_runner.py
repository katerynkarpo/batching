import pytest

from batching import prompts
from batching.backends import DummyBackend
from batching.data import PROCESSED
from batching.runner import GenCache, RunConfig, make_batches, run

HAVE_DATA = (PROCESSED / "sst5" / "test.jsonl").exists()


def test_make_batches_is_seeded_and_complete():
    recs = [{"id": str(i), "input": "x" * (i % 7)} for i in range(50)]
    a = make_batches(recs, 8, seed=3)
    b = make_batches(recs, 8, seed=3)
    c = make_batches(recs, 8, seed=4)
    assert a == b and a != c
    assert [len(x) for x in a] == [8] * 6 + [2]
    assert sorted(r["id"] for x in a for r in x) == sorted(r["id"] for r in recs)
    ls = make_batches(recs, 8, seed=0, strategy="length_sorted")
    assert all(len(r["input"]) <= len(s["input"]) for x in ls for r, s in zip(x, x[1:]))


def test_build_messages_format():
    recs = [{"id": "a", "input": "great film", "meta": {}}, {"id": "b", "input": "awful", "meta": {}}]
    demos = [{"id": "d", "input": "ok film", "target": "neutral", "meta": {}}]
    msgs = prompts.build_messages("sst5", recs, demos)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[2]["content"] == "[1] neutral"
    assert "[1] great film\n[2] awful" in msgs[3]["content"]
    assert "numbered [1] to [2]" in msgs[3]["content"]
    assert prompts.max_new_tokens("sst5", 32) == 8 * 32 + 32


@pytest.mark.skipif(not HAVE_DATA, reason="data not downloaded")
@pytest.mark.parametrize("task,few_shot", [("sst5", 0), ("gsm8k", 0), ("absa_e2e", 2), ("strategyqa", 0), ("wikitq", 0)])
def test_oracle_end_to_end(tmp_path, task, few_shot):
    cfg = RunConfig(name="t", task=task, model="oracle", backend="dummy", n_items=16, batch_sizes=[1, 4],
                    permutations=2, few_shot=few_shot, results_dir=str(tmp_path))
    cache = GenCache(tmp_path / "cache.sqlite")
    sums = run(cfg, DummyBackend("oracle", task), cache)
    assert len(sums) == 3                       # b=1 once, b=4 twice
    assert all(s["primary"] == pytest.approx(1.0) for s in sums)
    assert all(s["missing_rate"] == 0 for s in sums)
    # rerun hits the cache and the summary files
    sums2 = run(cfg, DummyBackend("oracle", task), cache)
    assert sums2[0]["primary"] == sums[0]["primary"]


@pytest.mark.skipif(not HAVE_DATA, reason="data not downloaded")
def test_garbage_outputs_are_counted(tmp_path):
    cfg = RunConfig(name="t", task="sst5", model="junk", backend="dummy", n_items=8, batch_sizes=[4],
                    permutations=1, results_dir=str(tmp_path))
    s = run(cfg, DummyBackend("garbage", "sst5"), GenCache(tmp_path / "c.sqlite"))[0]
    assert s["missing_rate"] == 1.0 and s["parse_failure_rate"] == 1.0 and s["accuracy"] == 0.0


def test_gec_paper_prompt_messages():
    from batching import gec_prompts

    recs = [{"id": "a", "input": "He go home .", "meta": {}}, {"id": "b", "input": "I has a cat .", "meta": {}}]
    zs = prompts.build_messages("gec", recs, prompt="minimal_edits")
    fs = prompts.build_messages("gec", recs, prompt="minimal_edits", few_shot=8)
    assert [m["role"] for m in zs] == [m["role"] for m in fs] == ["system", "user"]
    assert zs[0]["content"] == gec_prompts.GEC_PROMPTS["minimal_edits"]
    assert fs[0]["content"].startswith(zs[0]["content"]) and fs[0]["content"].count("\nCorrection: ") == 8
    assert zs[1] == fs[1] and "[1] He go home .\n[2] I has a cat ." in zs[1]["content"]
    # the taxonomy prompt extends the minimal-edits prompt, as in the paper
    assert gec_prompts.GEC_PROMPTS["taxonomy"].startswith(gec_prompts.GEC_PROMPTS["minimal_edits"].split("RULES:")[0])
    with pytest.raises(ValueError):
        prompts.build_messages("gec", recs, prompt="nope")
    with pytest.raises(ValueError):
        prompts.build_messages("sst5", recs, prompt="base")
    assert prompts.gec_max_new_tokens(recs) == 2 * 4 + 8 + 2 * 5 + 8 + 64


@pytest.mark.skipif(not (PROCESSED / "gec" / "bea19_dev.jsonl").exists(), reason="data not downloaded")
def test_gec_conditions_oracle_end_to_end(tmp_path):
    cache = GenCache(tmp_path / "cache.sqlite")
    for label, few_shot in [("gec_taxonomy_zs", 0), ("gec_taxonomy_fs", 8)]:
        cfg = RunConfig(name="t", task="gec", model="oracle", backend="dummy", n_items=24, batch_sizes=[1, 10],
                        permutations=1, few_shot=few_shot, prompt="taxonomy", label=label,
                        results_dir=str(tmp_path))
        sums = run(cfg, DummyBackend("oracle", "gec"), cache)
        assert cfg.out_dir.parent.name == label
        assert [s["condition"] for s in sums] == [label, label]
        assert all(s["primary"] == pytest.approx(1.0) and s["missing_rate"] == 0 for s in sums)
