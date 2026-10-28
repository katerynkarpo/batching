# Data

`data/raw/` and `data/processed/` are not committed (licensed datasets, reproducible).
Rebuild everything with:

    .venv/bin/python -I scripts/download_data.py --root .

## Unified record format (`data/processed/<task>/<split>.jsonl`)

    {"id": "<task>:<local id>", "task": str, "input": str, "target": Any, "meta": {...}}

| task | files | target | meta |
|---|---|---|---|
| gec | bea19_dev (4384), bea19_train (2000 sample), conll14_test (1312) | corrected sentence (annotator 0) | references per annotator, m2_file, m2_index |
| gsm8k | train (7473), test (1319) | final numeric answer | solution |
| sst5 | train (8544), validation (1101), test (2210) | label text (very negative … very positive) | label_id |
| tweeteval | train (45615), validation (2000), test (12284) | negative / neutral / positive | label_id |
| absa_e2e | rest14_{train,test} (3041/800), laptop14_{train,test} (3045/800) | list of {term, polarity} incl. conflict; [] if no aspects | domain, n_aspects, has_conflict |
| absa_atsc | rest14_{train,test} (3602/1120), laptop14_{train,test} (2313/638) | polarity (conflict dropped) | aspect, sentence_id, n_aspects_in_sentence |
| strategyqa | train (1603), test (687) | yes / no | facts, qid |
| wikitq | train (11321), dev (2831), test (4344) | list of answer strings | table {header, rows}, table_path |

Sources: W&I+LOCNESS v2.1 and CoNLL-2014 official tarballs (M2 kept in `data/raw/gec` for ERRANT / M2 scorer);
`openai/gsm8k`, `SetFit/sst5`, `cardiffnlp/tweet_eval`, `ChilleD/StrategyQA` from HF Hub;
SemEval-2014 Task 4 via InstructABSA CSVs; WikiTableQuestions v1.0.2 compact release.

Rules: WikiTQ is held-out for RQ3/RQ4. Train splits are only for few-shot demonstrations.
GEC scoring must go through the original M2 files, not the jsonl references.
