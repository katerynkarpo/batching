# Evaluation decisions (2026-10-28)

- **Batch output format.** Every answer on its own line, prefixed `[k]`. `split_batch_output`
  attaches continuation lines to the last marker, tolerates `1.` / `1)` / `Answer 3:` and code
  fences, records extra / duplicate numbers. Missing items -> None -> counted as wrong and
  reported in `parse_failure_rate` (kept separate from quality in all tables).
- **Classification tasks** (sst5, tweeteval, absa_atsc, strategyqa): label search longest-first;
  if an "answer is / :" cue exists, the label after the cue wins. Accuracy + macro-F1.
- **GSM8K**: `#### x` > "answer is x" > `\boxed{x}` > last number. Numeric comparison.
- **ABSA E2E**: micro P/R/F1 over (term, polarity) pairs and over terms. Extra signals:
  `hallucinated_term_rate` (predicted terms absent from the input sentence; leakage proxy),
  `empty_accuracy` (sentences with no gold aspects), `pred_terms_per_sentence`.
  Conflict labels kept in gold for E2E; dropped for ATSC (standard).
- **WikiTQ**: official evaluator ported to Python 3 (`wtq_eval.py`), answers separated by ` | `.
- **GEC**: ERRANT 3.0.2. Hypotheses are spaCy-retokenised, then annotated against the source.
  Retokenisation is chunk-wise and leaves source tokens and clitics (`'m`, `n't`, ...) alone:
  plain spaCy turns an already tokenised `I 'm` into `I ' m`, a spurious edit for any model
  that copies the tokenised input (125 of 4384 BEA-dev gold sentences changed, gold F0.5 0.984;
  after the fix 7 sentences, gold F0.5 0.997).
  References are **re-annotated with the same ERRANT** from (source, reference) pairs
  (`ref_mode="reannotate"`, default): gold-as-hypothesis then scores F0.5 = 1.0, whereas the
  original human spans give ~0.91 on BEA dev purely from alignment differences.
  `ref_mode="original"` is kept for comparison with papers that use the raw M2.
  Retokenisation changes ~2% of sentences (quote characters), costing ≤0.03 F0.5 on gold;
  unavoidable since model outputs are detokenised text.
  Missing/unparsable predictions fall back to the source (= no correction).
  `fp` from errant_compare is the overcorrection count; `unchanged_rate` is also reported.
