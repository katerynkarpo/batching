# Related work

## Srivastava et al. 2025/2026 — "Batch Prompting Suppresses Overthinking in Reasoning Models"
arXiv 2511.04108 (v1 Nov 2025, v4 Feb 2026). Found by Kateryna, 2026-11-02.

**What they do.** Batch prompting on *reasoning* models only (DeepSeek-R1, OpenAI o1, API, closed
box). 13 benchmarks, 100 samples each, batch sizes 1/5/10/15, homogeneous batches. Main result:
reasoning tokens drop ~76% (≈2950 → 710) while accuracy is flat or slightly up on average
(75.0 → 77–79 at b=15); non-monotonic, several subtasks drop at b=5 (e.g. ACE argument
identification 31.8 → 25.2), GSM8K and Penguins drop ~1 point at b=15 for o1. Explicit
"think less" instructions fail to reduce overthinking; batching does.

**Mechanisms.** Three *conjectures*, explicitly not tested: (1) shared-context pressure
(shorter traces when queries share the context), (2) sequential anchoring (earlier answers
bias later ones toward brevity / format via in-context pattern matching), (3) implicit difficulty
calibration. Evidence = token statistics, one qualitative case, manual annotation of 500 traces
for hedging ("wait" 21 → 1 on one task). No attention analysis, no ablations, no predictor.
Future work: "adaptive batch sizes based on estimated query difficulty".

**Overlap with our plan.** Partial overlap with RQ1 (characterisation), none with RQ2–RQ4:
- Their (1) is the behavioural face of our attention-dilution hypothesis; their (2) is our
  in-batch ICL / anchoring alternative. They state they have not isolated either. RQ2 does.
- No criterion, no tool, no transfer evaluation → RQ3/RQ4 untouched; they name it as future work.
- Closed models only → nothing on open-weight internals.
- Homogeneous batches only; heterogeneous mixtures flagged as untested → our mixed-batch control
  (length-sorted vs random, cross-task mixing) covers it.

**What to borrow.**
- Add a reasoning-mode condition: Qwen3-8B with `enable_thinking: true` on GSM8K/StrategyQA,
  log reasoning tokens per item. Replicates their effect on an open model, then RQ2 can look
  inside it. Cheap: one config flag.
- Add black-box output signals to RQ3 features: reasoning/completion tokens per item vs b,
  hedging-word rate ("wait", "but", "let me re-check"), answer-length shift.
- Cite as the main evidence that batching changes *decoding behaviour*, not only quality.

## Cheng, Kasai & Yu 2023 — Batch Prompting (EMNLP Industry)
Original method. Gains in cost/latency; quality degrades with larger b on some tasks, worse for
long/complex inputs. No mechanism.

## Karpo & Chernodub 2026 — Larger context window, fewer overcorrections (Findings EMNLP)
Batching improves minimal-edit GEC F0.5 across LLM families by reducing overcorrection.
Attention dilution proposed as explanation, unconfirmed. Starting point of this thesis.

## To read / add
- lost-in-the-middle / positional effects (Liu et al. 2023)
- many-shot ICL, batch calibration (Zhou et al. 2023), attention sinks (Xiao et al. 2023)
- in-context interference between examples; overthinking surveys (Huang et al. 2025, cited by
  Srivastava et al.)
