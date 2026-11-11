# Running on a rented A100

Workflow: code + processed data go up with rsync, results come back with rsync. The GPU box
never needs the raw downloads except the GEC M2 files (ERRANT scoring reads them), so
`data/raw/gec` is included.

## 1. Push the project

    export GPU=user@host           # e.g. root@1.2.3.4 -p 22022 for RunPod: use -e "ssh -p 22022"
    rsync -avz --exclude .venv --exclude .git --exclude results --exclude 'data/raw' \
          ./ $GPU:~/batching/
    rsync -avz data/raw/gec/ $GPU:~/batching/data/raw/gec/

## 2. Set up once

    ssh $GPU
    cd ~/batching && bash remote/setup_a100.sh
    export HF_TOKEN=hf_...          # gated Gemma weights

## 3. Verify, then sweep

    .venv/bin/python -m pytest -q
    .venv/bin/python scripts/run.py --config configs/smoke.yaml       # oracle, no model, ~1 min
    .venv/bin/python scripts/run.py --config configs/smoke_gpu.yaml   # Qwen3-8B, ~5 min
    tmux new -s rq1
    .venv/bin/python scripts/run.py --config configs/rq1.yaml 2>&1 | tee results/rq1.log

Every (task, model, b, perm) setting writes `summary.json` and is skipped on rerun; generations
are cached in `results/cache/generations.sqlite`, so an interrupted sweep resumes for free.

## 3b. GEC batch-size sweep (paper prompts, b = 1..200)

`configs/gec_bsweep.yaml`: 11 models x 6 prompt conditions (base / minimal_edits / taxonomy, each
zero-shot and 8-shot) x 12 batch sizes on the full BEA dev set. One process per model, because
vLLM keeps the GPU memory until the process exits:

    export GPUS="0 1 2 3"                      # one queue of models per GPU (default: GPU 0 only)
    bash remote/run_gec_bsweep.sh --smoke      # every model: 600 sentences, b in {1, 20, 200}
    tmux new -s gec
    bash remote/run_gec_bsweep.sh              # full sweep; re-run to resume

With several GPUs the script balances the queues by model size and runs them in parallel.

Check in the smoke log, per model: it loads, `trunc=0`, `missing` is not ~1.0 at b=1 (chat template
or thinking problem), and b=200 does not run out of memory. The weights of all models are ~370 GB;
on a smaller disk run with `PURGE_WEIGHTS=1`. `MODELS="..."` restricts the list.

    .venv/bin/python scripts/aggregate.py --name gec_bsweep_v1

## 4. Pull results

    rsync -avz $GPU:~/batching/results/ ./results/
    .venv/bin/python scripts/aggregate.py --name rq1_v1

## Budget (rough, vLLM on one A100 80GB, 8B model, bf16)

Per model: 7 tasks x 512 items x 16 settings (b=1 once, 5 batch sizes x 3 perms) ≈ 57k item-generations.
Short-output tasks are prompt-bound (~2-3k items/min); GSM8K and GEC dominate (~300 and ~100
completion tokens per item). Expect 1-2 GPU hours per 8B model, 2-3 for 12B. Sweep both models in
one session; keep the box for RQ2 attention runs separately (those use the `hf` backend, which is
slower, on subsets of 128 items).

## Backend notes

- `vllm`: greedy, `chat_template_kwargs` passed to `llm.chat`. Qwen3 needs `enable_thinking: false`.
- `hf`: Transformers with left padding, greedy; `micro_batch` prompts per forward. Use for RQ2
  (attention hooks) and for any model vLLM cannot load.
- `openai`: any OpenAI-compatible endpoint (`base_url`, `OPENAI_API_KEY`), for API models in RQ1.
- Switching backend changes the cache key; outputs of vllm and hf can differ at the token level.
