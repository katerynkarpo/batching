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
