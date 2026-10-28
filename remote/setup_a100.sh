#!/usr/bin/env bash
# One-time setup on a fresh CUDA box (Ubuntu 22.04+, driver >= 535, e.g. RunPod / Lambda A100 80GB).
# Usage: bash remote/setup_a100.sh   (from ~/batching after rsync, see remote/README.md)
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv venv --python 3.12 .venv
# vLLM pins its own torch build; let it drive the resolution
uv pip install --python .venv/bin/python -e ".[gpu]"
.venv/bin/python -m spacy download en_core_web_sm

# Hugging Face cache on the big disk if there is one (RunPod: /workspace)
if [ -d /workspace ]; then
  mkdir -p /workspace/hf
  echo 'export HF_HOME=/workspace/hf' >> ~/.bashrc
  export HF_HOME=/workspace/hf
fi

echo
echo "GPU:"; nvidia-smi --query-gpu=name,memory.total --format=csv
echo
echo "Next:"
echo "  1. export HF_TOKEN=...   (needed for gated Gemma weights)"
echo "  2. .venv/bin/python -m pytest -q                      # metrics/parsers"
echo "  3. .venv/bin/python scripts/run.py --config configs/smoke.yaml      # no-model pipeline check"
echo "  4. .venv/bin/python scripts/run.py --config configs/smoke_gpu.yaml  # model check"
echo "  5. tmux new -s rq1; .venv/bin/python scripts/run.py --config configs/rq1.yaml 2>&1 | tee results/rq1.log"
