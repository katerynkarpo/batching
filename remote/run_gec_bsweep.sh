#!/usr/bin/env bash
# GEC batch-size sweep, one process per model: vLLM does not release GPU memory inside a
# process, so two models loaded in one run do not fit on the card.
#
#   bash remote/run_gec_bsweep.sh                    # full sweep on GPU 0
#   GPUS="0 1 2 3" bash remote/run_gec_bsweep.sh     # one queue of models per GPU, in parallel
#   bash remote/run_gec_bsweep.sh --smoke            # every model, 600 sentences, b in {1, 20, 200}, one prompt
#   MODELS="Qwen/Qwen3-8B google/gemma-4-12B-it" bash remote/run_gec_bsweep.sh
#   PURGE_WEIGHTS=1 bash remote/run_gec_bsweep.sh    # delete each model's weights after it finishes
# Arguments after --smoke are passed to scripts/run.py (e.g. --tasks gec_taxonomy_zs).
#
# With several GPUs the models are spread so that every queue carries a similar total size
# (largest first, each to the least loaded GPU); run time grows roughly with model size.
# All models together are ~370 GB of weights; use PURGE_WEIGHTS=1 on a smaller disk.
# Finished settings are skipped and generations are cached, so the script can be re-run after a crash.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
CONFIG=configs/gec_bsweep.yaml
EXTRA=()
LOG=results/gec_bsweep
if [ "${1:-}" = "--smoke" ]; then
  shift
  EXTRA=(--name smoke_gec_bsweep --n-items 600 --batch-sizes 1 20 200 --tasks gec_taxonomy_fs)
  LOG=results/smoke_gec_bsweep
fi
EXTRA+=("$@")
mkdir -p "$LOG"
GPUS=${GPUS:-0}
MODELS=${MODELS:-$($PY -c "import yaml; print(' '.join(m['id'] for m in yaml.safe_load(open('$CONFIG'))['models']))")}

# one line per GPU: "<gpu> <model> <model> ..."
PLAN=$($PY - "$GPUS" $MODELS <<'PYEOF'
import re, sys
gpus, models = sys.argv[1].split(), sys.argv[2:]
def size(m):                      # parameter count in billions, read from the model id
    hit = re.search(r"(\d+(?:\.\d+)?)B\b", m.split("/")[-1])
    return float(hit.group(1)) if hit else 10.0
queues, load = {g: [] for g in gpus}, {g: 0.0 for g in gpus}
for m in sorted(models, key=size, reverse=True):
    g = min(gpus, key=lambda x: load[x])
    queues[g].append(m); load[g] += size(m)
for g in gpus:
    print(g, *queues[g])
PYEOF
)
echo "$PLAN" | sed 's/^/GPU /'
echo

run_queue() {                     # run_queue <gpu> <model>...
  local gpu=$1; shift
  for m in "$@"; do
    echo "===== [gpu $gpu] $m  $(date '+%F %T')"
    if ! CUDA_VISIBLE_DEVICES=$gpu $PY scripts/run.py --config "$CONFIG" --models "$m" "${EXTRA[@]}" \
           > >(sed -u "s/^/[gpu $gpu] /" | tee -a "$LOG/${m//\//_}.log") 2>&1; then
      echo "$m" >> "$LOG/failed.txt"   # go on with the next model
      echo "===== [gpu $gpu] FAILED $m"
    fi
    if [ "${PURGE_WEIGHTS:-0}" = "1" ]; then
      rm -rf "${HF_HOME:-$HOME/.cache/huggingface}/hub/models--${m//\//--}"
    fi
  done
}

rm -f "$LOG/failed.txt"
while read -r gpu models; do
  [ -n "$models" ] && run_queue "$gpu" $models &
done <<< "$PLAN"
wait
echo
if [ -s "$LOG/failed.txt" ]; then echo "FAILED:"; cat "$LOG/failed.txt"; exit 1; fi
echo "all models finished"
