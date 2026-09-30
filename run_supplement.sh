#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/protocol_v2_5}"
MODELS_CONFIG="${MODELS_CONFIG:-configs/models.yaml}"
MODEL_ROOT="${MODEL_ROOT:-$ROOT_DIR/models}"

# Offline diagnostics read existing predictions and write only supplement/.
python scripts/analyze_benchmark_supplement.py --run-root "$RUN_ROOT"

# Explicit optional GPU stage: identical development images, all prompt variants.
if [[ "${FORMAT_ABLATION:-0}" == "1" ]]; then
  mapfile -t MODEL_KEYS < <(python - "$MODELS_CONFIG" <<'PY'
import sys, yaml
with open(sys.argv[1], encoding="utf-8") as f:
    print("\n".join(yaml.safe_load(f)["models"]))
PY
)
  for model in "${MODEL_KEYS[@]}"; do
    for dataset in msd care; do
      python scripts/run_t3_format_ablation.py \
        --manifest "$RUN_ROOT/manifests/benchmark_${dataset}_dev.jsonl" \
        --model "$model" --models-config "$MODELS_CONFIG" --model-root "$MODEL_ROOT" \
        --output-dir "$RUN_ROOT/supplement/format/$model/$dataset"
    done
  done
fi
