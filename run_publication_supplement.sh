#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/protocol_v2_5}"
MODEL_ROOT="${MODEL_ROOT:-$ROOT_DIR/models}"
MODELS_CONFIG="${MODELS_CONFIG:-configs/models.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$RUN_ROOT/supplement/publication}"
ARTIFACT_ARGS=()
if [[ -n "${ARTIFACT_ROOT:-}" ]]; then ARTIFACT_ARGS=(--artifact-root "$ARTIFACT_ROOT"); fi

python scripts/prepare_human_review.py --run-root "$RUN_ROOT" \
  --output-dir "$OUTPUT_ROOT/human_review" "${ARTIFACT_ARGS[@]}"

MODE_ARGS=(--prepare-only)
if [[ "${EVIDENCE_CONTROL:-0}" == "1" ]]; then MODE_ARGS=(); fi
mapfile -t MODEL_KEYS < <(python - "$MODELS_CONFIG" <<'PY'
import sys, yaml
with open(sys.argv[1], encoding="utf-8") as stream:
    print("\n".join(yaml.safe_load(stream)["models"]))
PY
)
for model in "${MODEL_KEYS[@]}"; do
  for dataset in msd care; do
    python scripts/run_evidence_control.py \
      --manifest "$RUN_ROOT/manifests/benchmark_${dataset}_dev.jsonl" \
      --evaluation-manifest "$RUN_ROOT/manifests/benchmark_${dataset}.jsonl" \
      --model "$model" --models-config "$MODELS_CONFIG" --model-root "$MODEL_ROOT" \
      --output-dir "$OUTPUT_ROOT/evidence/$model/$dataset" \
      "${ARTIFACT_ARGS[@]}" "${MODE_ARGS[@]}"
  done
done
