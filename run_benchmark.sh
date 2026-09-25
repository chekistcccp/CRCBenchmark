#!/usr/bin/env bash
set -euo pipefail

# ColoGround-Bench orchestration. Data and models are intentionally not committed.
# Required environment examples:
#   MSD_ROOT=/data/MSD/Task10_Colon
#   CARE_ROOT=/data/CARE
#   CARE_MAPPING=/data/CARE/care_index.csv      # required if filenames do not prove patient/slice order
#   CARE_TUMOR_LABEL=...                        # set only after official CARE label semantics are verified
#   CARE_NORMAL_LABEL=...                       # set only after official CARE label semantics are verified
#   CARE_INDEX_SOURCE=txt|bbox_txt|bbox_csv|all_npz  # primary default: txt
#   CARE_SPLITS="test"                           # primary default: CARE test only
#
# IMPORTANT: before enabling CARE, run:
#   CARE_SOURCE=/path/to/CARE.zip bash run_data_audit.sh
# or audit the extracted directory. No CARE label semantics are assumed by default.

MODELS_CONFIG=${MODELS_CONFIG:-configs/models.yaml}
BENCHMARK_CONFIG=${BENCHMARK_CONFIG:-configs/benchmark.yaml}
MANIFEST=${MANIFEST:-manifests/benchmark_v1.jsonl}
NUM_GPUS=${NUM_GPUS:-1}

mkdir -p manifests predictions results models artifacts

python scripts/smoke_test.py
python scripts/check_gpu.py

INDEX_ARGS=(--output manifests/cases.jsonl)
if [[ -n "${MSD_ROOT:-}" ]]; then
  INDEX_ARGS+=(--msd-root "$MSD_ROOT")
fi
if [[ -n "${CARE_ROOT:-}" ]]; then
  if [[ -z "${CARE_TUMOR_LABEL:-}" ]]; then
    echo "ERROR: CARE_ROOT is set but CARE_TUMOR_LABEL is not." >&2
    echo "Run scripts/inspect_care.py first and verify official label semantics; do not guess label IDs." >&2
    exit 2
  fi
  INDEX_ARGS+=(--care-root "$CARE_ROOT" --care-tumor-label "$CARE_TUMOR_LABEL" --care-index-source "${CARE_INDEX_SOURCE:-txt}" --care-splits ${CARE_SPLITS:-test})
  if [[ -n "${CARE_NORMAL_LABEL:-}" ]]; then
    INDEX_ARGS+=(--care-normal-label "$CARE_NORMAL_LABEL")
  fi
  if [[ -n "${CARE_MAPPING:-}" ]]; then
    INDEX_ARGS+=(--care-mapping "$CARE_MAPPING")
  fi
fi

python scripts/index_datasets.py "${INDEX_ARGS[@]}"
python scripts/build_benchmark.py --cases manifests/cases.jsonl --config "$BENCHMARK_CONFIG" --output "$MANIFEST"

if [[ "${DOWNLOAD_MODELS:-1}" == "1" ]]; then
  python scripts/download_models.py --config "$MODELS_CONFIG" --model-root models
fi

if [[ -n "${RUN_MODELS:-}" ]]; then
  read -r -a MODEL_KEYS <<< "$RUN_MODELS"
else
  mapfile -t MODEL_KEYS < <(python - <<'PY'
import yaml
with open('configs/models.yaml','r',encoding='utf-8') as f:
    print('\n'.join(yaml.safe_load(f)['models'].keys()))
PY
)
fi

echo "GPU workers: $NUM_GPUS"
echo "Models: ${MODEL_KEYS[*]}"

for model in "${MODEL_KEYS[@]}"; do
  echo "=== $model ==="
  mkdir -p "predictions/$model"
  pids=()
  for ((g=0; g<NUM_GPUS; g++)); do
    CUDA_VISIBLE_DEVICES=$g python scripts/run_inference.py \
      --model "$model" --models-config "$MODELS_CONFIG" --benchmark-config "$BENCHMARK_CONFIG" \
      --manifest "$MANIFEST" --shard-index "$g" --num-shards "$NUM_GPUS" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  python scripts/merge_shards.py --dir "predictions/$model" --output "predictions/$model/all.jsonl"
  python scripts/evaluate.py --manifest "$MANIFEST" --predictions "predictions/$model/all.jsonl" --output-dir "results/$model"
done

echo "Benchmark complete. See results/."
