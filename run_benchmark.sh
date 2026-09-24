#!/usr/bin/env bash
set -euo pipefail

# ColoGround-Bench orchestration. Data and models are intentionally not committed.
# Required environment examples:
#   MSD_ROOT=/data/MSD/Task10_Colon
#   CARE_ROOT=/data/CARE
#   CARE_MAPPING=/data/CARE/care_index.csv   # optional but recommended if filenames do not encode patient/slice

MODELS_CONFIG=${MODELS_CONFIG:-configs/models.yaml}
BENCHMARK_CONFIG=${BENCHMARK_CONFIG:-configs/benchmark.yaml}
MANIFEST=${MANIFEST:-manifests/benchmark_v1.jsonl}
NUM_GPUS=${NUM_GPUS:-4}

mkdir -p manifests predictions results models artifacts

python scripts/smoke_test.py
python scripts/index_datasets.py \
  ${MSD_ROOT:+--msd-root "$MSD_ROOT"} \
  ${CARE_ROOT:+--care-root "$CARE_ROOT"} \
  ${CARE_MAPPING:+--care-mapping "$CARE_MAPPING"} \
  --output manifests/cases.jsonl

python scripts/build_benchmark.py --cases manifests/cases.jsonl --config "$BENCHMARK_CONFIG" --output "$MANIFEST"

if [[ "${DOWNLOAD_MODELS:-1}" == "1" ]]; then
  python scripts/download_models.py --config "$MODELS_CONFIG" --model-root models
fi

mapfile -t MODEL_KEYS < <(python - <<'PY'
import yaml
with open('configs/models.yaml','r',encoding='utf-8') as f:
    print('\n'.join(yaml.safe_load(f)['models'].keys()))
PY
)

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
