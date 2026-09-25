#!/usr/bin/env bash
set -euo pipefail

# ColoGround-Bench full automatic runner.
# Slurm is intentionally NOT invoked here: no sbatch/srun/salloc commands.
# Allocate/submit the job yourself, then run: bash run.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"

MSD_ROOT="${MSD_ROOT:-$ROOT_DIR/data/MSD}"
CARE_ROOT="${CARE_ROOT:-$ROOT_DIR/data/extracted/CARE}"
CARE_SPLITS="${CARE_SPLITS:-test}"
CARE_INDEX_SOURCE="${CARE_INDEX_SOURCE:-txt}"
CARE_TUMOR_LABEL="${CARE_TUMOR_LABEL:-}"
CARE_NORMAL_LABEL="${CARE_NORMAL_LABEL:-}"
ENABLE_CARE="${ENABLE_CARE:-1}"

MODELS_CONFIG="${MODELS_CONFIG:-configs/models.yaml}"
BENCHMARK_CONFIG="${BENCHMARK_CONFIG:-configs/benchmark.yaml}"
MODEL_ROOT="${MODEL_ROOT:-$ROOT_DIR/models}"
CASES_MANIFEST="${CASES_MANIFEST:-manifests/cases.jsonl}"
BENCHMARK_MANIFEST="${BENCHMARK_MANIFEST:-manifests/benchmark_v1.jsonl}"
BOOTSTRAP="${BOOTSTRAP:-2000}"

# Debug-only override. Empty means ALL configured models.
ONLY_MODELS="${ONLY_MODELS:-}"

mkdir -p manifests predictions results artifacts "$MODEL_ROOT" logs

echo "======================================================================"
echo "ColoGround-Bench automatic runner"
echo "Repository : $ROOT_DIR"
echo "Date       : $(date -Iseconds)"
echo "Host       : $(hostname)"
echo "Python     : $(command -v python)"
echo "Slurm job  : ${SLURM_JOB_ID:-not detected}"
echo "CUDA vis.  : ${CUDA_VISIBLE_DEVICES:-not set}"
echo "======================================================================"

python scripts/smoke_test.py
python scripts/check_gpu.py

if [[ ! -d "$MSD_ROOT" ]]; then
  echo "ERROR: MSD_ROOT does not exist: $MSD_ROOT" >&2
  exit 2
fi

INDEX_ARGS=(--msd-root "$MSD_ROOT" --output "$CASES_MANIFEST")

if [[ "$ENABLE_CARE" == "1" ]]; then
  if [[ ! -d "$CARE_ROOT" ]]; then
    echo "ERROR: CARE enabled but CARE_ROOT does not exist: $CARE_ROOT" >&2
    echo "For temporary MSD-only debugging: ENABLE_CARE=0 bash run.sh" >&2
    exit 2
  fi
  if [[ -z "$CARE_TUMOR_LABEL" || -z "$CARE_NORMAL_LABEL" ]]; then
    echo "ERROR: CARE label semantics are not configured." >&2
    echo "Set CARE_TUMOR_LABEL and CARE_NORMAL_LABEL after semantic verification." >&2
    exit 2
  fi
  INDEX_ARGS+=(
    --care-root "$CARE_ROOT"
    --care-splits $CARE_SPLITS
    --care-index-source "$CARE_INDEX_SOURCE"
    --care-tumor-label "$CARE_TUMOR_LABEL"
    --care-normal-label "$CARE_NORMAL_LABEL"
  )
fi

echo
echo "===== [1/6] Index datasets ====="
python scripts/index_datasets.py "${INDEX_ARGS[@]}"

echo
echo "===== [2/6] Build benchmark ====="
python scripts/build_benchmark.py \
  --cases "$CASES_MANIFEST" \
  --config "$BENCHMARK_CONFIG" \
  --output "$BENCHMARK_MANIFEST"

echo
echo "===== [3/6] Download ALL configured model weights ====="
# No --models argument: download every entry in configs/models.yaml.
python scripts/download_models.py \
  --config "$MODELS_CONFIG" \
  --model-root "$MODEL_ROOT"

if [[ -n "$ONLY_MODELS" ]]; then
  echo "WARNING: ONLY_MODELS is a debugging override; this is not the full protocol."
  read -r -a MODEL_KEYS <<< "$ONLY_MODELS"
else
  mapfile -t MODEL_KEYS < <(python - "$MODELS_CONFIG" <<'PY'
import sys, yaml
with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
for key in cfg["models"]:
    print(key)
PY
)
fi

if [[ "${#MODEL_KEYS[@]}" -eq 0 ]]; then
  echo "ERROR: no models configured." >&2
  exit 2
fi

echo "Models to run: ${MODEL_KEYS[*]}"
echo "Single-GPU sequential execution. Slurm is not invoked by this script."

echo
echo "===== [4/6] Run inference ====="
FAILED_MODELS=()

for model in "${MODEL_KEYS[@]}"; do
  echo
  echo "----------------------------------------------------------------------"
  echo "MODEL: $model"
  echo "----------------------------------------------------------------------"

  pred_dir="predictions/$model"
  result_dir="results/$model"
  pred_file="$pred_dir/all.jsonl"
  mkdir -p "$pred_dir" "$result_dir"

  if ! python scripts/run_inference.py \
      --model "$model" \
      --models-config "$MODELS_CONFIG" \
      --benchmark-config "$BENCHMARK_CONFIG" \
      --manifest "$BENCHMARK_MANIFEST" \
      --model-root "$MODEL_ROOT" \
      --output "$pred_file" \
      --shard-index 0 \
      --num-shards 1; then
    echo "ERROR: inference failed for $model" >&2
    FAILED_MODELS+=("$model:inference")
    continue
  fi

  echo "===== [5/6] Evaluate $model ====="
  if ! python scripts/evaluate.py \
      --manifest "$BENCHMARK_MANIFEST" \
      --predictions "$pred_file" \
      --output-dir "$result_dir" \
      --bootstrap "$BOOTSTRAP"; then
    echo "ERROR: evaluation failed for $model" >&2
    FAILED_MODELS+=("$model:evaluation")
    continue
  fi

  echo "Completed: $model"
done

echo
echo "===== [6/6] Final status ====="
if [[ "${#FAILED_MODELS[@]}" -gt 0 ]]; then
  printf "%s\n" "${FAILED_MODELS[@]}" > results/failed_models.txt
  echo "Failures:"
  printf "  - %s\n" "${FAILED_MODELS[@]}"
  echo "Successful outputs are retained; re-running bash run.sh resumes predictions."
  exit 1
fi

rm -f results/failed_models.txt
echo "======================================================================"
echo "Completed successfully."
echo "Predictions : $ROOT_DIR/predictions/"
echo "Results     : $ROOT_DIR/results/"
echo "Manifest    : $ROOT_DIR/$BENCHMARK_MANIFEST"
echo "======================================================================"
