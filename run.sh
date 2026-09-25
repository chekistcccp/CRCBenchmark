#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# ColoGround-Bench full automatic experiment runner
#
# Scheduler policy:
#   - NO sbatch / srun / salloc calls.
#   - Submit/allocate the Slurm job manually.
#   - Inside the allocated job, run:
#         bash run.sh
#
# Primary compute:
#   - 1 x NVIDIA H20 or H100
#   - BF16
#   - non-quantized
#
# Formal experiment suite:
#   1) MSD once
#   2) CARE branch A: canonical class 1 = tumor, class 2 = normal
#   3) CARE branch B: canonical class 2 = tumor, class 1 = normal
#
# IMPORTANT:
#   The two CARE branches are semantic-sensitivity branches. Do NOT decide the
#   true label meaning by selecting the branch with better model performance.
# ============================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"

DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/data}"
MODEL_ROOT="${MODEL_ROOT:-$ROOT_DIR/models}"
MODELS_CONFIG="${MODELS_CONFIG:-configs/models.yaml}"
BENCHMARK_CONFIG="${BENCHMARK_CONFIG:-configs/benchmark.yaml}"
BOOTSTRAP="${BOOTSTRAP:-2000}"

ENABLE_CARE="${ENABLE_CARE:-1}"
AUTO_PREPARE_DATA="${AUTO_PREPARE_DATA:-1}"

# Optional explicit already-extracted roots. If empty, prepare_data.py discovers
# or extracts from data/raw/.
MSD_ROOT="${MSD_ROOT:-}"
CARE_ROOT="${CARE_ROOT:-}"

# Debug-only model subset. Empty = ALL models in configs/models.yaml.
ONLY_MODELS="${ONLY_MODELS:-}"

mkdir -p   manifests   predictions   results   artifacts/msd   artifacts/care_tumor1_normal2   artifacts/care_tumor2_normal1   "$MODEL_ROOT"   logs

echo "========================================================================"
echo "ColoGround-Bench automatic full experiment"
echo "Repository   : $ROOT_DIR"
echo "Date         : $(date -Iseconds)"
echo "Host         : $(hostname)"
echo "Python       : $(command -v python)"
echo "Slurm job ID : ${SLURM_JOB_ID:-not detected}"
echo "CUDA visible : ${CUDA_VISIBLE_DEVICES:-not set}"
echo "========================================================================"

# ---------------------------------------------------------------------------
# 0. Preflight
# ---------------------------------------------------------------------------
echo
echo "===== [0/7] Preflight ====="
python scripts/smoke_test.py
python scripts/check_gpu.py

# ---------------------------------------------------------------------------
# 1. Data discovery / extraction
# ---------------------------------------------------------------------------
echo
echo "===== [1/7] Prepare datasets ====="

if [[ "$AUTO_PREPARE_DATA" == "1" && ( -z "$MSD_ROOT" || ( "$ENABLE_CARE" == "1" && -z "$CARE_ROOT" ) ) ]]; then
  PREPARE_ARGS=(--data-root "$DATA_ROOT" --output manifests/data_paths.json)
  if [[ "$ENABLE_CARE" != "1" ]]; then
    PREPARE_ARGS+=(--skip-care)
  fi
  python scripts/prepare_data.py "${PREPARE_ARGS[@]}"

  if [[ -z "$MSD_ROOT" ]]; then
    MSD_ROOT="$(python - <<'PY'
import json
print(json.load(open("manifests/data_paths.json"))["msd_root"])
PY
)"
  fi

  if [[ "$ENABLE_CARE" == "1" && -z "$CARE_ROOT" ]]; then
    CARE_ROOT="$(python - <<'PY'
import json
print(json.load(open("manifests/data_paths.json"))["care_root"])
PY
)"
  fi
fi

if [[ -z "$MSD_ROOT" || ! -d "$MSD_ROOT" ]]; then
  echo "ERROR: valid MSD_ROOT not found." >&2
  echo "Put the MSD archive under data/raw/MSD/ or set MSD_ROOT explicitly." >&2
  exit 2
fi

if [[ "$ENABLE_CARE" == "1" ]]; then
  if [[ -z "$CARE_ROOT" || ! -d "$CARE_ROOT" ]]; then
    echo "ERROR: valid CARE_ROOT not found." >&2
    echo "Put CARE.zip under data/raw/CARE/ or set CARE_ROOT explicitly." >&2
    exit 2
  fi
fi

echo "MSD_ROOT : $MSD_ROOT"
if [[ "$ENABLE_CARE" == "1" ]]; then
  echo "CARE_ROOT: $CARE_ROOT"
fi

# ---------------------------------------------------------------------------
# 2. Build experiment manifests
# ---------------------------------------------------------------------------
echo
echo "===== [2/7] Build frozen experiment manifests ====="

echo "[MSD] indexing"
python scripts/index_datasets.py   --msd-root "$MSD_ROOT"   --output manifests/cases_msd.jsonl

echo "[MSD] benchmark"
python scripts/build_benchmark.py   --cases manifests/cases_msd.jsonl   --config "$BENCHMARK_CONFIG"   --output manifests/benchmark_msd.jsonl   --artifact-root artifacts/msd   --experiment-name msd

EXPERIMENTS=("msd")

if [[ "$ENABLE_CARE" == "1" ]]; then
  echo "[CARE A] tumor=1 normal=2"
  python scripts/index_datasets.py     --care-root "$CARE_ROOT"     --care-splits test     --care-index-source txt     --care-tumor-label 1     --care-normal-label 2     --output manifests/cases_care_tumor1_normal2.jsonl

  python scripts/build_benchmark.py     --cases manifests/cases_care_tumor1_normal2.jsonl     --config "$BENCHMARK_CONFIG"     --output manifests/benchmark_care_tumor1_normal2.jsonl     --artifact-root artifacts/care_tumor1_normal2     --experiment-name care_tumor1_normal2

  echo "[CARE B] tumor=2 normal=1"
  python scripts/index_datasets.py     --care-root "$CARE_ROOT"     --care-splits test     --care-index-source txt     --care-tumor-label 2     --care-normal-label 1     --output manifests/cases_care_tumor2_normal1.jsonl

  python scripts/build_benchmark.py     --cases manifests/cases_care_tumor2_normal1.jsonl     --config "$BENCHMARK_CONFIG"     --output manifests/benchmark_care_tumor2_normal1.jsonl     --artifact-root artifacts/care_tumor2_normal1     --experiment-name care_tumor2_normal1

  EXPERIMENTS+=("care_tumor1_normal2" "care_tumor2_normal1")
fi

# ---------------------------------------------------------------------------
# 3. Download all model weights
# ---------------------------------------------------------------------------
echo
echo "===== [3/7] Download ALL configured model weights ====="
python scripts/download_models.py   --config "$MODELS_CONFIG"   --model-root "$MODEL_ROOT"

if [[ -n "$ONLY_MODELS" ]]; then
  echo "WARNING: ONLY_MODELS is a debugging override; formal run uses all models."
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

echo "Models     : ${MODEL_KEYS[*]}"
echo "Experiments: ${EXPERIMENTS[*]}"

# ---------------------------------------------------------------------------
# 4-5. Inference + evaluation
# ---------------------------------------------------------------------------
echo
echo "===== [4/7] Run all models on all experiment branches ====="

FAILED=()

for model in "${MODEL_KEYS[@]}"; do
  echo
  echo "========================================================================"
  echo "MODEL: $model"
  echo "========================================================================"

  for exp in "${EXPERIMENTS[@]}"; do
    manifest="manifests/benchmark_${exp}.jsonl"
    pred_dir="predictions/$model/$exp"
    result_dir="results/$model/$exp"
    pred_file="$pred_dir/all.jsonl"

    mkdir -p "$pred_dir" "$result_dir"

    echo
    echo "---- $model / $exp : inference ----"
    if ! python scripts/run_inference.py         --model "$model"         --models-config "$MODELS_CONFIG"         --benchmark-config "$BENCHMARK_CONFIG"         --manifest "$manifest"         --model-root "$MODEL_ROOT"         --output "$pred_file"         --shard-index 0         --num-shards 1; then
      echo "ERROR: inference failed: $model / $exp" >&2
      FAILED+=("$model/$exp:inference")
      continue
    fi

    echo "---- $model / $exp : evaluation ----"
    if ! python scripts/evaluate.py         --manifest "$manifest"         --predictions "$pred_file"         --output-dir "$result_dir"         --bootstrap "$BOOTSTRAP"; then
      echo "ERROR: evaluation failed: $model / $exp" >&2
      FAILED+=("$model/$exp:evaluation")
      continue
    fi
  done
done

# ---------------------------------------------------------------------------
# 6. Collect summaries
# ---------------------------------------------------------------------------
echo
echo "===== [6/7] Collect experiment summaries ====="
python scripts/collect_results.py   --results-root results   --output results/all_experiments_summary.json

# ---------------------------------------------------------------------------
# 7. Final status
# ---------------------------------------------------------------------------
echo
echo "===== [7/7] Final status ====="

python - <<'PY'
import json, os, platform, datetime
from pathlib import Path
meta = {
    "timestamp": datetime.datetime.now().astimezone().isoformat(),
    "host": platform.node(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "experiments": [
        "msd",
        "care_tumor1_normal2",
        "care_tumor2_normal1",
    ],
    "care_note": (
        "CARE branches are semantic-sensitivity analyses. "
        "Do not infer the true label semantics from model performance."
    ),
}
Path("results").mkdir(exist_ok=True)
Path("results/run_metadata.json").write_text(
    json.dumps(meta, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
PY

if [[ "${#FAILED[@]}" -gt 0 ]]; then
  printf "%s\n" "${FAILED[@]}" > results/failed_experiments.txt
  echo "Run completed with failures:"
  printf "  - %s\n" "${FAILED[@]}"
  echo "Successful predictions/results were preserved."
  echo "Re-run 'bash run.sh' to resume incomplete inference items."
  exit 1
fi

rm -f results/failed_experiments.txt

echo "========================================================================"
echo "ALL EXPERIMENTS COMPLETED"
echo "MSD results:"
echo "  results/<model>/msd/"
if [[ "$ENABLE_CARE" == "1" ]]; then
  echo "CARE branch A (tumor=1, normal=2):"
  echo "  results/<model>/care_tumor1_normal2/"
  echo "CARE branch B (tumor=2, normal=1):"
  echo "  results/<model>/care_tumor2_normal1/"
fi
echo "Combined summary:"
echo "  results/all_experiments_summary.json"
echo "========================================================================"
