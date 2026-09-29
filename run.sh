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
# Formal experiment suite: MSD and CARE, each with disjoint dev/eval patients.
# CARE: background=0, normal=1, other foreground labels=tumor (canonical 2).
# ============================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"

DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/data}"
MODEL_ROOT="${MODEL_ROOT:-$ROOT_DIR/models}"
MODELS_CONFIG="${MODELS_CONFIG:-configs/models.yaml}"
BENCHMARK_CONFIG="${BENCHMARK_CONFIG:-configs/benchmark.yaml}"
export MODELS_CONFIG BENCHMARK_CONFIG
BOOTSTRAP="${BOOTSTRAP:-2000}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/protocol_v2_5}"
MANIFEST_DIR="$RUN_ROOT/manifests"
PREDICTION_DIR="$RUN_ROOT/predictions"
RESULT_DIR="$RUN_ROOT/results"
ARTIFACT_DIR="$RUN_ROOT/artifacts"
export MANIFEST_DIR RESULT_DIR

ENABLE_CARE="${ENABLE_CARE:-1}"
export ENABLE_CARE
AUTO_PREPARE_DATA="${AUTO_PREPARE_DATA:-1}"
EVAL_ONLY="${EVAL_ONLY:-0}"
PILOT_ONLY="${PILOT_ONLY:-0}"

# Optional explicit already-extracted roots. If empty, prepare_data.py discovers
# or extracts from data/raw/.
MSD_ROOT="${MSD_ROOT:-}"
CARE_ROOT="${CARE_ROOT:-}"

# Debug-only model subset. Empty = ALL models in configs/models.yaml.
ONLY_MODELS="${ONLY_MODELS:-}"

mkdir -p "$MANIFEST_DIR" "$PREDICTION_DIR" "$RESULT_DIR" \
  "$ARTIFACT_DIR/dev/msd" "$ARTIFACT_DIR/dev/care" \
  "$ARTIFACT_DIR/eval/msd" "$ARTIFACT_DIR/eval/care" "$MODEL_ROOT" logs

echo "========================================================================"
echo "ColoGround-Bench automatic full experiment"
echo "Repository   : $ROOT_DIR"
echo "Run outputs  : $RUN_ROOT"
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
echo "===== [0/8] Preflight ====="
python scripts/bootstrap_runtime.py
python scripts/smoke_test.py
python scripts/check_gpu.py
python - <<'PY'
import torch, transformers
print("Frozen runtime:")
print("  torch       =", torch.__version__)
print("  transformers=", transformers.__version__)
print("  cuda        =", torch.version.cuda)
PY

if [[ "$EVAL_ONLY" == "1" ]]; then
  echo
  echo "===== Evaluation-only recovery mode ====="
  echo "Using existing manifests and predictions; no model will be loaded."
  RECOVERY_ARGS=()
  if [[ "$ENABLE_CARE" != "1" ]]; then
    RECOVERY_ARGS+=(--skip-care)
  fi
  python scripts/evaluate_existing.py \
    --models-config "$MODELS_CONFIG" \
    --manifest-root "$MANIFEST_DIR" \
    --predictions-root "$PREDICTION_DIR" \
    --results-root "$RESULT_DIR" \
    --bootstrap "$BOOTSTRAP" \
    "${RECOVERY_ARGS[@]}"
  python scripts/collect_results.py \
    --results-root "$RESULT_DIR" \
    --output "$RESULT_DIR/all_experiments_summary.json"
  rm -f "$RESULT_DIR/failed_experiments.txt"
  echo "Evaluation-only recovery completed."
  echo "Combined summary: $RESULT_DIR/all_experiments_summary.json"
  exit 0
fi

# ---------------------------------------------------------------------------
# 1. Data discovery / extraction
# ---------------------------------------------------------------------------
echo
echo "===== [1/8] Prepare datasets ====="

if [[ "$AUTO_PREPARE_DATA" == "1" && ( -z "$MSD_ROOT" || ( "$ENABLE_CARE" == "1" && -z "$CARE_ROOT" ) ) ]]; then
  PREPARE_ARGS=(--data-root "$DATA_ROOT" --output "$MANIFEST_DIR/data_paths.json")
  if [[ "$ENABLE_CARE" != "1" ]]; then
    PREPARE_ARGS+=(--skip-care)
  fi
  python scripts/prepare_data.py "${PREPARE_ARGS[@]}"

  if [[ -z "$MSD_ROOT" ]]; then
    MSD_ROOT="$(python - <<'PY'
import json
import os
from pathlib import Path
print(json.loads((Path(os.environ["MANIFEST_DIR"]) / "data_paths.json").read_text())["msd_root"])
PY
)"
  fi

  if [[ "$ENABLE_CARE" == "1" && -z "$CARE_ROOT" ]]; then
    CARE_ROOT="$(python - <<'PY'
import json
import os
from pathlib import Path
print(json.loads((Path(os.environ["MANIFEST_DIR"]) / "data_paths.json").read_text())["care_root"])
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
echo "===== [2/8] Build frozen experiment manifests ====="

read -r DEV_MSD DEV_CARE SPLIT_SEED PILOT_MSD_CASE PILOT_CARE_CASE < <(python - "$BENCHMARK_CONFIG" <<'PY'
import sys, yaml
with open(sys.argv[1], encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
print(cfg["splits"]["dev_msd"], cfg["splits"]["dev_care"], cfg["seed"], cfg["splits"]["pilot_case_msd"], cfg["splits"]["pilot_case_care"])
PY
)

echo "[MSD] indexing"
python scripts/index_datasets.py   --msd-root "$MSD_ROOT"   --output "$MANIFEST_DIR/cases_msd.jsonl"

python scripts/split_cases.py --cases "$MANIFEST_DIR/cases_msd.jsonl" \
  --dev-size "$DEV_MSD" --seed "$SPLIT_SEED" --required-dev-case "$PILOT_MSD_CASE" \
  --dev-output "$MANIFEST_DIR/cases_msd_dev.jsonl" \
  --eval-output "$MANIFEST_DIR/cases_msd_eval.jsonl" \
  --split-report "$MANIFEST_DIR/split_msd.json"

echo "[MSD] dev and evaluation benchmarks"
python scripts/build_benchmark.py --cases "$MANIFEST_DIR/cases_msd_dev.jsonl" --config "$BENCHMARK_CONFIG" --output "$MANIFEST_DIR/benchmark_msd_dev.jsonl" --artifact-root "$ARTIFACT_DIR/dev/msd" --experiment-name msd_dev
python scripts/build_benchmark.py --cases "$MANIFEST_DIR/cases_msd_eval.jsonl" --config "$BENCHMARK_CONFIG" --output "$MANIFEST_DIR/benchmark_msd.jsonl" --artifact-root "$ARTIFACT_DIR/eval/msd" --experiment-name msd
python scripts/audit_benchmark.py \
  --cases-dev "$MANIFEST_DIR/cases_msd_dev.jsonl" --cases-eval "$MANIFEST_DIR/cases_msd_eval.jsonl" \
  --manifest-dev "$MANIFEST_DIR/benchmark_msd_dev.jsonl" --manifest-eval "$MANIFEST_DIR/benchmark_msd.jsonl" \
  --output "$MANIFEST_DIR/coverage_msd.json"

EXPERIMENTS=("msd")

if [[ "$ENABLE_CARE" == "1" ]]; then
  echo "[CARE] background=0 normal=1 tumor=other foreground"
  python scripts/index_datasets.py --care-root "$CARE_ROOT" --care-splits test --care-index-source txt --output "$MANIFEST_DIR/cases_care.jsonl"
  python scripts/split_cases.py --cases "$MANIFEST_DIR/cases_care.jsonl" \
    --dev-size "$DEV_CARE" --seed "$SPLIT_SEED" --required-dev-case "$PILOT_CARE_CASE" \
    --dev-output "$MANIFEST_DIR/cases_care_dev.jsonl" \
    --eval-output "$MANIFEST_DIR/cases_care_eval.jsonl" \
    --split-report "$MANIFEST_DIR/split_care.json"
  python scripts/build_benchmark.py --cases "$MANIFEST_DIR/cases_care_dev.jsonl" --config "$BENCHMARK_CONFIG" --output "$MANIFEST_DIR/benchmark_care_dev.jsonl" --artifact-root "$ARTIFACT_DIR/dev/care" --experiment-name care_dev
  python scripts/build_benchmark.py --cases "$MANIFEST_DIR/cases_care_eval.jsonl" --config "$BENCHMARK_CONFIG" --output "$MANIFEST_DIR/benchmark_care.jsonl" --artifact-root "$ARTIFACT_DIR/eval/care" --experiment-name care
  python scripts/audit_benchmark.py \
    --cases-dev "$MANIFEST_DIR/cases_care_dev.jsonl" --cases-eval "$MANIFEST_DIR/cases_care_eval.jsonl" \
    --manifest-dev "$MANIFEST_DIR/benchmark_care_dev.jsonl" --manifest-eval "$MANIFEST_DIR/benchmark_care.jsonl" \
    --output "$MANIFEST_DIR/coverage_care.json"
  EXPERIMENTS+=("care")
fi

# ---------------------------------------------------------------------------
# 3. Download all model weights
# ---------------------------------------------------------------------------
echo
echo "===== [3/8] Download ALL configured model weights ====="
if [[ -n "$ONLY_MODELS" ]]; then
  read -r -a SELECTED_MODEL_KEYS <<< "$ONLY_MODELS"
  python scripts/download_models.py --config "$MODELS_CONFIG" \
    --model-root "$MODEL_ROOT" --models "${SELECTED_MODEL_KEYS[@]}"
else
  python scripts/download_models.py --config "$MODELS_CONFIG" --model-root "$MODEL_ROOT"
fi

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
# 4. Adapter smoke tests
# ---------------------------------------------------------------------------
echo
echo "===== [4/8] Real-image model adapter smoke tests ====="
SMOKE_FAILED=()
SMOKE_CARE_ARGS=()
if [[ "$ENABLE_CARE" == "1" ]]; then
  SMOKE_CARE_ARGS=(--care-manifest "$MANIFEST_DIR/benchmark_care_dev.jsonl")
fi
for model in "${MODEL_KEYS[@]}"; do
  echo
  echo "[adapter smoke] $model"
  if ! python scripts/model_adapter_smoke.py \
    --model "$model" \
    --models-config "$MODELS_CONFIG" \
    --benchmark-config "$BENCHMARK_CONFIG" \
    --model-root "$MODEL_ROOT" \
    --manifest "$MANIFEST_DIR/benchmark_msd_dev.jsonl" \
    "${SMOKE_CARE_ARGS[@]}" \
    --report "$RESULT_DIR/pilot/$model.json" \
    --strict; then
    SMOKE_FAILED+=("$model")
  fi
done

if [[ "${#SMOKE_FAILED[@]}" -gt 0 ]]; then
  echo "Adapter pilot found no valid track outputs for: ${SMOKE_FAILED[*]}" >&2
  echo "Full inference was not started." >&2
  exit 1
fi
echo "All model adapters produced at least one valid pilot output; see per-track format reports above."
if [[ "$PILOT_ONLY" == "1" ]]; then
  echo "Pilot complete. Full inference was not started."
  exit 0
fi

# ---------------------------------------------------------------------------
# 5. Inference + evaluation
# ---------------------------------------------------------------------------
echo
echo "===== [5/8] Run all models on all experiment branches ====="

FAILED=()

for model in "${MODEL_KEYS[@]}"; do
  echo
  echo "========================================================================"
  echo "MODEL: $model"
  echo "========================================================================"

  for exp in "${EXPERIMENTS[@]}"; do
    manifest="$MANIFEST_DIR/benchmark_${exp}.jsonl"
    pred_dir="$PREDICTION_DIR/$model/$exp"
    result_dir="$RESULT_DIR/$model/$exp"
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
echo "===== [6/8] Collect experiment summaries ====="
python scripts/collect_results.py   --results-root "$RESULT_DIR"   --output "$RESULT_DIR/all_experiments_summary.json"

# ---------------------------------------------------------------------------
# 7. Final status
# ---------------------------------------------------------------------------
echo
echo "===== [7/8] Final status ====="

python - <<'PY'
import datetime, hashlib, importlib.metadata, json, os, platform, subprocess
from pathlib import Path

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def git_commit():
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None

result_dir = Path(os.environ["RESULT_DIR"])
manifest_dir = Path(os.environ["MANIFEST_DIR"])
meta = {
    "timestamp": datetime.datetime.now().astimezone().isoformat(),
    "protocol": "v2.5",
    "git_commit": git_commit(),
    "host": platform.node(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "experiments": ["msd"] + (["care"] if os.environ.get("ENABLE_CARE", "1") == "1" else []),
    "care_label_mapping": {"background": 0, "normal": 1, "tumor_foreground": ">1", "source": "user_confirmed"},
    "software_versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "accelerate", "modelscope")},
    "configuration_sha256": {name: sha256(os.environ[name]) for name in ("MODELS_CONFIG", "BENCHMARK_CONFIG")},
    "manifest_sha256": {p.name: sha256(p) for p in sorted(manifest_dir.glob("benchmark_*.jsonl"))},
    "patient_splits": {p.name: json.loads(p.read_text(encoding="utf-8")) for p in sorted(manifest_dir.glob("split_*.json"))},
    "manifest_coverage": {p.name: json.loads(p.read_text(encoding="utf-8")) for p in sorted(manifest_dir.glob("coverage_*.json"))},
    "pilot_format_reports": {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted((result_dir / "pilot").glob("*.json"))},
}
result_dir.mkdir(parents=True, exist_ok=True)
(result_dir / "run_metadata.json").write_text(
    json.dumps(meta, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
PY

if [[ "${#FAILED[@]}" -gt 0 ]]; then
  printf "%s\n" "${FAILED[@]}" > "$RESULT_DIR/failed_experiments.txt"
  echo "Run completed with failures:"
  printf "  - %s\n" "${FAILED[@]}"
  echo "Successful predictions/results were preserved."
  echo "Re-run 'bash run.sh' to resume incomplete inference items."
  exit 1
fi

rm -f "$RESULT_DIR/failed_experiments.txt"

echo "========================================================================"
echo "ALL EXPERIMENTS COMPLETED"
echo "MSD results:"
echo "  $RESULT_DIR/<model>/msd/"
if [[ "$ENABLE_CARE" == "1" ]]; then
  echo "CARE (normal=1, tumor=other foreground):"
  echo "  $RESULT_DIR/<model>/care/"
fi
echo "Combined summary:"
echo "  $RESULT_DIR/all_experiments_summary.json"
echo "========================================================================"
