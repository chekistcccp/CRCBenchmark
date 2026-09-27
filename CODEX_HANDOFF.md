# CODEX HANDOFF — CRCBenchmark / ColoGround-Bench

This file is a handoff from the current ChatGPT conversation to Codex.
Treat this file as the current project state and continue work from the repository itself.

## 1. Project

Repository: `chekistcccp/CRCBenchmark`

Working title:

**ColoGround-Bench: Benchmarking Visual Grounding, Volumetric Consistency, and Counterfactual Faithfulness of Vision-Language Models in Colorectal CT**

Goal:
Build a training-free/open-weight VLM benchmark for colorectal CT using public datasets, with the benchmark itself as the research contribution.

The benchmark evaluates whether a VLM:
1. retrieves lesion-containing slices;
2. grounds predictions to the true lesion;
3. understands local volumetric continuity;
4. distinguishes tumor from normal rectal wall in CARE;
5. changes its decision when lesion evidence is removed.

No task-specific VLM training or fine-tuning.

## 2. Datasets

### MSD Task10 Colon

Expected structure after extraction:

```text
data/extracted/MSD/.../imagesTr/
data/extracted/MSD/.../labelsTr/
```

The loader recursively finds the real MSD root containing `imagesTr/` and `labelsTr/`.

MSD provides:
- 3D NIfTI CT
- tumor segmentation
- physical z spacing

Used for T1/T2/T3/T5.

### CARE

Recommended raw archive location:

```text
data/raw/CARE/CARE.zip
```

Primary cohort is frozen to:

```text
split        = test
index source = test.txt
patients     = 81
slices       = 6,461
```

CARE packaged image data are treated as preprocessed image arrays, not raw HU.

Raw labels are canonicalized with:

```text
label > 2 -> 2
```

The medical meaning of canonical class 1 vs class 2 is unresolved, so both semantic branches are run automatically:

```text
care_tumor1_normal2
  tumor  = 1
  normal = 2

care_tumor2_normal1
  tumor  = 2
  normal = 1
```

Do NOT infer the true label semantics from whichever branch gives better model performance.

CARE has no reliable physical z spacing in the public release. Therefore:
- T3 boundary error in slices is valid;
- T3 boundary error in mm must be `None/null`.

## 3. Benchmark tracks

### T1 Lesion Retrieval
Primary metric: Recall@3

### T2 Visual Grounding
Primary metric: Pointing Accuracy

### T3 Volumetric Consistency
Primary metric: Slice F1

Also report:
- boundary error in slices
- boundary error in mm only when physical z spacing exists

### T4 CARE Hard Negative
Primary metric: Pairwise Accuracy

### T5 Counterfactual Faithfulness
Primary metric: Faithfulness Gap

No weighted overall score.

## 4. Current model panel

`configs/models.yaml` currently contains six modern Qwen3.5-era models:

```text
qwen35_9b
qwen36_27b
glm46v_flash
internvl35_8b_hf
medgemma15_4b
gemma4_26b_a4b
```

Roles:

- Qwen3.5-9B: primary contemporary baseline
- Qwen3.6-27B: newer/larger same-family scaling comparison
- GLM-4.6V-Flash: cross-family general VLM
- InternVL3.5-8B-HF: cross-family HF-standard VLM
- MedGemma 1.5 4B IT: medical-specialist VLM
- Gemma 4 26B-A4B IT: Google multimodal MoE peer

Legacy custom-code models were removed from the primary roster to avoid framework-version incompatibility.

All model weights are downloaded automatically through ModelScope into:

```text
models/
```

## 5. Runtime policy

User requirement: PyTorch is the frozen dependency; other dependencies may adapt.

Install the frozen PyTorch runtime separately:

```bash
pip install torch==2.13.0 torchvision==0.28.0 \
  --index-url https://download.pytorch.org/whl/cu126
```

Important:
- wheel version strings such as `2.13.0+cu126` are valid and should be treated as base version `2.13.0`;
- CUDA runtime must be 12.6;
- do not upgrade/downgrade torch from project scripts.

The remaining stack is flexible through `requirements.txt`, currently using version ranges such as:

```text
transformers>=5.17,<6
accelerate>=1.2
modelscope>=1.31
```

If torchaudio is present and compiled for a different CUDA version, repair torchaudio only with `--no-deps` so torch is untouched.

## 6. Main execution entrypoint

The canonical experiment entrypoint is:

```bash
bash run.sh
```

Slurm policy:
- repository contains no sbatch/srun/salloc submission scripts;
- user manually submits/allocates the Slurm job;
- inside the allocated job, run `bash run.sh`.

Typical:

```bash
conda activate crcbench
cd CRCBenchmark
bash run.sh
```

The run automatically:
1. performs runtime/GPU preflight;
2. discovers/extracts datasets;
3. builds MSD benchmark;
4. builds CARE tumor=1/normal=2;
5. builds CARE tumor=2/normal=1;
6. downloads all configured models;
7. runs one-image adapter smoke tests;
8. runs all model/experiment combinations sequentially on one GPU;
9. evaluates results;
10. collects `results/all_experiments_summary.json`.

## 7. Resume behavior

Prediction files are written to:

```text
predictions/<model>/<experiment>/all.jsonl
```

Inference is resumable by item ID.

Important recent improvement:
`scripts/run_inference.py` now checks whether the selected manifest is already fully predicted before loading the VLM. If complete, it exits immediately without reloading model weights.

## 8. Evaluation-only recovery

If inference is complete but evaluation code was fixed, do NOT rerun VLM inference.

Use:

```bash
EVAL_ONLY=1 bash run.sh
```

This calls:

```text
scripts/evaluate_existing.py
```

It:
- loads no VLM;
- checks prediction completeness;
- recomputes all model/experiment metrics;
- rebuilds `results/all_experiments_summary.json`.

## 9. Most recent bug and fix

The most recent full run completed model inference but CARE T3 evaluation failed for all models because `spacing_z_mm` is `None` for CARE.

Old buggy code:

```python
boundary_error_mm = float(err) * float(item["gt"]["spacing_z_mm"])
```

This raised:

```text
TypeError: float() argument must be a string or a real number, not 'NoneType'
```

Fixed behavior:

```python
spacing = item["gt"].get("spacing_z_mm")
boundary_error_mm = (
    float(err) * float(spacing)
    if spacing is not None
    else None
)
```

Regression tests were added for:
- T3 with missing spacing;
- T3 with valid spacing.

The uploaded run log showed:
- no `inference failed` events;
- 12 evaluation failures = 6 models × 2 CARE semantic branches;
- all 12 were the same missing-spacing T3 bug.

Therefore the immediate recommended recovery command is:

```bash
git pull
pytest -q
EVAL_ONLY=1 bash run.sh
```

Do not rerun the full expensive inference unless prediction completeness checks report missing items.

## 10. Important repository files

```text
run.sh
requirements.txt
configs/models.yaml
configs/benchmark.yaml
PROTOCOL.md
README.md

scripts/bootstrap_runtime.py
scripts/prepare_data.py
scripts/index_datasets.py
scripts/build_benchmark.py
scripts/download_models.py
scripts/model_adapter_smoke.py
scripts/run_inference.py
scripts/evaluate.py
scripts/evaluate_existing.py
scripts/collect_results.py

src/crcbenchmark/models/registry.py
src/crcbenchmark/models/pipeline_adapter.py
src/crcbenchmark/inference.py
src/crcbenchmark/evaluate.py
src/crcbenchmark/stats.py

tests/test_metrics.py
```

## 11. Current scientific constraints

Keep these invariant unless explicitly redesigning the benchmark:

- no task-specific VLM fine-tuning;
- no synthetic clinical labels;
- no pathology/MSI/T-stage claims unsupported by the public data;
- masks are for benchmark construction/evaluation, not shown as GT overlays to the VLM;
- CARE and MSD should be reported separately;
- patient is the primary independent statistical unit;
- 2,000 patient-level bootstrap resamples;
- invalid outputs remain failures;
- no LLM-as-a-Judge as a primary metric;
- no weighted overall leaderboard score;
- do not infer CARE class semantics from performance.

## 12. Immediate next task for Codex

Start by verifying the repository after pull:

```bash
pytest -q
EVAL_ONLY=1 bash run.sh
```

Then inspect:

```text
results/all_experiments_summary.json
results/failed_experiments.txt
```

If `failed_experiments.txt` does not exist and all summaries are present, proceed to:
1. validate result completeness;
2. inspect per-track/per-dataset metrics;
3. produce publication-ready summary tables;
4. check for invalid-output rates and any model-specific parser failures;
5. clean repeated Transformers generation warnings if they remain.

## 13. Working style requested by user

- Chinese communication preferred.
- Research-grade and publication-oriented.
- Prefer complete runnable code over fragments.
- Prefer one-command workflows.
- Model weights should come from ModelScope.
- Primary hardware: one H20 or H100 under Slurm.
- User manually manages Slurm submission; do not add Slurm scripts unless explicitly asked.
- Avoid changing the frozen PyTorch runtime unless the user explicitly requests it.
