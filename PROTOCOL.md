# ColoGround-Bench Protocol v2.2

ColoGround-Bench is a training-free benchmark for open vision-language models on colorectal CT. It uses only real expert segmentation annotations from MSD Task10 Colon and CARE; no T stage, pathology, MSI, prognosis, necrosis, or synthetic clinical labels are created.

## Tracks

1. **T1 Lesion Retrieval** — rank tumor-bearing slices among same-patient hard negatives. Primary metric: Recall@3.
2. **T2 Visual Grounding** — output a normalized point and bounding box on tumor-positive slices. Primary metric: Pointing Accuracy.
3. **T3 Volumetric Consistency** — identify tumor-positive slices around entry/exit boundaries. Primary metric: patient-level Slice F1.
4. **T4 CARE Hard Negative** — same-patient tumor ROI versus normal rectal wall ROI. Primary metric: pairwise accuracy and swap consistency.
5. **T5 Counterfactual Faithfulness** — compare original images, lesion-specific suppression, matched-control suppression and dose-response perturbations. Primary continuous metric is the Faithfulness Gap when candidate likelihoods are available; a categorical flip-based equivalent is retained for custom models whose forward method does not expose stable likelihoods.

## Dataset-specific roles

### MSD Task10 Colon

MSD is treated as a true 3D CT dataset with NIfTI image/mask pairs. It can support T1, T2, T3 and T5 directly after standard QC. Foreground tumor label `1` is defined by the task segmentation mask.

### CARE public packaged release

### CARE v3 audit findings and frozen primary cohort

The released CARE archive contains 26,656 train NPZ files and 6,461 test NPZ files. All filenames are parseable as `case_id + source_slice_index`. Across all NPZ files this yields 318 train case IDs and 81 test case IDs with no overlap, i.e. 399 unique case IDs. This does not exactly reproduce the paper's reported total of 398 patients.

Audit v3 reconciled all release-provided index sources:

| Index source | Train slices | Test slices | Train cases | Test cases | Unique cases |
|---|---:|---:|---:|---:|---:|
| all NPZ | 26,656 | 6,461 | 318 | 81 | 399 |
| `txt` | 26,656 | 6,461 | 318 | 81 | 399 |
| `bbox_txt` | 26,537 | 6,424 | 318 | 81 | 399 |
| `bbox_csv` | 26,537 | 6,424 | 318 | 81 | 399 |

No full train+test source simultaneously reproduces the paper-reported 33,024 slice pairs and 398 patients. The v3 scorer therefore no longer reports a unique "best" source when all candidates tie.

For the **primary ColoGround-Bench CARE cohort**, the protocol is now frozen to:

```text
CARE split        = test
CARE index source = test.txt
patients          = 81
slices            = 6,461
```

This choice is reproducible and matches the published test-cohort size exactly. `test.txt` contains 6,461 unique entries, every entry has a corresponding NPZ, and it covers all 6,461 NPZ files in the released test directory. All 81 test cases contain at least one locally consecutive 9-slice window, so local T3 evaluation remains feasible.

The released train cohort is **not included in the primary benchmark** because its released slice/patient counts do not reconcile cleanly with the publication. Train data may be used only for development, code debugging, or supplementary sensitivity analysis. The 6,424-slice bbox-defined test subset may likewise be reported as a supplementary sensitivity cohort, but not as the primary CARE result.

The packaged images are 512 × 512 with values in [0,1]. Raw sampled labels include values 0,1,2,3. CRCBenchmark mirrors the official U-SAM preprocessing rule `mask[mask > 2] = 2`, producing canonical labels 0,1,2. The medical meaning of canonical classes 1 and 2 remains an explicit semantic gate until independently verified.

The CARE release used by the public U-SAM loader is treated conservatively as preprocessed 2D NPZ image-label pairs until the actual downloaded archive has been audited.

The benchmark must not assume any of the following without evidence:

- that CARE NPZ pixel values are original HU;
- that foreground value `1` is normal wall;
- that foreground value `2` is tumor;
- that numeric NPZ filenames correspond to patient or slice order;
- that adjacent files form a contiguous 3D volume.

CARE activation therefore follows a mandatory two-gate procedure.

### Gate 1 — label semantics

Run `scripts/inspect_care.py` and verify the label mapping against the official release documentation/examples. CARE indexing requires explicit `--care-tumor-label`; T4 additionally requires an explicit verified normal-wall label.

No default CARE foreground semantics are embedded in the code.

### Gate 2 — patient/slice mapping

T1/T3 and all patient-level CARE statistics require a proven `case_id` and `slice_index` mapping.

The mapping may come from:

1. a filename convention that is explicitly verified and parsed for every included NPZ; or
2. an external `care_index.csv` containing at least:

```text
case_id,slice_index,npz_path,split
```

Optional physical-spacing fields may be added:

```text
slice_spacing,pixel_spacing_y,pixel_spacing_x
```

If filenames do not fully prove patient identity and ordering, the indexer fails with an explicit error instead of silently fabricating 3D groups.

For the currently audited release, filename parsing provides a strong patient/slice mapping candidate. CARE T1 becomes eligible after label semantics are verified. CARE T3 uses only locally consecutive source-slice windows; gaps elsewhere in the same retained patient series do not invalidate an otherwise consecutive local window. Individual slices are never treated as independent patients in primary statistical inference.

## CARE image intensity handling

MSD uses HU windowing.

CARE packaged NPZ images are used as provided. If arrays are in `[0,1]`, they are directly mapped to display intensity. If they are preprocessed but not normalized, robust display scaling is used. HU windowing is not applied to CARE unless original HU semantics are independently demonstrated.

## Read-only CARE audit

Before extraction, a downloaded archive can be inspected using:

```bash
python scripts/inspect_care.py data/raw/CARE/CARE.zip --output manifests/care_audit_v3.json
```

The audit reads archive members, bbox CSV files and a deterministic sample of NPZ files without extracting the full archive. It reports observed structures and values but intentionally marks:

- `label_semantics_verified = false`;
- `patient_slice_mapping_verified = false`;
- `care_3d_tracks_enabled = false`.

These flags are scientific safeguards, not parser failures.

## Non-negotiable evaluation rules

- Ground-truth masks are never rendered as model inputs.
- No task-specific VLM training or fine-tuning.
- No LLM judge.
- No chain-of-thought scoring.
- Main generation is greedy (`do_sample=False`).
- All primary statistics are patient-level.
- All models run the same frozen benchmark manifest.
- Invalid responses are retained as failures and separately counted.
- No aggregate weighted leaderboard score is created.
- CARE label semantics are never guessed from numeric IDs alone.
- CARE 3D continuity is never inferred from file order alone.
- MSD and CARE results are reported separately where their representations or eligible tracks differ; they are not naively pooled into a pseudo cross-center score.

## Current model scope

The benchmark uses a **Qwen3.5-era modern VLM roster** and a single frozen runtime based on Transformers 5.17.0. The primary model set is:

1. Qwen3.5-9B — primary contemporary baseline;
2. Qwen3.6-27B — newer/larger same-family scaling point;
3. GLM-4.6V-Flash — compact general-purpose peer model;
4. InternVL3.5-8B-HF — HF-standard InternVL peer model;
5. MedGemma 1.5 4B IT — 2026 medical-specialist multimodal baseline;
6. Gemma 4 26B-A4B IT — 2026 Google multimodal MoE peer model.

Legacy models that require substantially different remote custom-code loading paths are excluded from the primary roster to avoid confounding the scientific comparison with framework-version incompatibilities.

The benchmark is designed for open-weight VLMs runnable on a single NVIDIA H20 or H100 GPU. BF16 is the default inference precision and task-specific quantization is not part of the primary protocol. Models are executed sequentially in independent Python processes. The benchmark manifest and metrics remain fixed across models.

Model choice can still be updated in future benchmark revisions, but a published experiment must report the exact model IDs, revisions, Transformers version, and ModelScope snapshot source used.


## Execution and scheduler policy

The canonical experiment entry point is:

```bash
bash run.sh
```

The repository is intentionally scheduler-agnostic. `run.sh` never calls `sbatch`, `srun`, `salloc`, or any other Slurm command. GPU/node allocation and job submission are performed manually by the user or institutional scheduler configuration.

The primary hardware protocol is one NVIDIA H20 or H100 GPU with BF16 inference. All configured models are executed sequentially in separate Python processes on the GPU allocated by the surrounding Slurm job.

Before inference, `run.sh` automatically downloads every model listed in `configs/models.yaml` through ModelScope. Existing complete snapshots are reused; incomplete sharded checkpoints are detected and resumed/re-fetched. The official full experiment does not require separate manual model-download commands.

Inference outputs are checkpointed incrementally at item level. Re-running `run.sh` after preemption reuses completed predictions and continues unfinished items. A debugging-only `ONLY_MODELS` override is supported, but the primary benchmark protocol runs all configured models.


## CARE unresolved-label sensitivity protocol

Because the released CARE data expose canonical foreground labels 1 and 2 but the available release documentation does not yet provide a sufficiently explicit numeric-to-medical-class statement, the computational protocol does not block on a single assumed mapping.

Two complete, pre-specified CARE semantic branches are constructed from the same frozen 81-patient / 6,461-slice test cohort:

1. `care_tumor1_normal2`: canonical class 1 is treated as tumor and class 2 as normal rectal tissue;
2. `care_tumor2_normal1`: canonical class 2 is treated as tumor and class 1 as normal rectal tissue.

Raw CARE label values greater than 2 are first canonicalized with the official U-SAM rule `label > 2 -> 2`. Each semantic branch is built into a separate benchmark manifest and separate artifact directory, and every configured VLM is evaluated on both branches.

The two branches are semantic sensitivity analyses, not competing clinical ground truths. The correct medical class mapping must not be selected on the basis of model accuracy, grounding, or faithfulness performance. If authoritative annotation evidence later establishes the numeric mapping, the matching branch becomes the primary CARE result and the inverted branch is retained as a label-inversion sensitivity/control analysis.

MSD is evaluated only once and is not duplicated across the two CARE branches.


## Automatic data preparation

The canonical `run.sh` accepts raw dataset archives under:

```text
data/raw/MSD/
data/raw/CARE/CARE.zip
```

Supported MSD/CARE archive formats for automatic extraction are ZIP, TAR, TAR.GZ, and TGZ. Data are extracted under `data/extracted/`, and the code recursively locates the actual MSD root containing `imagesTr/labelsTr` and CARE root containing `test/test_npz/test.txt`.

Extraction is resumable at the file level: an already extracted file with the expected uncompressed size is skipped. This is intended for preemptible/time-limited Slurm jobs.


## Frozen software runtime

The primary benchmark runtime is frozen to:

```text
Python       = 3.11
PyTorch      = 2.13.0
torchvision  = 0.28.0
PyTorch CUDA wheel index = cu126
Transformers = >=5.17,<6 (resolved at environment installation)
```

PyTorch and torchvision are installed separately using the official CUDA 12.6 wheel index:

```bash
pip install torch==2.13.0 torchvision==0.28.0 \
  --index-url https://download.pytorch.org/whl/cu126
```

The project `requirements.txt` intentionally excludes torch and torchvision to prevent the remaining Python dependencies from replacing the frozen CUDA runtime.


Only the PyTorch/CUDA binary runtime is frozen. The model-framework layer (Transformers, Accelerate, ModelScope and related pure-Python packages) uses compatible version ranges from `requirements.txt` and is not automatically mutated by `run.sh`. This avoids coupling the benchmark to one unnecessary patch release while preserving a reproducible major-version boundary for modern Qwen3.5-era multimodal models.
