# ColoGround-Bench Protocol v2.5

ColoGround-Bench is a training-free benchmark for open vision-language models on colorectal CT. It uses only real expert segmentation annotations from MSD Task10 Colon and CARE; no T stage, pathology, MSI, prognosis, necrosis, or synthetic clinical labels are created.

## Tracks

1. **T1 Lesion Retrieval** — rank tumor-bearing slices among same-patient hard negatives. Primary metric: true positive-slice Recall@3; Hit@3 is reported separately.
2. **T2 Visual Grounding** — output a normalized point and bounding box on tumor-positive slices. Primary metric: Pointing Accuracy.
3. **T3 Volumetric Consistency** — identify tumor-positive slices around entry/exit boundaries. Primary metric: patient-level Slice F1.
4. **T4 CARE Hard Negative** — same-patient tumor ROI versus normal rectal wall ROI. Report pairwise accuracy, valid-pair coverage, both-correct rate, swap consistency, and side preference.
5. **T5 Exploratory Counterfactual Response** — compare original images, lesion-specific suppression, matched-control suppression and dose-response perturbations. The original PRESENT recognition rate is the coverage gate. Continuous Faithfulness Gap remains unavailable until candidate likelihoods are validated; decision-only scores are reported separately for recognized originals.

### v2.5 T3 construction and score interpretation

For each eligible tumor entry and exit, the benchmark builds nine truly consecutive source slices. The true boundary is placed at one of positions C–G using a fixed SHA-256-derived permutation per patient and side; the first feasible position is selected. This selection uses only patient identity, source-slice availability, and the frozen seed. It never uses model predictions. The prompt does not disclose the boundary side or position. The evaluation manifest records the chosen `boundary_slot`, and a pre-inference audit rejects a formal cohort with only one boundary position.

Patient-level Slice F1 remains the primary T3 metric. Selecting all nine slices and selecting the fixed center slice are separate baselines. Boundary error is descriptive: a fixed-center strategy can do well on boundary location without identifying the complete tumor-positive interval. The old v2.4 T3 placed every boundary at E and is retained as a historical pilot result; v2.5 T3 scores are not directly comparable to it.

The v2.4 evaluation results were inspected before this structural correction. v2.5 reuses the same patient cohort, so it is a revised benchmark on reused data, not an independent prospective confirmation. Model outputs did not determine boundary slots, task thresholds, or output-parser changes. Future external validation requires a new patient cohort.

For T1–T4, the primary score counts malformed answers as failures and reports their rate. A secondary `conditional_*` score uses only valid-format answers, with its own patient count. A valid `["NONE"]` on a tumor-positive T1/T3 question is a wrong abstention, not a malformed answer. The conditional score must be read alongside coverage and must not replace the primary score. Output parsing remains frozen and does not infer answers from free-form clinical explanations after viewing evaluation responses.

## Dataset-specific roles

### Frozen v2.5 and supplementary diagnostic design (2026-09-30)

v2.5 primary manifests, prompts, parsing, and scores remain frozen. These additions are post-hoc diagnostics motivated by the observed v2.5 results; they are not new primary endpoints or independent validation. Low scores and format failures remain reportable benchmark findings.

- T3 additionally reports per-window balanced accuracy (mean sensitivity and specificity) and exact positive-set match, aggregated first within patient. Both classes must occur in a window. All-positive and all-negative constant decisions have balanced accuracy 0.5; malformed responses score zero and retain their invalid flag. This diagnostic penalizes selecting every slice despite high positive prevalence.
- Three nonvisual baselines are reported: all slices, fixed E, and a development-only position prior. For the prior, positive-label prevalence is averaged within each development patient and then across patients; labels with prevalence at least 0.5 are always selected for every evaluation image in that dataset. No evaluation labels or model outputs fit the prior, and the prior cannot access entry/exit side. It remains a post-hoc added baseline.
- Differences from each baseline use paired patient bootstrap (2,000 resamples, seed 42, percentile 95% intervals). They are exploratory intervals without multiplicity correction; no significance-based model selection or overall leaderboard is performed.
- Coverage audits report boundary position jointly with entry/exit side and report positive-slice counts. Marginal position balance alone does not imply balanced class prevalence or removal of every position prior.
- A separate GPU format experiment uses every available T3 development image from each dataset. All models receive three paired variants: the frozen current prompt, that prompt plus a format-only `["A","C"]` example, and that prompt plus a format-only `["G","I"]` example. Images, labels, decoding configuration, and token budget are identical across variants. Both example positions are reported to expose example-induced position effects. Every variant is reported; the script never selects a winning prompt or applies it to evaluation patients.
- The format experiment rejects any input row not marked `benchmark_split=dev`, records raw answers, fingerprints, model configuration, and paired patient differences in format validity and F1. Formal evaluation rejects these development manifests. GPU execution is optional and separate from the offline supplement.

Run `bash run_supplement.sh` to analyze existing full v2.5 results without model inference. Inside an allocated GPU environment, `FORMAT_ABLATION=1 bash run_supplement.sh` additionally runs the development experiment for all configured models and both datasets. All outputs go under `runs/protocol_v2_5/supplement/`. This workflow does not rebuild primary images or overwrite primary results. The input must include both dataset manifests and existing predictions.

### Development response/label association control (2026-10-02)

This is an exploratory post-hoc diagnostic of T3 measurement validity, motivated by the completed format experiment. It does not revise v2.5, select prompts, or establish independent validation. It uses existing development responses only; identical image/prompt pairs need no new inference for this statistical control.

- Include every configured model, both datasets, and all three format variants. Require complete unique predictions, exact reconstruction of the format manifest from the development manifest, disjoint development/evaluation patient IDs, manifest fingerprints, and matching model configuration metadata. Record prediction-file SHA-256 hashes. Missing model results are an error, not a silently smaller comparison family.
- Within each dataset and variant, require identical prompts. Group patients by available boundary sides (entry only, exit only, or both). Uniformly permute entire patient response blocks within each group, matching entry to entry and exit to exit. This preserves within-patient dependence, side-specific answer tendencies, invalid responses, and the cohort label distribution. Do not select donors by ground truth or response quality. Full permutations include fixed points. Singleton groups are excluded from both matched and permuted scores and explicitly reported.
- Average window scores within patient and then across eligible patients. Malformed answers remain zero-scored failures. Compare observed balanced accuracy with 9,999 permutations using seed 20261002 and the same permutation schedule across models/variants. The one-sided Monte Carlo p-value is `(1 + count(null >= observed)) / 10000`; ties count against evidence. Apply Holm correction across all model/dataset/variant BA tests (36 in this roster). F1 and exact-set differences are descriptive endpoints without additional significance tests.
- Report matched means, permutation means, their differences, and null 2.5/97.5 percentiles. Those percentiles describe the permutation distribution and are **not confidence intervals**. The null assumes response-block exchangeability across patients within each side-pattern group; it is not a causal intervention on images. Positive association would still need independent replication and confound checks. Failure to reject does not prove that a model ignores images, especially with 20 development patients per dataset.
- The primary claim remains benchmark performance and failure modes. Do not use this diagnostic to claim clinical efficacy, rank overall model quality, or replace the frozen F1 endpoint with a favorable alternative. An independent patient cohort is required before prospective generalization claims. T5 remains exploratory and gated by original-image recognition.

After completing the format experiment, run `ASSOCIATION_ANALYSIS=1 bash run_supplement.sh`, or run `python scripts/analyze_t3_association.py` alone. No GPU is needed. Output: `runs/protocol_v2_5/supplement/t3_association.json`.

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

The packaged images are 512 × 512 with values in [0,1]. Raw sampled labels include values 0,1,2,3. The user-confirmed interpretation for this protocol is `0=background`, `1=normal tissue`, and all other positive labels `=tumor`. Labels greater than 1 are mapped to canonical class 2, consistent with the official U-SAM collapse of values greater than 2. This mapping is user supplied; the released numeric codebook has not been independently verified from publisher documentation.

The CARE release used by the public U-SAM loader is treated conservatively as preprocessed 2D NPZ image-label pairs until the actual downloaded archive has been audited.

The benchmark must not assume any of the following without evidence:

- that CARE NPZ pixel values are original HU;
- that numeric NPZ filenames correspond to patient or slice order;
- that adjacent files form a contiguous 3D volume.

### Patient/slice mapping

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

For the currently audited release, filename parsing provides a strong patient/slice mapping candidate. CARE T3 uses only locally consecutive source-slice windows; gaps elsewhere in the same retained patient series do not invalidate an otherwise consecutive local window. Individual slices are never treated as independent patients in primary statistical inference.

## CARE image intensity handling

MSD uses HU windowing.

CARE packaged NPZ images are used as provided. If arrays are in `[0,1]`, they are directly mapped to display intensity. If they are preprocessed but not normalized, robust display scaling is used. HU windowing is not applied to CARE unless original HU semantics are independently demonstrated.

## Read-only CARE audit

Before extraction, a downloaded archive can be inspected using:

```bash
python scripts/inspect_care.py data/raw/CARE/CARE.zip --output manifests/care_audit_v3.json
```

The audit reads archive members, bbox CSV files and a deterministic sample of NPZ files without extracting the full archive. Its historical `label_semantics_verified=false` flag means the archive itself did not resolve the codebook; protocol v2.5 uses the mapping supplied by the user.

## Non-negotiable evaluation rules

- Ground-truth masks are never rendered as model inputs.
- No task-specific VLM training or fine-tuning.
- No LLM judge.
- No chain-of-thought scoring.
- Main generation is greedy (`do_sample=False`).
- All primary statistics are patient-level.
- All models run the same frozen benchmark manifest.
- Development and formal-evaluation patients are disjoint. The output-format pilot uses development patients only.
- MSD `colon_001` and CARE `case17105001`, previously used for v2.3 adapter pilots, are forced into development cohorts and excluded from formal evaluation.
- Invalid responses are retained as failures and separately counted.
- Primary scores include invalid responses; conditional valid-format scores are secondary and always include their denominator.
- T1/T2/T3 include pre-specified random, spatial, and trivial-selection baselines respectively.
- No aggregate weighted leaderboard score is created.
- CARE uses the user-confirmed label mapping and records its provenance.
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

Protocol v2.5 writes new manifests, artifacts, predictions, and results under `runs/protocol_v2_5/` by default, preserving v2.3 and v2.4. A stable SHA-256 patient split assigns 20 MSD and 20 CARE cases to development and the remainder to formal evaluation. Prediction resume and formal evaluation require matching manifest SHA-256 fingerprints. Before model inference, the runner saves `coverage_msd.json` and `coverage_care.json` with track-level patient counts and T3 boundary positions. A GPU output-format pilot can be run with `PILOT_ONLY=1 bash run.sh`; it uses development patients only and exits before full inference.

The repository is intentionally scheduler-agnostic. `run.sh` never calls `sbatch`, `srun`, `salloc`, or any other Slurm command. GPU/node allocation and job submission are performed manually by the user or institutional scheduler configuration.

The primary hardware protocol is one NVIDIA H20 or H100 GPU with BF16 inference. All configured models are executed sequentially in separate Python processes on the GPU allocated by the surrounding Slurm job.

Before inference, `run.sh` automatically downloads every model listed in `configs/models.yaml` through ModelScope. Existing complete snapshots are reused; incomplete sharded checkpoints are detected and resumed/re-fetched. The official full experiment does not require separate manual model-download commands.

Inference outputs are checkpointed incrementally at item level. Re-running `run.sh` after preemption reuses completed predictions and continues unfinished items. A debugging-only `ONLY_MODELS` override is supported, but the primary benchmark protocol runs all configured models.


## CARE label protocol

The 81-patient CARE test cohort is split at patient level into 20 development and 61 formal-evaluation patients. Only one mapping is run: `0=background`, `1=normal`, `raw labels >1=tumor`, canonically stored as 2. The former inverted-label branch remains in the archived v2.3 outputs and is not part of v2.5. This mapping was provided by the user; a publication should state that provenance rather than claim that the released documentation independently proves it.


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
