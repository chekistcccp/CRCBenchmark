# ColoGround-Bench Protocol v1.3

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

### Current release audit findings

The first read-only audit of the downloaded CARE archive found a `DataV6` package with 26,656 train NPZ files and 6,461 test NPZ files. Every NPZ filename was conservatively parseable as `case_id + slice_index`, yielding 318 train case IDs and 81 test case IDs. Sampled images were 512 × 512 with values in [0,1], and sampled raw masks contained values 0,1,2,3.

The official U-SAM CARE dataloader canonicalizes the released masks with `mask[mask > 2] = 2`; CRCBenchmark now mirrors this preprocessing rule, so raw value 3 is merged into canonical class 2 before benchmark construction. The medical meaning of canonical classes 1 and 2 is still kept explicit and must be verified before semantic CARE tracks are frozen.

The archive-level sequence audit showed that not every retained patient series is globally contiguous. This is compatible with the CARE paper's statement that slices not containing the rectum were removed. Therefore CARE T3 no longer requires an entire patient sequence to be contiguous; only local windows whose original source slice indices are consecutive are eligible.

The observed train NPZ count differs from the paper-reported 26,563 train pairs, while the test count matches the reported 6,461. CRCBenchmark therefore uses the official bbox CSV membership as the default CARE inclusion list and treats extra unreferenced NPZ files as audit items rather than silently adding them to the benchmark.

A v2 audit must still be reviewed before benchmark freeze to determine train/test case overlap, exact CSV↔NPZ set differences, and the content of small metadata files included in the archive.


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
python scripts/inspect_care.py data/raw/CARE/CARE.zip --output manifests/care_audit.json
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

The benchmark is designed for open-weight VLMs runnable on the target 4 × RTX 3090 system. Model choice can be updated independently of the frozen dataset protocol; the benchmark, not a specific Qwen version, is the primary research object.
