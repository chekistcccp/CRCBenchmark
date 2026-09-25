# CRCBenchmark / ColoGround-Bench

**ColoGround-Bench** is a training-free benchmark for evaluating whether open vision-language models truly **detect, localize, track, discriminate, and rely on** colorectal tumor evidence in CT.

The benchmark uses two public datasets with different roles:

- **MSD Task10 Colon** — full 3D NIfTI CT volumes with primary colon-tumor masks; used for volumetric retrieval, grounding, boundary consistency, and counterfactual testing.
- **CARE** — the public packaged release used by U-SAM is handled conservatively as preprocessed 2D `image`/`label` NPZ pairs until its patient/slice mapping and label semantics are verified from the downloaded release.

The project intentionally does **not** create synthetic T stage, pathology, MSI, prognosis, necrosis, or other unavailable clinical labels.

## Five benchmark tracks

| Track | Question | Dataset availability | Primary metric |
|---|---|---|---|
| T1 Lesion Retrieval | Can the VLM find tumor-bearing slices among same-patient hard negatives? | MSD; CARE only if patient/slice order is verified | Recall@3 |
| T2 Visual Grounding | Can it point to the actual tumor and draw a useful box? | MSD + CARE after CARE label semantics are verified | Pointing Accuracy |
| T3 Volumetric Consistency | Can it track tumor appearance/disappearance across consecutive slices? | MSD; CARE only if contiguous order is proven | Slice F1 |
| T4 CARE Hard Negative | Can it distinguish tumor from normal rectal wall? | CARE only after label semantics are verified | Pairwise Accuracy |
| T5 Counterfactual Faithfulness | Does removing the true lesion change the model more than removing matched control tissue? | MSD + CARE after CARE label semantics are verified | Faithfulness Gap |

## Hardware target

Primary target: **4 × NVIDIA RTX 3090 24 GB**. No task-specific VLM training or fine-tuning is performed.

## Installation

```bash
conda create -n crcbench python=3.11 -y
conda activate crcbench
pip install -r requirements.txt
export PYTHONPATH=$PWD/src:$PYTHONPATH
```

## Data layout

### MSD Task10 Colon

```text
data/MSD/
├── imagesTr/
│   ├── colon_001.nii.gz
│   └── ...
└── labelsTr/
    ├── colon_001.nii.gz
    └── ...
```

### CARE: audit first, do not guess

Place the downloaded archive without modifying its contents:

```text
data/raw/CARE/CARE.zip
```

The public U-SAM loader expects a packaged layout broadly equivalent to:

```text
CARE/
├── train/
│   ├── train_bbox.csv
│   └── train_npz/*.npz
└── test/
    ├── test_bbox.csv
    └── test_npz/*.npz
```

Each sampled NPZ is expected to contain at least `image` and `label`, but **this repository does not assume what label IDs 1/2 mean until the downloaded release is inspected and the semantics are verified**.

Likewise, CARE is **not assumed to be a recoverable 3D volume**. T1/T3 are enabled for CARE only when patient identity and slice ordering can be proven from filenames or an explicit mapping CSV.

## Step 1 — read-only CARE audit before extraction

After downloading `CARE.zip`:

```bash
CARE_SOURCE=data/raw/CARE/CARE.zip bash run_data_audit.sh
```

or directly:

```bash
python scripts/inspect_care.py data/raw/CARE/CARE.zip \
  --sample-n 20 \
  --output manifests/care_audit.json
```

The audit reads the ZIP **without extracting the full dataset** and reports:

- detected directory structure;
- train/test NPZ counts;
- `train_bbox.csv` / `test_bbox.csv` preview;
- sampled NPZ keys;
- `image.shape`, dtype, min/max;
- `label.shape` and observed label values;
- how many filenames can be conservatively parsed as `case_id + slice_index`;
- inferred case counts and contiguous-order diagnostics;
- safety status for CARE T1/T3 and label semantics.

The audit deliberately keeps these scientific gates closed:

```text
label_semantics_verified: false
patient_slice_mapping_verified: false
care_3d_tracks_enabled: false
```

Seeing label values such as `[0, 1, 2]` does **not** by itself establish which foreground label is tumor or normal wall.

## Step 2 — optional extraction and second audit

```bash
mkdir -p data/extracted/CARE
unzip data/raw/CARE/CARE.zip -d data/extracted/CARE

python scripts/inspect_care.py data/extracted/CARE \
  --sample-n 50 \
  --output manifests/care_audit_extracted.json
```

The code automatically searches through an extra wrapper directory such as `CARE/` if the archive contains one.

## Step 3 — decide whether CARE patient-level reconstruction is possible

### Case A: filenames fully prove patient + slice order

If the audit shows 100% conservative filename parsing and sensible multi-slice patient groups, the filename convention can be considered a candidate mapping, but it should still be manually checked before freezing the benchmark.

### Case B: filenames do not prove patient + slice order

Do **not** infer it from numeric file order. Obtain or construct an explicit mapping:

```csv
case_id,slice_index,npz_path,split,slice_spacing,pixel_spacing_y,pixel_spacing_x
patient001,0,train/train_npz/xxx.npz,train,1.25,0.75,0.75
```

Without a verified mapping, CARE T1/T3 remain disabled and slice-level samples must not be treated as independent patients for publication statistics.

## Step 4 — index datasets only after CARE semantics are verified

MSD can be indexed immediately:

```bash
python scripts/index_datasets.py \
  --msd-root data/MSD \
  --output manifests/cases.jsonl
```

CARE requires explicit verified label IDs. No default is accepted:

```bash
python scripts/index_datasets.py \
  --msd-root data/MSD \
  --care-root data/extracted/CARE \
  --care-mapping data/extracted/CARE/care_index.csv \
  --care-tumor-label <VERIFIED_ID> \
  --care-normal-label <VERIFIED_ID> \
  --output manifests/cases.jsonl
```

If filenames themselves are verified to encode patient/slice order, `--care-mapping` can be omitted.

## Important CARE intensity rule

MSD is original CT and uses HU windowing. The public CARE NPZ images are treated as **preprocessed packaged images** unless original HU semantics are independently verified.

Therefore the code does **not** blindly apply `WL=50 / WW=400` to CARE arrays. Normalized CARE arrays are mapped directly to 8-bit images; non-normalized packaged arrays use robust display scaling rather than pretending they are Hounsfield Units.

## Build and run benchmark

After indexing:

```bash
python scripts/build_benchmark.py \
  --cases manifests/cases.jsonl \
  --config configs/benchmark.yaml \
  --output manifests/benchmark_v1.jsonl
```

Full run:

```bash
export MSD_ROOT=/absolute/path/to/MSD

# Enable these only AFTER the CARE audit and semantic verification:
# export CARE_ROOT=/absolute/path/to/CARE
# export CARE_MAPPING=/absolute/path/to/care_index.csv
# export CARE_TUMOR_LABEL=<VERIFIED_ID>
# export CARE_NORMAL_LABEL=<VERIFIED_ID>

bash run_benchmark.sh
```

## Reproducibility and safety rules

- Ground-truth masks are never overlaid on model inputs.
- The benchmark manifest is frozen before model comparison.
- Greedy deterministic decoding is used (`do_sample=False`).
- Invalid outputs are retained as failures; they are not manually repaired.
- No LLM-as-a-Judge is used.
- No chain-of-thought quality score is used.
- Patient is the statistical sampling unit.
- CARE slices are never promoted to patient-level samples unless patient identity is verified.
- CARE label IDs are never assigned medical meaning by assumption.
- CARE packaged image values are never assumed to be HU without evidence.
- No arbitrary weighted overall leaderboard score is computed.

See [PROTOCOL.md](PROTOCOL.md) for the scientific design boundaries.
