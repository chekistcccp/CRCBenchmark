# CRCBenchmark / ColoGround-Bench

**ColoGround-Bench** is a training-free benchmark for evaluating whether open vision-language models truly **detect, localize, track, discriminate, and rely on** colorectal tumor evidence in CT.

The benchmark uses two public datasets:

- **MSD Task10 Colon** — primary colon tumor segmentation masks.
- **CARE** — rectal cancer CT with pixel-level labels for normal rectal wall and rectal tumor.

The project intentionally does **not** create synthetic T stage, pathology, MSI, prognosis, or other unavailable clinical labels.

## Five benchmark tracks

| Track | Question | Primary metric |
|---|---|---|
| T1 Lesion Retrieval | Can the VLM find tumor-bearing slices among same-patient hard negatives? | Recall@3 |
| T2 Visual Grounding | Can it point to the actual tumor and draw a useful box? | Pointing Accuracy |
| T3 Volumetric Consistency | Can it track tumor appearance/disappearance across consecutive slices? | Slice F1 |
| T4 CARE Hard Negative | Can it distinguish tumor from normal rectal wall in the same patient? | Pairwise Accuracy |
| T5 Counterfactual Faithfulness | Does removing the true lesion change the model more than removing matched control tissue? | Faithfulness Gap |

## Tested open VLMs

All weights are downloaded **only from ModelScope** using `modelscope.snapshot_download()` and are then loaded from the local snapshot path.

- `Qwen/Qwen3.5-9B`
- `OpenGVLab/InternVL3-8B`
- `OpenBMB/MiniCPM-V-4_5`
- `Qwen/Qwen2.5-VL-7B-Instruct`
- `google/medgemma-4b-it`
- `lingshu-medical-mllm/Lingshu-7B`

**Qwen3.8-27B is not used.**

## Hardware target

Primary target: **4 × NVIDIA RTX 3090 24 GB**.

The default launcher runs four independent patient shards in data parallel. No task-specific training or fine-tuning is performed.

## Installation

Recommended Linux + CUDA environment:

```bash
conda create -n crcbench python=3.11 -y
conda activate crcbench
pip install -r requirements.txt
export PYTHONPATH=$PWD/src:$PYTHONPATH
```

If a ModelScope model requires license acceptance or authentication, complete that on ModelScope before running `download_models.py`.

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

### CARE

The official U-SAM loader expects the public packaged form:

```text
data/CARE/
├── train/
│   ├── train_bbox.csv
│   └── train_npz/*.npz
└── test/
    ├── test_bbox.csv
    └── test_npz/*.npz
```

Each NPZ should contain `image` and `label` arrays. CARE labels are interpreted as:

- `1`: normal rectal wall
- `2`: rectal tumor

For T1/T3 patient-level volumetric tasks, the benchmark must know `case_id` and `slice_index`. If filenames do not encode these values, provide an explicit mapping CSV:

```csv
case_id,slice_index,npz_path,split,slice_spacing,pixel_spacing_y,pixel_spacing_x
patient001,0,train/train_npz/xxx.npz,train,1.25,0.75,0.75
```

T3 is automatically skipped when consecutive ordering cannot be established.

## Quick start

### 1. Index datasets

```bash
python scripts/index_datasets.py \
  --msd-root data/MSD \
  --care-root data/CARE \
  --output manifests/cases.jsonl
```

With explicit CARE mapping:

```bash
python scripts/index_datasets.py \
  --msd-root data/MSD \
  --care-root data/CARE \
  --care-mapping data/CARE/care_index.csv \
  --output manifests/cases.jsonl
```

### 2. Build the frozen benchmark artifacts

```bash
python scripts/build_benchmark.py \
  --cases manifests/cases.jsonl \
  --config configs/benchmark.yaml \
  --output manifests/benchmark_v1.jsonl
```

### 3. Download models from ModelScope

```bash
python scripts/download_models.py --model-root models
```

Or download one model:

```bash
python scripts/download_models.py --models qwen35_9b --model-root models
```

### 4. Run one model on one GPU

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_inference.py \
  --model qwen35_9b \
  --manifest manifests/benchmark_v1.jsonl
```

### 5. Four-GPU sharded inference

```bash
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g python scripts/run_inference.py \
    --model qwen35_9b \
    --manifest manifests/benchmark_v1.jsonl \
    --shard-index $g --num-shards 4 &
done
wait

python scripts/merge_shards.py \
  --dir predictions/qwen35_9b \
  --output predictions/qwen35_9b/all.jsonl
```

### 6. Evaluate

```bash
python scripts/evaluate.py \
  --manifest manifests/benchmark_v1.jsonl \
  --predictions predictions/qwen35_9b/all.jsonl \
  --output-dir results/qwen35_9b
```

### 7. Full benchmark

Set dataset paths and run:

```bash
export MSD_ROOT=/absolute/path/to/MSD
export CARE_ROOT=/absolute/path/to/CARE
# optional:
# export CARE_MAPPING=/absolute/path/to/care_index.csv

bash run_benchmark.sh
```

## Reproducibility rules

- Ground-truth masks are never overlaid on model inputs.
- The benchmark manifest is frozen before model comparison.
- Greedy deterministic decoding is used (`do_sample=False`).
- Invalid outputs are retained as failures; they are not manually repaired.
- No LLM-as-a-Judge is used.
- No chain-of-thought quality score is used.
- Patient is the statistical sampling unit.
- The main summary uses 2,000 patient-level bootstrap resamples.
- No arbitrary weighted overall leaderboard score is computed.

## Counterfactual scoring

T5 asks `PRESENT` vs `ABSENT` on:

1. original slice,
2. true-lesion suppression,
3. matched-control suppression,
4. a second perturbation operator,
5. graded lesion suppression.

When the model exposes a standard conditional forward pass, the code computes candidate sequence likelihoods. For models whose custom remote code does not expose a stable likelihood interface, the exact same experiment is evaluated using categorical decision flips and `score_mode=decision` is recorded. This avoids inventing pseudo-probabilities from free-text confidence statements.

## Protocol

See [PROTOCOL.md](PROTOCOL.md) for the frozen scientific protocol and design boundaries.
