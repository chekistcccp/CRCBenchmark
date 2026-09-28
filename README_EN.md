# ColoGround-Bench / CRCBenchmark

[中文](README.md)

A zero-shot benchmark for open vision-language models on colorectal CT. Its five tracks test tumor-slice retrieval (T1), point and box grounding (T2), consecutive-slice recognition (T3), tumor-versus-normal region discrimination (T4), and exploratory response to lesion perturbation (T5). Tasks use real segmentation masks and do not invent clinical labels.

See [PROTOCOL.md](PROTOCOL.md) for the full v2.4 specification. The previous instructions are preserved in the [v2.3 archive](docs/README_EN_V2_3_ARCHIVE.md).

## Protocol v2.4

- **MSD Task10 Colon** provides 3D CT and tumor masks for T1/T2/T3/T5.
- **CARE** uses the released 81-patient test cohort. The user-confirmed mapping is `0=background`, `1=normal tissue`, and every other foreground label `=tumor`; raw labels greater than 1 are mapped to canonical tumor class 2. This mapping has not been independently verified from the release's numeric codebook.
- A deterministic, patient-disjoint split reserves 20 MSD and 20 CARE patients for development. Formal evaluation uses the remaining 106 MSD and 61 CARE patients. The adapter pilot uses development patients only.
- MSD `colon_001` and CARE `case17105001`, used in v2.3 adapter pilots, are forced into development and excluded from formal evaluation.
- Formal evaluation checks that all prediction IDs are unique, complete, and tied to the exact evaluation manifest fingerprint.
- T1/T2/T3 report pre-specified simple baselines. T4 reports accuracy, valid-pair coverage, both-correct rate, swap consistency, and side preference. T5 reports original-image recognition coverage first; decision-only perturbation results are exploratory until candidate likelihoods have been validated.
- Outputs default to `runs/protocol_v2_4/`, preserving v2.3 results.

## Running on an allocated GPU

The repository does not submit Slurm jobs. Inside an allocated single-H20 or single-H100 GPU job:

```bash
cd CRCBenchmark
conda activate crcbench
PILOT_ONLY=1 bash run.sh   # optional development-patient format check
bash run.sh                # full benchmark; resumes completed items
```

Put raw archives under `data/raw/MSD/` and `data/raw/CARE/CARE.zip`, or set `MSD_ROOT` and `CARE_ROOT` to extracted directories. The runner prepares data, builds manifests, downloads the configured models, performs inference, and evaluates all tracks.

Use Python 3.11, PyTorch 2.13.0, torchvision 0.28.0, and CUDA 12.6. Install the PyTorch cu126 wheels before `pip install -r requirements.txt`.

Report patient-level intervals, invalid-response rates, task baselines, and task coverage alongside model scores. Low scores are valid benchmark findings; however, output-format failures must not be misrepresented as visual incompetence. T5 continuous Faithfulness Gap remains unavailable without validated likelihood scores.
