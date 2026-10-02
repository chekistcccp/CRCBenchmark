# ColoGround-Bench / CRCBenchmark

[中文](README.md)

A zero-shot benchmark for open vision-language models on colorectal CT. Its five tracks test tumor-slice retrieval (T1), point and box grounding (T2), consecutive-slice recognition (T3), tumor-versus-normal region discrimination (T4), and exploratory response to lesion perturbation (T5). Tasks use real segmentation masks and do not invent clinical labels.

See [PROTOCOL.md](PROTOCOL.md) for the full v2.5 specification. The previous instructions are preserved in the [v2.3 archive](docs/README_EN_V2_3_ARCHIVE.md).

## Protocol v2.5

- **MSD Task10 Colon** provides 3D CT and tumor masks for T1/T2/T3/T5.
- **CARE** uses the released 81-patient test cohort. The user-confirmed mapping is `0=background`, `1=normal tissue`, and every other foreground label `=tumor`; raw labels greater than 1 are mapped to canonical tumor class 2. This mapping has not been independently verified from the release's numeric codebook.
- A deterministic, patient-disjoint split reserves 20 MSD and 20 CARE patients for development. Formal evaluation uses the remaining 106 MSD and 61 CARE patients. The adapter pilot uses development patients only.
- MSD `colon_001` and CARE `case17105001`, used in v2.3 adapter pilots, are forced into development and excluded from formal evaluation.
- Formal evaluation checks that all prediction IDs are unique, complete, and tied to the exact evaluation manifest fingerprint.
- T3 places the true boundary at a seeded position C–G within nine consecutive slices, removing the fixed-E shortcut in v2.4. A pre-inference manifest audit checks patient isolation, task coverage, and boundary positions.
- T1/T2/T3 report pre-specified simple baselines. T4 reports accuracy, valid-pair coverage, both-correct rate, swap consistency, and side preference. Primary scores count malformed responses as failures; scores restricted to valid responses are secondary. T5 reports original-image recognition coverage first; decision-only perturbation results are exploratory until candidate likelihoods have been validated.
- Outputs default to `runs/protocol_v2_5/`, preserving v2.3 and v2.4 results. T3 scores from v2.4 and v2.5 use different questions and are not directly comparable.

## Running on an allocated GPU

The repository does not submit Slurm jobs. Inside an allocated single-H20 or single-H100 GPU job:

```bash
cd CRCBenchmark
conda activate crcbench
PILOT_ONLY=1 bash run.sh   # optional development-patient format check
bash run.sh                # full benchmark; resumes completed items
```

Put raw archives under `data/raw/MSD/` and `data/raw/CARE/CARE.zip`, or set `MSD_ROOT` and `CARE_ROOT` to extracted directories. The runner prepares data, builds manifests, downloads the configured models, performs inference, and evaluates all tracks.

Development-patient pilot checks are saved under `results/pilot/` by model. A pilot passes when the adapter produces at least one valid-format answer; formal scores still count every malformed answer and report valid-format-only scores separately.

Use Python 3.11, PyTorch 2.13.0, torchvision 0.28.0, and CUDA 12.6. Install the PyTorch cu126 wheels before `pip install -r requirements.txt`.

With existing full v2.5 results, run `bash run_supplement.sh` for offline T3 balanced accuracy, exact-set match, development-fitted position priors, and paired patient differences. On an allocated GPU, `FORMAT_ABLATION=1 bash run_supplement.sh` also runs all three prompt variants on identical development images for every model and both datasets. All variants are reported; no winner is selected. Outputs stay under `runs/protocol_v2_5/supplement/`. These are post-hoc supplementary diagnostics; the v2.5 primary protocol remains frozen.

After the format experiment is complete, run `ASSOCIATION_ANALYSIS=1 bash run_supplement.sh` for offline patient-block response/label association tests. Responses are permuted within boundary-side availability groups, retaining within-patient dependence. The report includes matched versus permuted balanced accuracy, 9,999 permutations, and Holm correction across all 36 model/dataset/variant tests. This is an exploratory development analysis, not a causal image intervention; non-significance does not prove image neglect. See [the current analysis](docs/RESULT_ANALYSIS_2026-10-02.md).

For the new publication supplements, `bash run_publication_supplement.sh` prepares development-only original/text-only/neutral-image controls and a blinded 78-question human feasibility packet. `EVIDENCE_CONTROL=1 bash run_publication_supplement.sh` runs all six models on an allocated GPU (6,048 predictions). Outputs are separate in `supplement/publication/`; raw image/mask hashes and runtime/model metadata protect resumption. The input change includes format and distribution effects and does not prove causal lesion grounding. No doctors have participated yet. See [the six-paper literature comparison and next experiments](docs/PUBLICATION_GAP_REVIEW_2026-10-02.md).

Report patient-level intervals, invalid-response rates, task baselines, and task coverage alongside model scores. Low scores are valid benchmark findings; however, output-format failures must not be misrepresented as visual incompetence. T5 continuous Faithfulness Gap remains unavailable without validated likelihood scores.
