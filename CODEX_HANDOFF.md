# CRCBenchmark current handoff (2026-10-02)

This replaces the historical v2.3 handoff. Read PROTOCOL.md and docs/PUBLICATION_GAP_REVIEW_2026-10-02.md for current science and execution details.

- ColoGround-Bench is a benchmark; no task-specific VLM fine-tuning, synthetic clinical labels, or weighted overall leaderboard.
- Primary protocol v2.5 is frozen. Existing v2.3/v2.4 results are historical; do not run the former two CARE semantic branches.
- CARE mapping is user-confirmed: 0 background, 1 normal tissue, all positive labels >1 tumor (canonical 2). Publisher numeric codebook and clinical validity of new rendered questions remain unverified.
- Patient split: MSD 20 development / 106 evaluation; CARE released test 20 development / 61 evaluation. Track coverage varies (CARE T1 evaluation 26 patients, development 8).
- Six configured open VLMs, greedy generation, 96 new tokens, sequential execution on allocated H20/H100. ModelScope source; no automatic Slurm submissions.
- Primary inference is complete. Do not repeatedly run bash run.sh to improve scores or change the parser after seeing evaluation responses.
- Format sensitivity experiment: 1,368 development responses; complete. MedGemma example-GI copies labels on 76/76 images. InternVL valid formatting can coexist with all-slice selection.
- Patient-block association supplement: 36 tests, 9,999 permutations; all Holm p=1. Descriptive evidence does not establish that models ignore images or prove clinical efficacy.
- New publication evidence controls: original / text_only / neutral_image on development T1-T4, all six models, 6,048 predictions. Code and offline preparation verified; real GPU runs are pending.
- Human feasibility tools prepare 78 blinded questions. The user currently has no participating doctors. Do not claim human scores or expert validation. Synthetic software-check files are clearly labeled and are not study results.
- Exact weight checksums/revisions still need archive; snapshot configuration hashes are not weight identity.

## Commands

```bash
# Existing offline analysis
ASSOCIATION_ANALYSIS=1 bash run_supplement.sh
# New supplement preparation (no models/GPU)
bash run_publication_supplement.sh
# New paired image-availability experiment inside an allocated GPU job
EVIDENCE_CONTROL=1 bash run_publication_supplement.sh
```

New outputs: runs/protocol_v2_5/supplement/publication/. Changing runtime, model configuration, relocated paths or inputs requires a new OUTPUT_ROOT. ARTIFACT_ROOT explicitly remaps an artifacts directory.

Next analyze complete paired condition scores, patient differences and invalid rates. Continue quality review, novelty comparison, version/data-card and licensing documentation. Independent new cohorts are needed for independent generalization claims, not a universal requirement for every descriptive benchmark paper. T5 stays exploratory and gated by original PRESENT recognition.

Preserve user-synced logs, artifacts, manifests and predictions. Stage only intentional code/document changes. The user authorizes GitHub synchronization; remote origin is the existing CRCBenchmark repository.
