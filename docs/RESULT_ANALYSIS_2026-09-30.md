# v2.5 benchmark assessment and supplementary design

## Scope and decision

The project remains a benchmark of open VLMs on colorectal CT. Its contribution is a reproducible measurement protocol and evidence about capabilities and failure modes. Model scores need not exceed simple baselines. v2.5 supports protocol-specific conclusions but does not yet establish reliable volumetric understanding, a comprehensive visual-capability ranking, or T5 faithfulness.

Freeze the v2.5 primary protocol and preserve its outputs. Add diagnostics and a development-only controlled format experiment instead of repeatedly changing the evaluation task to seek higher scores. All new diagnostics described below are post-hoc and should be identified as such in a publication. The evaluation cohort has already been examined and is not an independent prospective validation set.

## Verified experiment

- Job 156201 used commit `da434426ab8c9dbc74ea8a4c1dbf05d11661c353` on an H100.
- Six models × two datasets completed. All 14,514 predictions have unique, complete IDs and matching manifest fingerprints.
- MSD has 106 evaluation patients and 1,882 items; CARE has 61 evaluation patients and 537 items. Development/evaluation patients are disjoint.
- T3 uses 211 MSD and 119 CARE items. Boundary counts at C–G are 36/53/37/41/44 and 21/23/24/27/24 respectively.
- Marginal boundary dispersion does not remove all priors: CARE entry boundaries tend to occur earlier and exit boundaries later, increasing positive-slice prevalence. The audit now records side-specific positions and positive-slice counts.
- Non-T3 raw answers are identical to v2.4 for all models. These reruns demonstrate repeatability, not independent evidence.

## T3 diagnostics computed from existing predictions

Balanced accuracy is the mean of sensitivity and specificity within a window, then averaged within patient and across patients. Malformed outputs count as zero. A constant all-positive answer has balanced accuracy 0.5, even when its F1 is high. The table gives patient means and does not constitute an aggregate model ranking.

| Model / baseline | MSD F1 | MSD balanced accuracy | CARE F1 | CARE balanced accuracy |
|---|---:|---:|---:|---:|
| All nine slices | 0.675 | 0.500 | 0.791 | 0.500 |
| Development position prior | 0.614 | 0.570 | 0.759 | 0.595 |
| Qwen3.5 | 0.000 | 0.000 | 0.000 | 0.000 |
| Qwen3.6 | 0.649 | 0.504 | 0.797 | 0.539 |
| GLM-4.6V-Flash | 0.477 | 0.522 | 0.676 | 0.583 |
| InternVL3.5 | 0.244 | 0.220 | 0.132 | 0.102 |
| MedGemma | 0.672 | 0.500 | 0.789 | 0.500 |
| Gemma 4 | 0.580 | 0.535 | 0.577 | 0.510 |

The position prior sees only development labels, with equal patient weighting and a fixed prevalence threshold of 0.5. It always selects C–H on MSD and C–I on CARE, without seeing evaluation images or boundary side. Evaluation labels do not fit or select the prior.

No model has a mean balanced accuracy above this prior. For CARE, GLM minus prior is −0.012, exploratory paired 95% CI [−0.083, 0.061]; Qwen3.6 minus prior is −0.056 [−0.088, −0.022]. Intervals use 2,000 patient bootstrap resamples and have no multiplicity correction. They should not be used for significance-based model selection.

MedGemma selects all nine slices in 209/211 MSD questions and 118/119 CARE questions. Its high F1 therefore provides little evidence of selective localization. InternVL's patient-average invalid rate is 59.9% on MSD and 81.1% on CARE; its primary score includes this output-contract failure. The v2.5 revision changed both image windows and the prompt's JSON example, so cross-version differences cannot isolate either cause.

## Development-only format experiment

Use all available T3 development images, with identical images, labels, decoding settings, and token budget across three variants: current prompt, current prompt plus a format-only A/C example, and current prompt plus a format-only G/I example. Report every variant's format validity and F1, plus paired patient differences. The two example positions help reveal answer-position effects from the example itself.

The CLI rejects evaluation or mixed-split input, saves raw predictions and model/configuration provenance, validates resumable fingerprints, and never chooses a winner. Results must remain supplementary; applying a chosen prompt to the reused evaluation cohort would require an explicitly documented protocol revision. GPU results are pending; preparing the question set is not evidence that format compliance improved.

Run `bash run_supplement.sh` for offline diagnostics, or `FORMAT_ABLATION=1 bash run_supplement.sh` inside an allocated GPU job for the additional development experiment. Outputs are in `runs/protocol_v2_5/supplement/` and do not overwrite primary results.

## Publication boundaries

Report T1/T2 evidence with random/spatial baselines and coverage; T4 with side bias and swap consistency; T5 as exploratory with original-recognition coverage and its very small CARE sample (three patients). T3 F1, balanced accuracy, exact-set match, invalid rate, and position baselines should be presented together. The added metrics do not establish a new primary endpoint retrospectively. Further independent validation requires new patients; repeated evaluation on these same patients cannot supply that independence.
