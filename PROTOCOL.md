# ColoGround-Bench Protocol v1.1

ColoGround-Bench is a training-free benchmark for open vision-language models on colorectal CT. It uses only real expert segmentation annotations from MSD Task10 Colon and CARE; no T stage, pathology, MSI, prognosis, or synthetic clinical labels are created.

## Tracks

1. **T1 Lesion Retrieval** — rank tumor-bearing slices among same-patient hard negatives. Primary metric: Recall@3.
2. **T2 Visual Grounding** — output a normalized point and bounding box on tumor-positive slices. Primary metric: Pointing Accuracy.
3. **T3 Volumetric Consistency** — identify tumor-positive slices around entry/exit boundaries. Primary metric: patient-level Slice F1.
4. **T4 CARE Hard Negative** — same-patient tumor ROI versus normal rectal wall ROI. Primary metric: pairwise accuracy and swap consistency.
5. **T5 Counterfactual Faithfulness** — compare original images, lesion-specific suppression, matched-control suppression and dose-response perturbations. Primary continuous metric is the Faithfulness Gap when candidate likelihoods are available; a categorical flip-based equivalent is retained for custom models whose forward method does not expose stable likelihoods.

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

## Data notes

MSD is expected as `imagesTr/*.nii.gz` and `labelsTr/*.nii.gz`.

The public CARE training code uses `train/train_npz`, `test/test_npz`, and bbox CSV files. The repository attempts to infer `case_id` and `slice_index` from filenames ending in `_<slice>` or `-<slice>`. If the downloaded release uses another naming scheme, provide a CSV with:

```text
case_id,slice_index,npz_path,split,slice_spacing,pixel_spacing_y,pixel_spacing_x
```

T3 is skipped for CARE cases when patient-level slice ordering cannot be proven contiguous. This prevents fabricating 3D continuity from shuffled/filtered 2D pairs.

## Models

All checkpoints are downloaded **only via ModelScope** with `modelscope.snapshot_download`, then loaded from local paths by Transformers:

- `Qwen/Qwen3.5-9B`
- `OpenGVLab/InternVL3-8B`
- `OpenBMB/MiniCPM-V-4_5`
- `Qwen/Qwen2.5-VL-7B-Instruct`
- `google/medgemma-4b-it`
- `lingshu-medical-mllm/Lingshu-7B`

Qwen3.8-27B is intentionally excluded.
