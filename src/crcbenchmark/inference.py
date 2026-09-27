from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path
from .io import read_jsonl, write_jsonl
from .utils import extract_first_json
from .utils import parse_choice_response


def _normalize_choice(raw: str, choices: list[str]) -> str | None:
    return parse_choice_response(raw, choices)


def manifest_fingerprint(rows: list[dict]) -> str:
    payload = json.dumps(rows, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_resume_predictions(existing: list[dict], fingerprint: str, output_path: Path) -> None:
    mismatched = [row.get("item_id") for row in existing if row.get("manifest_sha256") != fingerprint]
    if mismatched:
        raise RuntimeError(
            f"{output_path}: {len(mismatched)} existing predictions have no matching "
            "manifest fingerprint. Use a fresh prediction output path for the rebuilt "
            "benchmark; do not reuse answers tied to different image order or labels."
        )


def run_manifest(model, rows, out_path, shard_index=0, num_shards=1, resume=True):
    out_path = Path(out_path)
    fingerprint = manifest_fingerprint(rows)
    old_rows = read_jsonl(out_path) if resume and out_path.exists() else []
    validate_resume_predictions(old_rows, fingerprint, out_path)
    existing = {r["item_id"]: r for r in old_rows}
    selected = [r for i, r in enumerate(rows) if i % num_shards == shard_index]
    outputs = list(existing.values()); done = set(existing)
    for item in selected:
        if item["item_id"] in done: continue
        t0=time.perf_counter(); raw=model.generate(item["image_path"],item["prompt"]); latency=time.perf_counter()-t0
        parsed=extract_first_json(raw); choice=scores=score_mode=None
        if item.get("choices"):
            choice=_normalize_choice(raw,item["choices"])
            scores=model.score_choices(item["image_path"],item["prompt"],item["choices"])
            score_mode="likelihood" if scores is not None else "decision"
            if scores is None and choice is not None: scores={c:float(c==choice) for c in item["choices"]}
        outputs.append({"item_id":item["item_id"],"track":item["track"],"dataset":item["dataset"],"case_id":item["case_id"],"raw_response":raw,"parsed":parsed,"choice":choice,"choice_scores":scores,"score_mode":score_mode,"latency_s":latency,"manifest_sha256":fingerprint})
        write_jsonl(out_path,outputs)
