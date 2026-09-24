from __future__ import annotations
import time
from pathlib import Path
from .io import read_jsonl, write_jsonl
from .utils import extract_first_json


def _normalize_choice(raw: str, choices: list[str]) -> str | None:
    s = raw.strip().upper()
    for c in choices:
        if s == c.upper() or s.startswith(c.upper() + "\n") or s.startswith(c.upper() + "."):
            return c
    tokens = [x.strip(' \t\n\r.,:;"\'[]{}()').upper() for x in s.split()]
    for c in choices:
        if c.upper() in tokens:
            return c
    return None


def run_manifest(model, rows, out_path, shard_index=0, num_shards=1, resume=True):
    out_path = Path(out_path)
    existing = {r["item_id"]: r for r in read_jsonl(out_path)} if resume and out_path.exists() else {}
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
        outputs.append({"item_id":item["item_id"],"track":item["track"],"dataset":item["dataset"],"case_id":item["case_id"],"raw_response":raw,"parsed":parsed,"choice":choice,"choice_scores":scores,"score_mode":score_mode,"latency_s":latency})
        write_jsonl(out_path,outputs)
