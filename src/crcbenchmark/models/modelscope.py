from __future__ import annotations

import json
from pathlib import Path

from modelscope import snapshot_download


def _indexed_weights_complete(local_dir: Path) -> bool:
    for index_name in (
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ):
        index_path = local_dir / index_name
        if not index_path.exists():
            continue
        try:
            obj = json.loads(index_path.read_text(encoding="utf-8"))
            files = sorted(set(obj.get("weight_map", {}).values()))
            return bool(files) and all((local_dir / f).exists() for f in files)
        except Exception:
            return False
    return False


def _snapshot_looks_complete(local_dir: Path) -> bool:
    if not (local_dir / "config.json").exists():
        return False

    if _indexed_weights_complete(local_dir):
        return True

    # Non-sharded checkpoints.
    direct_weights = list(local_dir.glob("*.safetensors")) + list(local_dir.glob("pytorch_model*.bin"))
    if direct_weights:
        return all(p.is_file() and p.stat().st_size > 0 for p in direct_weights)

    return False


def ensure_model(
    model_id: str,
    model_root: str | Path,
    revision: str | None = None,
) -> Path:
    model_root = Path(model_root)
    local_dir = model_root / model_id.replace("/", "__")
    local_dir.mkdir(parents=True, exist_ok=True)

    if _snapshot_looks_complete(local_dir):
        return local_dir

    # ModelScope reuses existing files in local_dir and resumes/re-fetches missing
    # snapshot files. This is important for preempted Slurm jobs.
    kwargs = {
        "model_id": model_id,
        "local_dir": str(local_dir),
    }
    if revision:
        kwargs["revision"] = revision

    downloaded = Path(snapshot_download(**kwargs))

    if not _snapshot_looks_complete(downloaded):
        raise RuntimeError(
            f"Model snapshot appears incomplete after download: {model_id} -> {downloaded}"
        )
    return downloaded
