from __future__ import annotations
from pathlib import Path
import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(root: str | Path, value: str | Path) -> Path:
    value = Path(value)
    return value if value.is_absolute() else Path(root) / value
