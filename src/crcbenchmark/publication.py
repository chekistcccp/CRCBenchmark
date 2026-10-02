"""Development-only experiments measuring task validity and image availability."""
from collections import defaultdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
from PIL import Image, ImageDraw

from .evaluate import eval_items, patient_aggregate
from .stats import summarize_patient_rows
from .supplement import paired_comparisons, score_slice_set
from .utils import parse_label_list_response

CONDITIONS = ("original", "text_only", "neutral_image")
ENDPOINTS = {"t1": ("recall_3", "none_specificity"), "t2": ("pointing_acc",),
             "t3": ("slice_f1", "balanced_accuracy"), "t4": ("pairwise_acc",)}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_artifact(path, artifact_root=None):
    original = Path(path)
    if artifact_root is None:
        if not original.is_file():
            raise FileNotFoundError(f"Missing artifact: {path}; use --artifact-root to relocate synced artifacts")
        return original.resolve()
    # Portable remapping is explicit and anchored at the last artifacts/ component.
    parts = str(path).replace("\\", "/").split("/")
    positions = [i for i, part in enumerate(parts) if part == "artifacts"]
    if not positions:
        raise ValueError(f"Cannot remap artifact without artifacts/ component: {path}")
    root = Path(artifact_root).resolve()
    candidate = root.joinpath(*parts[positions[-1] + 1:]).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("Artifact path escapes supplied root")
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def dev_items(manifest):
    if not manifest or any(r.get("benchmark_split") != "dev" for r in manifest):
        raise ValueError("Publication supplements require development patients only")
    if len({r["item_id"] for r in manifest}) != len(manifest):
        raise ValueError("Duplicate development item IDs")
    rows = [r for r in manifest if r["track"] in ENDPOINTS]
    if not rows:
        raise ValueError("No T1-T4 development items")
    return rows


def relocate_items(manifest, artifact_root=None):
    rows = []
    for source in dev_items(manifest):
        row = dict(source)
        row["image_path"] = str(resolve_artifact(row["image_path"], artifact_root))
        row["source_image_sha256"] = file_sha256(row["image_path"])
        if row.get("gt_mask_path"):
            row["gt_mask_path"] = str(resolve_artifact(row["gt_mask_path"], artifact_root))
            row["gt_mask_sha256"] = file_sha256(row["gt_mask_path"])
        rows.append(row)
    return rows


def evidence_items(source, neutral_root):
    rows = []
    for item in dev_items(source):
        with Image.open(item["image_path"]) as im:
            size = im.size
        neutral = Image.new("RGB", size, (128, 128, 128))
        geometry = {"t1": (3, 4, "ABCDEFGHIJKL"), "t3": (3, 3, "ABCDEFGHI"),
                    "t4": (1, 2, "AB")}.get(item["track"])
        if geometry:
            nr, nc, labels = geometry
            if size[0] % nc or size[1] % nr or size[0] // nc != size[1] // nr:
                raise ValueError("Unexpected montage geometry")
            tile = size[0] // nc
            draw = ImageDraw.Draw(neutral)
            for index, label in enumerate(labels):
                x, y = (index % nc) * tile, (index // nc) * tile
                draw.rectangle((x, y, x + 28, y + 24), fill="black")
                draw.text((x + 6, y + 3), label, fill="white")
        key = hashlib.sha256(item["item_id"].encode()).hexdigest()[:24]
        path = Path(neutral_root) / f"{key}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.resolve() == Path(item["image_path"]).resolve():
            raise ValueError("Neutral output would overwrite input")
        neutral.save(path)
        for condition in CONDITIONS:
            image_path = {"original": item["image_path"], "text_only": None,
                          "neutral_image": str(path.resolve())}[condition]
            rows.append({**item, "item_id": f'{item["item_id"]}:evidence:{condition}',
                         "source_item_id": item["item_id"], "evidence_condition": condition,
                         "image_path": image_path,
                         "input_image_sha256": file_sha256(image_path) if image_path else None})
    return rows


def score_dev_items(items, predictions):
    rows = eval_items(items, predictions)
    by_item = {r["item_id"]: r for r in items}
    by_pred = {r["item_id"]: r for r in predictions}
    for row in rows:
        row["format_valid"] = 1 - row["invalid"]
        if row["track"] == "t3":
            item = by_item[row["item_id"]]
            parsed = parse_label_list_response(by_pred[row["item_id"]]["raw_response"], "ABCDEFGHI")
            row.update(score_slice_set(item, parsed or []) if parsed is not None else
                       {"slice_f1": 0., "balanced_accuracy": 0., "exact_set": 0.})
    return patient_aggregate(rows)


def evidence_report(items, predictions):
    by_pred = {r["item_id"]: r for r in predictions}
    if len(by_pred) != len(predictions) or set(by_pred) != {r["item_id"] for r in items}:
        raise ValueError("Evidence predictions must uniquely and completely match manifest")
    patients = {}
    for condition in CONDITIONS:
        subset = [r for r in items if r["evidence_condition"] == condition]
        patients[condition] = score_dev_items(subset, [by_pred[r["item_id"]] for r in subset])
    contrasts = []
    for control in CONDITIONS[1:]:
        baseline = {(r["track"], r["case_id"]): r for r in patients[control]}
        paired = [{**r, **{f"control_{k}": v for k, v in baseline[(r["track"], r["case_id"])].items()
                          if isinstance(v, (int, float))}}
                  for r in patients["original"]]
        comparisons = [(track, metric, f"control_{metric}") for track, metrics in ENDPOINTS.items()
                       for metric in (*metrics, "format_valid")]
        contrasts.append({"control": control, "direction": "original_minus_control",
                          "comparisons": paired_comparisons(paired, comparisons)})
    return {"scope": "development_only_posthoc", "n_items": len(items),
            "conditions": {k: summarize_patient_rows(v) for k, v in patients.items()},
            "paired_differences": contrasts,
            "interval_policy": "exploratory paired patient bootstrap, 2000 resamples; no significance selection",
            "limitations": "Image availability effect includes input-distribution and format effects. Neither control proves causal lesion grounding; report all conditions and invalid rates."}


def runtime_provenance():
    packages = {}
    for name in ("torch", "torchvision", "transformers", "accelerate", "modelscope", "numpy", "Pillow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}


def select_human_items(manifest, patients_per_group=10, seed=20261002):
    if patients_per_group < 1:
        raise ValueError("patients_per_group must be positive")
    groups = defaultdict(lambda: defaultdict(list))
    for item in dev_items(manifest):
        groups[(item["dataset"], item["track"])][item["case_id"]].append(item)
    def order(value):
        return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()
    selected = []
    for group, cases in sorted(groups.items()):
        for case in sorted(cases, key=lambda c: order(f"{group}:{c}"))[:patients_per_group]:
            options = sorted(cases[case], key=lambda r: order(r["item_id"]))
            # Keep T4 swaps paired; other tracks have one question per patient.
            selected.extend(options if group[1] == "t4" else options[:1])
    return sorted(selected, key=lambda r: order("presentation:" + r["item_id"]))


def validity_agreement(first, second):
    categories = ("yes", "no", "uncertain")
    if not first or len(first) != len(second) or any(v not in categories for v in first + second):
        raise ValueError("Expected paired yes/no/uncertain validity ratings")
    observed = np.mean([a == b for a, b in zip(first, second)])
    expected = sum(first.count(c) * second.count(c) / len(first) ** 2 for c in categories)
    return {"n": len(first), "agreement": float(observed),
            "cohen_kappa": float((observed - expected) / (1 - expected)) if expected < 1 else None}
