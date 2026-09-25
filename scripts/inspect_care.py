#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import statistics
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import numpy as np


REFERENCE = {
    "paper_total_patients": 398,
    "paper_train_cases_methods": 318,
    "paper_test_cases": 81,
    "paper_train_pairs": 26563,
    "paper_test_pairs": 6461,
}


def infer_case_slice(name: str):
    stem = Path(name).stem
    for pat in (
        r"^(?P<case>.+?)[_-](?:slice|z)(?P<slice>\d+)$",
        r"^(?P<case>.+?)[_-](?P<slice>\d+)$",
    ):
        m = re.match(pat, stem, flags=re.IGNORECASE)
        if m and m.group("case"):
            return m.group("case"), int(m.group("slice"))
    return None


def logical_suffix(name: str):
    low = name.lower()
    if low.endswith(".nii.gz"):
        return ".nii.gz"
    return PurePosixPath(low).suffix or "<none>"


def detect_root(names):
    files = [x for x in names if x and not x.endswith("/")]
    candidates = {""}
    for n in files:
        parts = PurePosixPath(n).parts
        for i, p in enumerate(parts):
            if p in {"train", "test"}:
                candidates.add("/".join(parts[:i]))
    for root in sorted(candidates, key=len):
        prefix = (root.rstrip("/") + "/") if root else ""
        rel = [x[len(prefix):] if x.startswith(prefix) else x for x in files]
        if any(x.startswith("train/train_npz/") for x in rel) or any(
            x.startswith("test/test_npz/") for x in rel
        ):
            return root
    return None


def relname(name, root):
    if not root:
        return name
    prefix = root.rstrip("/") + "/"
    return name[len(prefix):] if name.startswith(prefix) else name


def longest_consecutive_run(values):
    vals = sorted(set(int(x) for x in values))
    if not vals:
        return 0
    best = cur = 1
    for a, b in zip(vals, vals[1:]):
        if b - a == 1:
            cur += 1
        else:
            best = max(best, cur)
            cur = 1
    return max(best, cur)


def filename_diagnostics(members):
    groups = defaultdict(list)
    failed = []
    duplicate_case_slice = []
    seen = set()

    for m in members:
        parsed = infer_case_slice(PurePosixPath(m).stem)
        if parsed is None:
            failed.append(PurePosixPath(m).stem)
            continue
        key = (parsed[0], int(parsed[1]))
        if key in seen:
            duplicate_case_slice.append(f"{key[0]}:{key[1]}")
        seen.add(key)
        groups[parsed[0]].append(int(parsed[1]))

    slice_counts = []
    full_contiguous = []
    longest_runs = []
    total_gaps = 0
    cases_with_gaps = 0

    for zs in groups.values():
        zs = sorted(set(zs))
        slice_counts.append(len(zs))
        gaps = [b - a for a, b in zip(zs, zs[1:]) if b - a > 1]
        if gaps:
            cases_with_gaps += 1
            total_gaps += len(gaps)
        if len(zs) > 1:
            full_contiguous.append(not gaps)
        longest_runs.append(longest_consecutive_run(zs))

    return {
        "count": len(members),
        "parse_count": len(members) - len(failed),
        "parse_fraction": (len(members) - len(failed)) / len(members) if members else 0.0,
        "inferred_case_count": len(groups),
        "case_ids": sorted(groups.keys()),
        "multi_slice_case_count": sum(len(set(z)) > 1 for z in groups.values()),
        "fully_contiguous_case_fraction": (
            sum(full_contiguous) / len(full_contiguous) if full_contiguous else None
        ),
        "cases_with_any_gap": cases_with_gaps,
        "gap_event_count": total_gaps,
        "cases_with_consecutive_run_ge_9": sum(r >= 9 for r in longest_runs),
        "cases_with_consecutive_run_ge_5": sum(r >= 5 for r in longest_runs),
        "slice_count_per_case": {
            "min": min(slice_counts) if slice_counts else None,
            "median": statistics.median(slice_counts) if slice_counts else None,
            "max": max(slice_counts) if slice_counts else None,
        },
        "longest_consecutive_run_per_case": {
            "min": min(longest_runs) if longest_runs else None,
            "median": statistics.median(longest_runs) if longest_runs else None,
            "max": max(longest_runs) if longest_runs else None,
        },
        "duplicate_case_slice_count": len(duplicate_case_slice),
        "duplicate_case_slice_examples": duplicate_case_slice[:20],
        "unparsed_examples": failed[:20],
    }


def parse_bbox_csv(text):
    rows = list(csv.reader(io.StringIO(text)))
    header = rows[0] if rows else []
    body = []
    for r in rows[1:] if rows else []:
        if not r:
            continue
        name = r[0].strip()
        if not name:
            continue
        body.append(r)
    ids = [r[0].strip() for r in body]
    dup = [x for x, c in Counter(ids).items() if c > 1]
    return {
        "header": header,
        "preview": rows[:11],
        "row_count_including_header": len(rows),
        "record_count": len(body),
        "unique_pid_count": len(set(ids)),
        "duplicate_pid_count": len(dup),
        "duplicate_pid_examples": sorted(dup)[:20],
        "_ids": ids,
    }


def membership_diagnostics(csv_info, npz_members):
    ids = list(csv_info.get("_ids", []))
    npz_by_stem = {PurePosixPath(x).stem: x for x in npz_members}
    csv_set = set(ids)
    npz_set = set(npz_by_stem.keys())

    csv_missing_npz = sorted(csv_set - npz_set)
    npz_not_in_csv = sorted(npz_set - csv_set)

    return {
        "csv_record_count": len(ids),
        "npz_count": len(npz_members),
        "csv_records_with_existing_npz": sum(x in npz_set for x in ids),
        "csv_missing_npz_count": len(csv_missing_npz),
        "csv_missing_npz_examples": csv_missing_npz[:20],
        "npz_not_referenced_by_csv_count": len(npz_not_in_csv),
        "npz_not_referenced_by_csv_examples": npz_not_in_csv[:20],
        "note": "The public U-SAM dataloader indexes CARE through the bbox CSV; unreferenced NPZ files are therefore not included by default in CRCBenchmark.",
    }


def npz_summary(read_bytes, members, sample_n):
    if not members:
        return {
            "sample_count": 0,
            "observed_raw_label_values": [],
            "canonical_label_values_after_usam_rule": [],
            "label_semantics": "UNVERIFIED",
        }

    idx = np.linspace(0, len(members) - 1, num=min(sample_n, len(members)), dtype=int)
    samples = []
    raw_labels = set()
    canonical_labels = set()
    shapes = Counter()
    label_shapes = Counter()
    mins, maxs = [], []

    for i in idx:
        name = members[int(i)]
        try:
            with np.load(io.BytesIO(read_bytes(name)), allow_pickle=False) as o:
                keys = list(o.files)
                item = {"path": name, "keys": keys}
                if "image" in o and "label" in o:
                    image = np.asarray(o["image"])
                    label = np.asarray(o["label"])
                    if not np.all(np.isfinite(label)):
                        raise ValueError("non-finite label values")
                    rounded = np.rint(label)
                    if not np.allclose(label, rounded):
                        raise ValueError("CARE label contains non-integer values")
                    raw_vals = sorted(set(int(x) for x in np.unique(rounded).tolist()))
                    canonical = rounded.astype(np.int16)
                    canonical[canonical > 2] = 2
                    canonical_vals = sorted(set(int(x) for x in np.unique(canonical).tolist()))

                    raw_labels.update(raw_vals)
                    canonical_labels.update(canonical_vals)
                    shapes[str(tuple(image.shape))] += 1
                    label_shapes[str(tuple(label.shape))] += 1
                    mins.append(float(np.nanmin(image)))
                    maxs.append(float(np.nanmax(image)))
                    item.update(
                        {
                            "image_shape": list(image.shape),
                            "image_dtype": str(image.dtype),
                            "image_min": float(np.nanmin(image)),
                            "image_max": float(np.nanmax(image)),
                            "label_shape": list(label.shape),
                            "label_dtype": str(label.dtype),
                            "raw_label_values": raw_vals,
                            "canonical_label_values_after_usam_rule": canonical_vals,
                        }
                    )
                samples.append(item)
        except Exception as e:
            samples.append({"path": name, "error": repr(e)})

    return {
        "sample_count": len(samples),
        "image_shapes": dict(shapes),
        "label_shapes": dict(label_shapes),
        "image_min_over_samples": min(mins) if mins else None,
        "image_max_over_samples": max(maxs) if maxs else None,
        "observed_raw_label_values": sorted(raw_labels),
        "canonical_label_values_after_usam_rule": sorted(canonical_labels),
        "official_usam_preprocessing_rule": "raw label values > 2 are collapsed to canonical class 2 before training/evaluation",
        "label_semantics": "UNVERIFIED",
        "samples": samples,
    }


def small_metadata_previews(names, size_lookup, read_bytes, max_bytes=1_000_000):
    selected = []
    for name in names:
        low = name.lower()
        if (
            low.endswith(".txt")
            or low.endswith(".md")
            or low.endswith(".csv#")
            or low.endswith("readme")
        ) and size_lookup(name) <= max_bytes:
            try:
                txt = read_bytes(name).decode("utf-8-sig", errors="replace")
                selected.append(
                    {
                        "path": name,
                        "size_bytes": size_lookup(name),
                        "preview": txt[:5000],
                    }
                )
            except Exception as e:
                selected.append({"path": name, "error": repr(e)})
    return selected


def finalize(report):
    train_cases = set(report["splits"]["train"]["filename_structure"].get("case_ids", []))
    test_cases = set(report["splits"]["test"]["filename_structure"].get("case_ids", []))
    overlap = sorted(train_cases & test_cases)
    total_unique = len(train_cases | test_cases)

    report["cross_split"] = {
        "train_case_count": len(train_cases),
        "test_case_count": len(test_cases),
        "overlap_case_count": len(overlap),
        "overlap_case_examples": overlap[:20],
        "total_unique_case_ids": total_unique,
        "reference_total_patients": REFERENCE["paper_total_patients"],
        "matches_reference_total": total_unique == REFERENCE["paper_total_patients"],
    }

    for split in ("train", "test"):
        report["splits"][split]["filename_structure"].pop("case_ids", None)
        csv_info = report["splits"][split].get("bbox_csv")
        if csv_info:
            csv_info.pop("_ids", None)

    fracs = [
        report["splits"][s]["filename_structure"]["parse_fraction"]
        for s in ("train", "test")
        if report["splits"][s]["filename_structure"]["count"]
    ]
    raw_vals = set()
    canonical_vals = set()
    for s in ("train", "test"):
        na = report["splits"][s]["npz_sample_audit"]
        raw_vals.update(na.get("observed_raw_label_values", []))
        canonical_vals.update(na.get("canonical_label_values_after_usam_rule", []))

    report["safety"] = {
        "label_semantics_verified": False,
        "patient_slice_mapping_verified": False,
        "observed_raw_label_values_from_samples": sorted(raw_vals),
        "canonical_label_values_after_usam_rule": sorted(canonical_vals),
        "filename_mapping_candidate": bool(fracs) and all(x == 1.0 for x in fracs),
        "cross_split_overlap_free": len(overlap) == 0,
        "care_t1_structurally_possible": bool(fracs) and all(x == 1.0 for x in fracs),
        "care_t3_local_windows_structurally_possible": any(
            report["splits"][s]["filename_structure"]["cases_with_consecutive_run_ge_9"] > 0
            for s in ("train", "test")
        ),
        "care_semantic_tracks_enabled": False,
    }

    report["next_steps"] = [
        "Use bbox CSV membership as the default CARE inclusion list because the official U-SAM dataloader is CSV-indexed.",
        "Do not assign normal/tumor meaning to canonical label IDs 1 and 2 until verified from official annotation evidence.",
        "Treat raw CARE label 3 according to the official U-SAM preprocessing rule: collapse raw values >2 to canonical class 2.",
        "CARE T1 can be enabled after label semantics are verified because filenames preserve case_id and slice_index.",
        "CARE T3 should use only local consecutive source-slice windows; do not require every retained slice in a patient to be globally contiguous.",
        "Review cross-split overlap, CSV/NPZ set differences, and small metadata-file previews before freezing the benchmark.",
        "Use CARE image arrays as packaged; do not apply an HU window unless original HU semantics are independently verified.",
    ]
    return report


def build_split_report(split, npz_members, bbox_name, read_bytes, sample_n):
    csv_info = None
    membership = None
    if bbox_name:
        csv_info = parse_bbox_csv(read_bytes(bbox_name).decode("utf-8-sig", errors="replace"))
        membership = membership_diagnostics(csv_info, npz_members)

    return {
        "filename_structure": filename_diagnostics(npz_members),
        "bbox_csv": csv_info,
        "csv_npz_membership": membership,
        "npz_sample_audit": npz_summary(read_bytes, npz_members, sample_n),
        "reference": {
            "paper_pair_count": REFERENCE[f"paper_{split}_pairs"],
            "archive_npz_count_matches_paper": len(npz_members)
            == REFERENCE[f"paper_{split}_pairs"],
            "npz_count_difference_vs_paper": len(npz_members)
            - REFERENCE[f"paper_{split}_pairs"],
        },
    }


def audit_zip(path: Path, sample_n: int):
    with zipfile.ZipFile(path) as z:
        infos = [x for x in z.infolist() if not x.is_dir()]
        names = [x.filename for x in infos]
        size_map = {x.filename: int(x.file_size) for x in infos}
        root = detect_root(names)
        rel = {n: relname(n, root) for n in names}
        split_members = {
            s: sorted(
                n
                for n in names
                if rel[n].startswith(f"{s}/{s}_npz/") and n.lower().endswith(".npz")
            )
            for s in ("train", "test")
        }
        bbox = {
            s: next((n for n in names if rel[n] == f"{s}/{s}_bbox.csv"), None)
            for s in ("train", "test")
        }

        def read_bytes(name):
            return z.read(name)

        report = {
            "audit_version": 2,
            "source": str(path),
            "source_type": "zip",
            "archive_size_bytes": path.stat().st_size,
            "inventory": {
                "file_count": len(names),
                "uncompressed_bytes": sum(x.file_size for x in infos),
                "extension_counts": dict(Counter(logical_suffix(x) for x in names).most_common()),
                "detected_dataset_root": root,
                "split_npz_counts": {s: len(v) for s, v in split_members.items()},
                "bbox_csv": bbox,
                "small_metadata_files": small_metadata_previews(
                    names, lambda n: size_map[n], read_bytes
                ),
            },
            "splits": {},
        }
        for s in ("train", "test"):
            report["splits"][s] = build_split_report(
                s, split_members[s], bbox[s], read_bytes, sample_n
            )
        return finalize(report)


def find_extracted_root(root: Path):
    for p in [root] + [x for x in root.rglob("*") if x.is_dir()]:
        if (p / "train" / "train_npz").is_dir() or (p / "test" / "test_npz").is_dir():
            return p
    raise FileNotFoundError(
        f"Could not locate train/train_npz or test/test_npz under {root}"
    )


def audit_dir(root: Path, sample_n: int):
    data = find_extracted_root(root)
    files = [p for p in data.rglob("*") if p.is_file()]
    names = [p.relative_to(data).as_posix() for p in files]
    pathmap = {p.relative_to(data).as_posix(): p for p in files}
    split_members = {
        s: sorted(
            n
            for n in names
            if n.startswith(f"{s}/{s}_npz/") and n.lower().endswith(".npz")
        )
        for s in ("train", "test")
    }
    bbox = {
        s: (f"{s}/{s}_bbox.csv" if (data / s / f"{s}_bbox.csv").is_file() else None)
        for s in ("train", "test")
    }

    def read_bytes(name):
        return pathmap[name].read_bytes()

    report = {
        "audit_version": 2,
        "source": str(root),
        "source_type": "extracted_directory",
        "detected_dataset_root": str(data),
        "inventory": {
            "file_count": len(files),
            "extension_counts": dict(Counter(logical_suffix(x) for x in names).most_common()),
            "split_npz_counts": {s: len(v) for s, v in split_members.items()},
            "bbox_csv": bbox,
            "small_metadata_files": small_metadata_previews(
                names, lambda n: pathmap[n].stat().st_size, read_bytes
            ),
        },
        "splits": {},
    }
    for s in ("train", "test"):
        report["splits"][s] = build_split_report(
            s, split_members[s], bbox[s], read_bytes, sample_n
        )
    return finalize(report)


def print_report(r):
    inv = r["inventory"]
    print("=" * 78)
    print("CARE DATA AUDIT v2 -- READ ONLY")
    print("=" * 78)
    print("Source:", r["source"])
    print("Type:", r["source_type"])
    print("Dataset root:", inv.get("detected_dataset_root", r.get("detected_dataset_root")))
    print("NPZ counts:", inv.get("split_npz_counts"))
    print("BBox CSV:", inv.get("bbox_csv"))

    for split, d in r["splits"].items():
        fs = d["filename_structure"]
        na = d["npz_sample_audit"]
        mem = d.get("csv_npz_membership") or {}
        ref = d.get("reference") or {}
        print("-" * 78)
        print(
            f"[{split}] npz={fs['count']} cases={fs['inferred_case_count']} "
            f"patient/slice parse={fs['parse_fraction']:.1%}"
        )
        print(
            "global contiguous fraction:",
            fs["fully_contiguous_case_fraction"],
            "| cases with run>=9:",
            fs["cases_with_consecutive_run_ge_9"],
        )
        if mem:
            print(
                "CSV records:",
                mem.get("csv_record_count"),
                "| CSV->NPZ missing:",
                mem.get("csv_missing_npz_count"),
                "| NPZ not in CSV:",
                mem.get("npz_not_referenced_by_csv_count"),
            )
        print(
            "paper pair count:",
            ref.get("paper_pair_count"),
            "| archive difference:",
            ref.get("npz_count_difference_vs_paper"),
        )
        print("image shapes:", na.get("image_shapes"))
        print(
            "image range:",
            na.get("image_min_over_samples"),
            "..",
            na.get("image_max_over_samples"),
        )
        print(
            "raw label values:",
            na.get("observed_raw_label_values"),
            "| canonical after U-SAM rule:",
            na.get("canonical_label_values_after_usam_rule"),
            "(SEMANTICS UNVERIFIED)",
        )
        if fs["unparsed_examples"]:
            print("unparsed examples:", fs["unparsed_examples"][:10])

    print("-" * 78)
    cs = r["cross_split"]
    print(
        "cross-split cases:",
        cs["train_case_count"],
        "+",
        cs["test_case_count"],
        "| overlap:",
        cs["overlap_case_count"],
        "| total unique:",
        cs["total_unique_case_ids"],
    )
    if cs["overlap_case_examples"]:
        print("overlap examples:", cs["overlap_case_examples"][:10])

    if inv.get("small_metadata_files"):
        print("-" * 78)
        print("Small metadata files found:")
        for x in inv["small_metadata_files"]:
            print(" -", x.get("path"), f"({x.get('size_bytes','?')} bytes)")

    print("-" * 78)
    for k, v in r["safety"].items():
        print(f"{k}: {v}")
    print("Next steps:")
    for x in r["next_steps"]:
        print(" -", x)


def main():
    p = argparse.ArgumentParser(description="Read-only CARE release audit")
    p.add_argument("source", help="CARE.zip or extracted CARE directory")
    p.add_argument("--sample-n", type=int, default=50)
    p.add_argument("--output", default="manifests/care_audit_v2.json")
    a = p.parse_args()

    src = Path(a.source)
    if src.is_dir():
        r = audit_dir(src, a.sample_n)
    elif src.name.lower().endswith(".zip"):
        r = audit_zip(src, a.sample_n)
    else:
        raise SystemExit(
            "CARE audit currently supports CARE.zip or an extracted directory."
        )

    print_report(r)
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nSaved audit report ->", out)


if __name__ == "__main__":
    main()
