#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import zipfile
from pathlib import Path


ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz")


def is_archive(path: Path) -> bool:
    n = path.name.lower()
    return any(n.endswith(s) for s in ARCHIVE_SUFFIXES)


def safe_target(base: Path, member_name: str) -> Path:
    base = base.resolve()
    target = (base / member_name).resolve()
    if target != base and base not in target.parents:
        raise RuntimeError(f"Unsafe archive path: {member_name}")
    return target


def extract_zip(path: Path, dest: Path):
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            target = safe_target(dest, info.filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.is_file() and target.stat().st_size == info.file_size:
                continue
            with z.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)


def extract_tar(path: Path, dest: Path):
    with tarfile.open(path, "r:*") as t:
        for member in t.getmembers():
            if member.isdir():
                safe_target(dest, member.name).mkdir(parents=True, exist_ok=True)
                continue
            if member.issym() or member.islnk() or not member.isfile():
                continue
            target = safe_target(dest, member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.is_file() and target.stat().st_size == member.size:
                continue
            src = t.extractfile(member)
            if src is None:
                continue
            with src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)


def extract_archive(path: Path, dest: Path):
    print(f"[extract] {path} -> {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    if path.name.lower().endswith(".zip"):
        extract_zip(path, dest)
    else:
        extract_tar(path, dest)


def find_msd_root(search_roots):
    for root in search_roots:
        if not root or not root.exists():
            continue
        candidates = [root] + [p for p in root.rglob("*") if p.is_dir()]
        for p in candidates:
            if (p / "imagesTr").is_dir() and (p / "labelsTr").is_dir():
                return p.resolve()
    return None


def find_care_root(search_roots):
    for root in search_roots:
        if not root or not root.exists():
            continue
        candidates = [root] + [p for p in root.rglob("*") if p.is_dir()]
        for p in candidates:
            if (
                (p / "test" / "test_npz").is_dir()
                and (p / "test" / "test.txt").is_file()
            ):
                return p.resolve()
    return None


def extract_all_archives(raw_dir: Path, out_dir: Path):
    archives = sorted(p for p in raw_dir.rglob("*") if p.is_file() and is_archive(p))
    if not archives:
        return
    for archive in archives:
        extract_archive(archive, out_dir)


def main():
    p = argparse.ArgumentParser(description="Prepare MSD and CARE from raw archives.")
    p.add_argument("--data-root", default="data")
    p.add_argument("--output", default="manifests/data_paths.json")
    p.add_argument("--skip-care", action="store_true")
    a = p.parse_args()

    data = Path(a.data_root)
    raw_msd = data / "raw" / "MSD"
    raw_care = data / "raw" / "CARE"
    out_msd = data / "extracted" / "MSD"
    out_care = data / "extracted" / "CARE"

    # Also support the older already-extracted data/MSD layout.
    msd = find_msd_root([data / "MSD", out_msd])
    if msd is None:
        if not raw_msd.exists():
            raise SystemExit(
                f"MSD not found. Put the MSD archive under {raw_msd}/ "
                "or an extracted dataset under data/MSD/."
            )
        extract_all_archives(raw_msd, out_msd)
        msd = find_msd_root([out_msd])
    if msd is None:
        raise SystemExit("Could not locate MSD imagesTr/labelsTr after extraction.")

    care = None
    if not a.skip_care:
        care = find_care_root([data / "CARE", out_care])
        if care is None:
            if not raw_care.exists():
                raise SystemExit(
                    f"CARE not found. Put CARE.zip under {raw_care}/ "
                    "or provide an extracted dataset under data/CARE/."
                )
            extract_all_archives(raw_care, out_care)
            care = find_care_root([out_care])
        if care is None:
            raise SystemExit("Could not locate CARE test/test_npz + test.txt after extraction.")

    result = {
        "msd_root": str(msd),
        "care_root": str(care) if care else None,
    }

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print("Saved:", out)


if __name__ == "__main__":
    main()
