#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import numpy as np


EXPECTED = {"train": 26563, "test": 6461}


def infer_case_slice(name: str):
    stem = Path(name).stem
    for pat in (
        r"^(?P<case>.+?)[_-](?:slice|z)?(?P<slice>\d+)$",
        r"^(?P<case>.+?)[_-](?P<slice>\d+)$",
    ):
        m = re.match(pat, stem, flags=re.IGNORECASE)
        if m and m.group("case"):
            return m.group("case"), int(m.group("slice"))
    return None


def logical_suffix(name: str):
    low = name.lower()
    return ".nii.gz" if low.endswith(".nii.gz") else (PurePosixPath(low).suffix or "<none>")


def detect_root(names):
    files=[x for x in names if x and not x.endswith("/")]
    candidates={""}
    for n in files:
        parts=PurePosixPath(n).parts
        for i,p in enumerate(parts):
            if p in {"train","test"}:
                candidates.add("/".join(parts[:i]))
    for root in sorted(candidates,key=len):
        prefix=(root.rstrip("/")+"/") if root else ""
        rel=[x[len(prefix):] if x.startswith(prefix) else x for x in files]
        if any(x.startswith("train/train_npz/") for x in rel) or any(x.startswith("test/test_npz/") for x in rel):
            return root
    return None


def relname(name, root):
    if not root:
        return name
    prefix=root.rstrip("/")+"/"
    return name[len(prefix):] if name.startswith(prefix) else name


def filename_diagnostics(members):
    groups=defaultdict(list)
    failed=[]
    for m in members:
        p=infer_case_slice(PurePosixPath(m).stem)
        if p is None:
            failed.append(PurePosixPath(m).stem)
        else:
            groups[p[0]].append(p[1])
    cont=[]
    for zs in groups.values():
        zs=sorted(set(zs))
        if len(zs)>1:
            cont.append(all(b-a==1 for a,b in zip(zs,zs[1:])))
    return {
        "count":len(members),
        "parse_count":len(members)-len(failed),
        "parse_fraction":(len(members)-len(failed))/len(members) if members else 0.0,
        "inferred_case_count":len(groups),
        "multi_slice_case_count":sum(len(set(z))>1 for z in groups.values()),
        "contiguous_multi_slice_fraction":sum(cont)/len(cont) if cont else None,
        "unparsed_examples":failed[:20],
    }


def npz_summary(read_bytes, members, sample_n):
    if not members:
        return {"sample_count":0,"observed_label_values":[],"label_semantics":"UNVERIFIED"}
    idx=np.linspace(0,len(members)-1,num=min(sample_n,len(members)),dtype=int)
    samples=[]
    labels=set()
    shapes=Counter()
    label_shapes=Counter()
    mins=[]; maxs=[]
    for i in idx:
        name=members[int(i)]
        try:
            with np.load(io.BytesIO(read_bytes(name)),allow_pickle=False) as o:
                keys=list(o.files)
                item={"path":name,"keys":keys}
                if "image" in o and "label" in o:
                    image=np.asarray(o["image"])
                    label=np.asarray(o["label"])
                    vals=[]
                    for x in np.unique(label).tolist():
                        fx=float(x); vals.append(int(fx) if fx.is_integer() else fx)
                    labels.update(vals)
                    shapes[str(tuple(image.shape))]+=1
                    label_shapes[str(tuple(label.shape))]+=1
                    mins.append(float(np.nanmin(image))); maxs.append(float(np.nanmax(image)))
                    item.update({
                        "image_shape":list(image.shape),
                        "image_dtype":str(image.dtype),
                        "image_min":float(np.nanmin(image)),
                        "image_max":float(np.nanmax(image)),
                        "label_shape":list(label.shape),
                        "label_dtype":str(label.dtype),
                        "label_values":vals,
                    })
                samples.append(item)
        except Exception as e:
            samples.append({"path":name,"error":repr(e)})
    return {
        "sample_count":len(samples),
        "image_shapes":dict(shapes),
        "label_shapes":dict(label_shapes),
        "image_min_over_samples":min(mins) if mins else None,
        "image_max_over_samples":max(maxs) if maxs else None,
        "observed_label_values":sorted(labels),
        "label_semantics":"UNVERIFIED",
        "samples":samples,
    }


def csv_preview(text):
    rows=list(csv.reader(io.StringIO(text)))
    return {"header":rows[0] if rows else [],"preview":rows[:11],"row_count_including_header":len(rows)}


def finalize(report):
    fracs=[v["filename_structure"]["parse_fraction"] for v in report["splits"].values() if v["filename_structure"]["count"]]
    vals=set()
    for v in report["splits"].values():
        vals.update(v["npz_sample_audit"]["observed_label_values"])
    report["safety"]={
        "label_semantics_verified":False,
        "patient_slice_mapping_verified":False,
        "care_3d_tracks_enabled":False,
        "observed_label_values_from_samples":sorted(vals),
        "filename_mapping_candidate":bool(fracs) and all(x==1.0 for x in fracs),
    }
    report["next_steps"]=[
        "Do not assign medical meaning to CARE label IDs until verified against official annotation documentation/examples.",
        "Do not enable CARE T1/T3 unless patient identity and consecutive slice order are explicitly verified.",
        "If filenames do not encode case_id and slice_index, obtain/build an explicit care_index.csv mapping.",
        "Use CARE image arrays as packaged; do not apply an HU window unless original HU semantics are independently verified.",
    ]
    return report


def audit_zip(path: Path, sample_n: int):
    with zipfile.ZipFile(path) as z:
        infos=[x for x in z.infolist() if not x.is_dir()]
        names=[x.filename for x in infos]
        root=detect_root(names)
        rel={n:relname(n,root) for n in names}
        split_members={
            s:sorted(n for n in names if rel[n].startswith(f"{s}/{s}_npz/") and n.lower().endswith(".npz"))
            for s in ("train","test")
        }
        bbox={
            s:next((n for n in names if rel[n]==f"{s}/{s}_bbox.csv"),None)
            for s in ("train","test")
        }
        report={
            "source":str(path),
            "source_type":"zip",
            "archive_size_bytes":path.stat().st_size,
            "inventory":{
                "file_count":len(names),
                "uncompressed_bytes":sum(x.file_size for x in infos),
                "extension_counts":dict(Counter(logical_suffix(x) for x in names).most_common()),
                "detected_dataset_root":root,
                "split_npz_counts":{s:len(v) for s,v in split_members.items()},
                "bbox_csv":bbox,
            },
            "splits":{},
        }
        for s in ("train","test"):
            d={"filename_structure":filename_diagnostics(split_members[s])}
            if bbox[s]:
                d["bbox_csv_preview"]=csv_preview(z.read(bbox[s]).decode("utf-8-sig",errors="replace"))
            d["npz_sample_audit"]=npz_summary(z.read,split_members[s],sample_n)
            d["expected_public_pair_count_reference"]=EXPECTED[s]
            d["count_matches_reference"]=len(split_members[s])==EXPECTED[s]
            report["splits"][s]=d
        return finalize(report)


def find_extracted_root(root: Path):
    for p in [root]+[x for x in root.rglob("*") if x.is_dir()]:
        if (p/"train"/"train_npz").is_dir() or (p/"test"/"test_npz").is_dir():
            return p
    raise FileNotFoundError(f"Could not locate train/train_npz or test/test_npz under {root}")


def audit_dir(root: Path, sample_n: int):
    data=find_extracted_root(root)
    files=[p for p in data.rglob("*") if p.is_file()]
    names=[p.relative_to(data).as_posix() for p in files]
    pathmap={p.relative_to(data).as_posix():p for p in files}
    split_members={
        s:sorted(n for n in names if n.startswith(f"{s}/{s}_npz/") and n.lower().endswith(".npz"))
        for s in ("train","test")
    }
    bbox={s:(f"{s}/{s}_bbox.csv" if (data/s/f"{s}_bbox.csv").is_file() else None) for s in ("train","test")}
    report={
        "source":str(root),
        "source_type":"extracted_directory",
        "detected_dataset_root":str(data),
        "inventory":{
            "file_count":len(files),
            "extension_counts":dict(Counter(logical_suffix(x) for x in names).most_common()),
            "split_npz_counts":{s:len(v) for s,v in split_members.items()},
            "bbox_csv":bbox,
        },
        "splits":{},
    }
    def read_bytes(n): return pathmap[n].read_bytes()
    for s in ("train","test"):
        d={"filename_structure":filename_diagnostics(split_members[s])}
        if bbox[s]:
            d["bbox_csv_preview"]=csv_preview(pathmap[bbox[s]].read_text(encoding="utf-8-sig",errors="replace"))
        d["npz_sample_audit"]=npz_summary(read_bytes,split_members[s],sample_n)
        d["expected_public_pair_count_reference"]=EXPECTED[s]
        d["count_matches_reference"]=len(split_members[s])==EXPECTED[s]
        report["splits"][s]=d
    return finalize(report)


def print_report(r):
    inv=r["inventory"]
    print("="*72)
    print("CARE DATA AUDIT -- READ ONLY")
    print("="*72)
    print("Source:",r["source"])
    print("Type:",r["source_type"])
    print("NPZ counts:",inv.get("split_npz_counts"))
    print("BBox CSV:",inv.get("bbox_csv"))
    for split,d in r["splits"].items():
        fs=d["filename_structure"]; na=d["npz_sample_audit"]
        print("-"*72)
        print(f"[{split}] files={fs['count']} patient/slice parse={fs['parse_fraction']:.1%}")
        print("inferred cases:",fs["inferred_case_count"],"multi-slice:",fs["multi_slice_case_count"])
        print("image shapes:",na.get("image_shapes"))
        print("image range:",na.get("image_min_over_samples"),"..",na.get("image_max_over_samples"))
        print("label values:",na.get("observed_label_values"),"(SEMANTICS UNVERIFIED)")
        if fs["unparsed_examples"]:
            print("unparsed examples:",fs["unparsed_examples"][:10])
    print("-"*72)
    for k,v in r["safety"].items(): print(f"{k}: {v}")
    print("Next steps:")
    for x in r["next_steps"]: print(" -",x)


def main():
    p=argparse.ArgumentParser(description="Read-only CARE release audit")
    p.add_argument("source",help="CARE.zip or extracted CARE directory")
    p.add_argument("--sample-n",type=int,default=20)
    p.add_argument("--output",default="manifests/care_audit.json")
    a=p.parse_args()
    src=Path(a.source)
    if src.is_dir():
        r=audit_dir(src,a.sample_n)
    elif src.name.lower().endswith(".zip"):
        r=audit_zip(src,a.sample_n)
    else:
        raise SystemExit("CARE audit currently supports CARE.zip or an extracted directory.")
    print_report(r)
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(r,indent=2,ensure_ascii=False),encoding="utf-8")
    print("\nSaved audit report ->",out)


if __name__=="__main__":
    main()
