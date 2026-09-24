from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path
from .preprocess import infer_case_slice


def index_msd(root):
    root = Path(root); rows = []
    for image_path in sorted((root/"imagesTr").glob("*.nii.gz")):
        mask_path = root/"labelsTr"/image_path.name
        if mask_path.exists():
            rows.append({"dataset":"MSD","case_id":image_path.name.replace(".nii.gz",""),"format":"nifti","image_path":str(image_path),"mask_path":str(mask_path),"split":"public_train"})
    return rows


def _care_csv_rows(csv_path, npz_dir, split):
    out=[]
    with open(csv_path,"r",encoding="utf-8-sig",newline="") as f:
        reader=csv.reader(f); header=next(reader,None); candidates=[] if header is None else [header]; candidates.extend(reader)
        for row in candidates:
            if not row: continue
            name=row[0].strip()
            if name.lower() in {"filename","file","name"}: continue
            npz_path=npz_dir/f"{name}.npz"
            if not npz_path.exists(): continue
            parsed=infer_case_slice(name)
            if parsed is None: continue
            case_id,slice_index=parsed
            out.append({"case_id":case_id,"slice_index":slice_index,"npz_path":str(npz_path),"split":split})
    return out


def index_care(root, mapping_csv=None):
    root=Path(root); slice_rows=[]
    if mapping_csv:
        import pandas as pd
        df=pd.read_csv(mapping_csv); required={"case_id","slice_index","npz_path"}; missing=required-set(df.columns)
        if missing: raise ValueError(f"CARE mapping missing columns: {sorted(missing)}")
        for r in df.to_dict("records"):
            p=Path(str(r["npz_path"])); p=p if p.is_absolute() else root/p; r["npz_path"]=str(p); r.setdefault("split","unknown"); slice_rows.append(r)
    else:
        for split in ("train","test"):
            csv_path=root/split/f"{split}_bbox.csv"; npz_dir=root/split/f"{split}_npz"
            if csv_path.exists() and npz_dir.exists(): slice_rows.extend(_care_csv_rows(csv_path,npz_dir,split))
    groups=defaultdict(list)
    for r in slice_rows: groups[(str(r.get("split","unknown")),str(r["case_id"]))].append(r)
    records=[]
    for (split,case_id),rows in sorted(groups.items()):
        rows=sorted(rows,key=lambda x:int(x["slice_index"]))
        records.append({"dataset":"CARE","case_id":case_id,"format":"care_npz_series","split":split,"slices":rows})
    return records


def build_case_index(msd_root, care_root, care_mapping=None):
    rows=[]
    if msd_root: rows.extend(index_msd(msd_root))
    if care_root: rows.extend(index_care(care_root,care_mapping))
    return rows
