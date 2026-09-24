from __future__ import annotations
from pathlib import Path
from modelscope import snapshot_download

def ensure_model(model_id:str,model_root:str|Path,revision:str|None=None)->Path:
    model_root=Path(model_root); local_dir=model_root/model_id.replace("/","__")
    if (local_dir/"config.json").exists(): return local_dir
    local_dir.mkdir(parents=True,exist_ok=True)
    kwargs={"model_id":model_id,"local_dir":str(local_dir)}
    if revision: kwargs["revision"]=revision
    return Path(snapshot_download(**kwargs))
