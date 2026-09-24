from __future__ import annotations
from .modelscope import ensure_model
from .pipeline_adapter import PipelineVLM

def build_model(model_key,models_cfg,model_root):
    spec=models_cfg["models"][model_key]; local=ensure_model(spec["modelscope_id"],model_root,spec.get("revision"))
    if spec.get("adapter","pipeline")!="pipeline": raise ValueError("unsupported adapter")
    return PipelineVLM(local,dtype=spec.get("dtype","bfloat16"),trust_remote_code=bool(spec.get("trust_remote_code",True)),max_new_tokens=int(spec.get("max_new_tokens",96))),local
