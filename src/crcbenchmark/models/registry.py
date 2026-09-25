from __future__ import annotations

from .chat_adapters import InternVLChatVLM, MiniCPMChatVLM
from .modelscope import ensure_model
from .pipeline_adapter import PipelineVLM


def build_model(model_key, models_cfg, model_root):
    spec = models_cfg["models"][model_key]
    local = ensure_model(
        spec["modelscope_id"],
        model_root,
        spec.get("revision"),
    )

    adapter = spec.get("adapter", "pipeline")
    common = dict(
        model_path=local,
        dtype=spec.get("dtype", "bfloat16"),
        trust_remote_code=bool(spec.get("trust_remote_code", True)),
        max_new_tokens=int(spec.get("max_new_tokens", 96)),
    )

    if adapter == "pipeline":
        model = PipelineVLM(**common)
    elif adapter == "internvl_chat":
        model = InternVLChatVLM(**common)
    elif adapter == "minicpm_chat":
        model = MiniCPMChatVLM(**common)
    else:
        raise ValueError(f"Unsupported adapter for {model_key}: {adapter}")

    return model, local
