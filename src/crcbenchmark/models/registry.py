from __future__ import annotations

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
    if adapter != "pipeline":
        raise ValueError(
            f"Unsupported adapter for {model_key}: {adapter}. "
            "The current benchmark intentionally uses native Transformers "
            "image-text-to-text models only."
        )

    model = PipelineVLM(
        model_path=local,
        dtype=spec.get("dtype", "bfloat16"),
        trust_remote_code=bool(spec.get("trust_remote_code", False)),
        max_new_tokens=int(spec.get("max_new_tokens", 96)),
    )
    return model, local
