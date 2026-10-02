from __future__ import annotations

from pathlib import Path

import torch
from transformers import pipeline

from .base import VLMAdapter


def _dtype(name):
    name = str(name).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    return torch.float32


def _extract_text(out):
    item = out[0] if isinstance(out, list) else out
    text = item.get("generated_text", item) if isinstance(item, dict) else item
    if isinstance(text, list):
        for msg in reversed(text):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, list):
                    return "".join(
                        x.get("text", "") for x in content if isinstance(x, dict)
                    ).strip()
                return str(content).strip()
    return str(text).strip()


class PipelineVLM(VLMAdapter):
    """
    Adapter for native Transformers image-text-to-text models.

    Transformers >=4.57 rejects passing a conversational chat in `text=`
    while also passing `images=` separately. The image is therefore embedded
    directly inside the chat content, matching the current Transformers API.
    """

    def __init__(
        self,
        model_path,
        dtype="bfloat16",
        trust_remote_code=True,
        max_new_tokens=96,
        enable_thinking=None,
    ):
        self.model_path = str(model_path)
        self.max_new_tokens = int(max_new_tokens)
        self.enable_thinking = enable_thinking
        self.pipe = pipeline(
            "image-text-to-text",
            model=self.model_path,
            dtype=_dtype(dtype),
            device_map="auto",
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        )

    def generate(self, image_path, prompt, max_new_tokens=None):
        content = []
        if image_path is not None:
            image_path = str(Path(image_path).resolve())
            content.append({"type": "image", "url": image_path})
        content.append({"type": "text", "text": prompt})
        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]

        call_kwargs = {}
        if self.enable_thinking is not None:
            # Transformers forwards preprocessing kwargs to apply_chat_template.
            # Qwen and GLM templates use this switch to start the final answer
            # directly, without spending the short answer budget on reasoning.
            call_kwargs["enable_thinking"] = self.enable_thinking
        out = self.pipe(
            text=messages,
            generate_kwargs={
                "max_new_tokens": max_new_tokens or self.max_new_tokens,
                "do_sample": False,
            },
            return_full_text=False,
            **call_kwargs,
        )
        return _extract_text(out)

    def score_choices(self, image_path, prompt, choices):
        # Choice likelihood extraction differs across the VLM families used in
        # this benchmark. Until a family-specific scorer is validated, return
        # None so the benchmark uses its pre-specified decision-mode fallback.
        return None
