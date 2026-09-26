#!/usr/bin/env python3
from __future__ import annotations

import importlib.metadata
import sys

from packaging.version import Version


TORCH_TARGET = "2.13.0"
TORCHVISION_RECOMMENDED = "0.28.0"
CUDA_TARGET = "12.6"


def version(pkg: str):
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return None


def base_version(v: str | None):
    if v is None:
        return None
    try:
        return Version(v).base_version
    except Exception:
        return str(v).split("+", 1)[0]


def detect_cuda_runtime():
    try:
        import torch
        return torch.version.cuda
    except Exception:
        return None


def main():
    torch_v = version("torch")
    vision_v = version("torchvision")
    tf_v = version("transformers")
    accelerate_v = version("accelerate")
    modelscope_v = version("modelscope")
    cuda_v = detect_cuda_runtime()

    print("[runtime] frozen PyTorch runtime")
    print(f"  torch required       : {TORCH_TARGET}")
    print(f"  CUDA runtime required: {CUDA_TARGET}")
    print(f"  torchvision expected : {TORCHVISION_RECOMMENDED}")

    print("[runtime] installed")
    print(f"  torch        : {torch_v}")
    print(f"  torch base   : {base_version(torch_v)}")
    print(f"  torchvision  : {vision_v}")
    print(f"  CUDA runtime : {cuda_v}")
    print(f"  transformers : {tf_v}")
    print(f"  accelerate   : {accelerate_v}")
    print(f"  modelscope   : {modelscope_v}")

    if base_version(torch_v) != TORCH_TARGET:
        raise SystemExit(
            f"torch is {torch_v!r}; expected base version {TORCH_TARGET!r}.\n"
            "Install with:\n"
            "pip install torch==2.13.0 torchvision==0.28.0 "
            "--index-url https://download.pytorch.org/whl/cu126"
        )

    if cuda_v != CUDA_TARGET:
        raise SystemExit(
            f"torch CUDA runtime is {cuda_v!r}; expected {CUDA_TARGET!r} (cu126)."
        )

    if vision_v is None:
        raise SystemExit(
            "torchvision is not installed. Recommended companion build:\n"
            "pip install torchvision==0.28.0 "
            "--index-url https://download.pytorch.org/whl/cu126"
        )

    # Do not force-install or downgrade any non-PyTorch package here.
    # The model stack is intentionally resolved by requirements.txt.
    try:
        import transformers  # noqa: F401
        import accelerate  # noqa: F401
        import modelscope  # noqa: F401
    except Exception as e:
        raise SystemExit(
            "The flexible model stack is installed but cannot be imported. "
            "Run: pip install -U -r requirements.txt\n"
            f"Original import error: {e}"
        )

    print("[runtime] Ready")
    print("  PyTorch/CUDA is frozen; other dependencies remain flexibly resolved.")


if __name__ == "__main__":
    main()
