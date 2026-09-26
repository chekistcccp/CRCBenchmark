#!/usr/bin/env python3
from __future__ import annotations

import importlib.metadata
import subprocess
import sys

from packaging.version import Version


TORCH_TARGET = "2.13.0"
TORCHVISION_RECOMMENDED = "0.28.0"
TORCHAUDIO_MATCH = "2.13.0"
CUDA_TARGET = "12.6"
TORCH_INDEX = "https://download.pytorch.org/whl/cu126"


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


def run(cmd):
    print("[runtime] " + " ".join(cmd))
    subprocess.check_call(cmd)


def torchaudio_import_ok() -> tuple[bool, str]:
    """
    Test torchaudio in a clean subprocess. This avoids poisoning the current
    interpreter with a partially imported binary module when CUDA ABIs differ.
    """
    if version("torchaudio") is None:
        return True, "not installed"

    code = (
        "import torch, torchaudio; "
        "print('torch=' + str(torch.__version__)); "
        "print('torchaudio=' + str(torchaudio.__version__)); "
        "print('cuda=' + str(torch.version.cuda))"
    )
    p = subprocess.run(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return p.returncode == 0, p.stdout.strip()


def repair_torchaudio_if_needed():
    audio_v = version("torchaudio")
    if audio_v is None:
        print("[runtime] torchaudio: not installed (OK; benchmark is image-only)")
        return

    ok, detail = torchaudio_import_ok()
    if ok and base_version(audio_v) == TORCHAUDIO_MATCH:
        print(f"[runtime] torchaudio: {audio_v} (compatible)")
        return

    print("[runtime] torchaudio is installed but incompatible with frozen PyTorch.")
    if detail:
        print("[runtime] torchaudio import check:")
        for line in detail.splitlines():
            print("  " + line)

    print(
        "[runtime] Repairing ONLY torchaudio; torch/torchvision will not be changed."
    )
    run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "--force-reinstall",
        "--no-deps",
        f"torchaudio=={TORCHAUDIO_MATCH}",
        "--index-url",
        TORCH_INDEX,
    ])

    ok, detail = torchaudio_import_ok()
    if not ok:
        raise SystemExit(
            "torchaudio still cannot be imported after cu126 repair.\n" + detail
        )
    print("[runtime] torchaudio repaired successfully:")
    for line in detail.splitlines():
        print("  " + line)


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
    print(f"  torchaudio   : {version('torchaudio')}")
    print(f"  CUDA runtime : {cuda_v}")
    print(f"  transformers : {tf_v}")
    print(f"  accelerate   : {accelerate_v}")
    print(f"  modelscope   : {modelscope_v}")

    # PyTorch is the frozen dependency. Never modify it here.
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
            "torchvision is not installed. Install the requested companion build:\n"
            "pip install torchvision==0.28.0 "
            "--index-url https://download.pytorch.org/whl/cu126"
        )

    # Other dependencies are allowed to adapt around the frozen PyTorch runtime.
    repair_torchaudio_if_needed()

    try:
        import transformers  # noqa: F401
        import accelerate  # noqa: F401
        import modelscope  # noqa: F401
        from transformers import pipeline  # noqa: F401
    except Exception as e:
        raise SystemExit(
            "The flexible model stack cannot import successfully while keeping "
            "the frozen PyTorch runtime. Run:\n"
            "pip install -U -r requirements.txt\n"
            f"Original import error: {e}"
        )

    print("[runtime] Transformers pipeline import: OK")
    print("[runtime] Ready")
    print("  PyTorch/CUDA is frozen; compatible non-PyTorch dependencies may be adjusted.")


if __name__ == "__main__":
    main()
