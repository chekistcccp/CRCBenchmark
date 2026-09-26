#!/usr/bin/env python3
from __future__ import annotations

import importlib.metadata
import os
import subprocess
import sys

from packaging.version import Version


TORCH_TARGET = "2.13.0"
TORCHVISION_TARGET = "0.28.0"
TORCHAUDIO_TARGET = "2.13.0"
TORCH_INDEX = "https://download.pytorch.org/whl/cu126"
CUDA_TARGET = "12.6"
TRANSFORMERS_TARGET = "5.17.0"


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


def run(cmd):
    print("[runtime] " + " ".join(cmd))
    subprocess.check_call(cmd)


def detect_cuda_runtime():
    try:
        import torch
        return torch.version.cuda
    except Exception:
        return None


def main():
    auto_fix = os.environ.get("CRCBENCH_AUTO_FIX_RUNTIME", "1") == "1"

    torch_v = version("torch")
    vision_v = version("torchvision")
    audio_v = version("torchaudio")
    tf_v = version("transformers")
    cuda_v = detect_cuda_runtime()

    print("[runtime] required environment")
    print(f"  torch base       : {TORCH_TARGET}")
    print(f"  torchvision base : {TORCHVISION_TARGET}")
    print(f"  torchaudio base  : {TORCHAUDIO_TARGET}")
    print(f"  CUDA runtime     : {CUDA_TARGET}")
    print(f"  torch index      : {TORCH_INDEX}")
    print(f"  transformers     : {TRANSFORMERS_TARGET}")

    print("[runtime] installed environment")
    print(f"  torch            : {torch_v}")
    print(f"  torch base       : {base_version(torch_v)}")
    print(f"  torchvision      : {vision_v}")
    print(f"  torchvision base : {base_version(vision_v)}")
    print(f"  torchaudio       : {audio_v}")
    print(f"  torchaudio base  : {base_version(audio_v)}")
    print(f"  CUDA runtime     : {cuda_v}")
    print(f"  transformers     : {tf_v}")

    torch_ok = (
        base_version(torch_v) == TORCH_TARGET
        and base_version(vision_v) == TORCHVISION_TARGET
        and base_version(audio_v) == TORCHAUDIO_TARGET
        and cuda_v == CUDA_TARGET
    )

    if not torch_ok:
        if not auto_fix:
            raise SystemExit(
                "PyTorch runtime mismatch. Install exactly:\n"
                f"pip install torch=={TORCH_TARGET} torchvision=={TORCHVISION_TARGET} "
                f"torchaudio=={TORCHAUDIO_TARGET} --index-url {TORCH_INDEX}"
            )

        print("[runtime] Installing the frozen CUDA 12.6 PyTorch runtime...")
        run([
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            f"torch=={TORCH_TARGET}",
            f"torchvision=={TORCHVISION_TARGET}",
            f"torchaudio=={TORCHAUDIO_TARGET}",
            "--index-url",
            TORCH_INDEX,
        ])

    if base_version(version("transformers")) != TRANSFORMERS_TARGET:
        if not auto_fix:
            raise SystemExit(
                f"Transformers runtime mismatch. Install transformers=={TRANSFORMERS_TARGET}."
            )

        print("[runtime] Installing the frozen Transformers runtime...")
        run([
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            f"transformers=={TRANSFORMERS_TARGET}",
        ])

    final_torch = version("torch")
    final_vision = version("torchvision")
    final_audio = version("torchaudio")
    final_tf = version("transformers")
    final_cuda = detect_cuda_runtime()

    if base_version(final_torch) != TORCH_TARGET:
        raise SystemExit(
            f"torch is {final_torch!r} (base={base_version(final_torch)!r}), "
            f"expected base {TORCH_TARGET!r}."
        )
    if base_version(final_vision) != TORCHVISION_TARGET:
        raise SystemExit(
            f"torchvision is {final_vision!r} (base={base_version(final_vision)!r}), "
            f"expected base {TORCHVISION_TARGET!r}."
        )
    if base_version(final_audio) != TORCHAUDIO_TARGET:
        raise SystemExit(
            f"torchaudio is {final_audio!r} (base={base_version(final_audio)!r}), "
            f"expected base {TORCHAUDIO_TARGET!r}."
        )
    if final_cuda != CUDA_TARGET:
        raise SystemExit(
            f"torch CUDA runtime is {final_cuda!r}, expected {CUDA_TARGET!r} (cu126)."
        )
    if base_version(final_tf) != TRANSFORMERS_TARGET:
        raise SystemExit(
            f"transformers is {final_tf!r}, expected {TRANSFORMERS_TARGET!r}."
        )

    print("[runtime] Ready")
    print(f"  torch        = {final_torch}")
    print(f"  torchvision  = {final_vision}")
    print(f"  torchaudio   = {final_audio}")
    print(f"  CUDA runtime = {final_cuda}")
    print(f"  transformers = {final_tf}")


if __name__ == "__main__":
    main()
