#!/usr/bin/env python3
from __future__ import annotations

import importlib.metadata
import os
import subprocess
import sys


TORCH_TARGET = "2.13.0"
TORCHVISION_TARGET = "0.28.0"
TORCH_INDEX = "https://download.pytorch.org/whl/cu126"
TRANSFORMERS_TARGET = "5.17.0"


def version(pkg: str):
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return None


def run(cmd):
    print("[runtime] " + " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    auto_fix = os.environ.get("CRCBENCH_AUTO_FIX_RUNTIME", "1") == "1"

    torch_v = version("torch")
    vision_v = version("torchvision")
    tf_v = version("transformers")

    print("[runtime] required environment")
    print(f"  torch        : {TORCH_TARGET}")
    print(f"  torchvision  : {TORCHVISION_TARGET}")
    print(f"  torch index  : {TORCH_INDEX}")
    print(f"  transformers : {TRANSFORMERS_TARGET}")
    print("[runtime] installed environment")
    print(f"  torch        : {torch_v}")
    print(f"  torchvision  : {vision_v}")
    print(f"  transformers : {tf_v}")

    torch_ok = torch_v == TORCH_TARGET and vision_v == TORCHVISION_TARGET

    if not torch_ok:
        if not auto_fix:
            raise SystemExit(
                "PyTorch runtime mismatch. Install exactly:\n"
                f"pip install torch=={TORCH_TARGET} torchvision=={TORCHVISION_TARGET} "
                f"--index-url {TORCH_INDEX}"
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
            "--index-url",
            TORCH_INDEX,
        ])

    if version("transformers") != TRANSFORMERS_TARGET:
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
    final_tf = version("transformers")

    if final_torch != TORCH_TARGET:
        raise SystemExit(
            f"torch is {final_torch!r}, expected {TORCH_TARGET!r}."
        )
    if final_vision != TORCHVISION_TARGET:
        raise SystemExit(
            f"torchvision is {final_vision!r}, expected {TORCHVISION_TARGET!r}."
        )
    if final_tf != TRANSFORMERS_TARGET:
        raise SystemExit(
            f"transformers is {final_tf!r}, expected {TRANSFORMERS_TARGET!r}."
        )

    print("[runtime] Ready")
    print(f"  torch        = {final_torch}")
    print(f"  torchvision  = {final_vision}")
    print(f"  transformers = {final_tf}")


if __name__ == "__main__":
    main()
