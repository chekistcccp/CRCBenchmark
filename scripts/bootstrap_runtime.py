#!/usr/bin/env python3
from __future__ import annotations

import importlib.metadata
import os
import subprocess
import sys

TARGET = "5.17.0"


def installed_version():
    try:
        return importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        return None


def main():
    current = installed_version()
    print(f"[runtime] transformers installed: {current}")
    print(f"[runtime] transformers required : {TARGET}")

    if current == TARGET:
        print("[runtime] OK")
        return

    auto_fix = os.environ.get("CRCBENCH_AUTO_FIX_RUNTIME", "1") == "1"
    if not auto_fix:
        raise SystemExit(
            f"Transformers {current!r} is incompatible with the frozen benchmark runtime. "
            f"Install transformers=={TARGET}."
        )

    print(
        "[runtime] Adjusting Transformers automatically. "
        "This aligns the runtime with the Qwen3.5/Qwen3.6 native multimodal model stack."
    )
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        f"transformers=={TARGET}",
    ]
    print("[runtime] " + " ".join(cmd))
    subprocess.check_call(cmd)

    new = installed_version()
    if new != TARGET:
        raise SystemExit(
            f"Runtime bootstrap completed but Transformers is {new!r}, expected {TARGET}."
        )
    print(f"[runtime] Ready: transformers=={new}")


if __name__ == "__main__":
    main()
