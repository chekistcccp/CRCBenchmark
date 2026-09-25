#!/usr/bin/env python3
from __future__ import annotations

import importlib.metadata
import os
import subprocess
import sys

TARGET = "4.57.6"


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
        "This fixes Transformers 5.x incompatibility with InternVL/MiniCPM remote-code models."
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
