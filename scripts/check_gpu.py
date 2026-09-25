#!/usr/bin/env python3
from __future__ import annotations

import sys

try:
    import torch
except Exception as e:
    raise SystemExit(f"PyTorch import failed: {e}")

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit("No CUDA GPU is visible. Check driver/CUDA/CUDA_VISIBLE_DEVICES.")

count = torch.cuda.device_count()
print("Visible GPU count:", count)

for i in range(count):
    p = torch.cuda.get_device_properties(i)
    total_gib = p.total_memory / (1024 ** 3)
    print(f"GPU {i}: {p.name}")
    print(f"  memory: {total_gib:.1f} GiB")
    print(f"  compute capability: {p.major}.{p.minor}")

print("BF16 supported:", torch.cuda.is_bf16_supported())

if count > 1:
    print("NOTE: ColoGround-Bench primary configuration is single-GPU by default.")
    print("      Set NUM_GPUS explicitly only when you intentionally want data-parallel sharding.")

if not torch.cuda.is_bf16_supported():
    print("WARNING: current model configs use bfloat16; use a BF16-capable GPU or adjust dtype.")
