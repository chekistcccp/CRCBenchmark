#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from crcbenchmark.metrics import recall_at_k, iou_xyxy, binary_prf
from crcbenchmark.perturb import nested_fraction_mask
assert recall_at_k(["A","B"], {"B"}, 1)==0
assert recall_at_k(["A","B"], {"B"}, 2)==1
assert abs(iou_xyxy([0,0,10,10],[0,0,10,10])-1)<1e-9
m=np.zeros((16,16),bool); m[4:12,4:12]=1
assert nested_fraction_mask(m,.5).sum() <= m.sum()
print("smoke test passed")
