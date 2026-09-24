import numpy as np
from crcbenchmark.metrics import recall_at_k,reciprocal_rank,iou_xyxy,binary_prf
from crcbenchmark.perturb import nested_fraction_mask

def test_retrieval_metrics():
    ranked=["C","A","B"]; positives={"B"}
    assert recall_at_k(ranked,positives,2)==0 and recall_at_k(ranked,positives,3)==1 and reciprocal_rank(ranked,positives)==1/3

def test_iou():
    assert iou_xyxy([0,0,10,10],[0,0,10,10])==1
    assert iou_xyxy([0,0,10,10],[10,10,20,20])==0

def test_binary_prf():
    p,r,f=binary_prf({"A","B"},{"B","C"}); assert p==0.5 and r==0.5 and f==0.5

def test_nested_masks():
    m=np.zeros((32,32),bool); m[8:24,8:24]=1
    a=nested_fraction_mask(m,0.25); b=nested_fraction_mask(m,0.75)
    assert a.sum()<=b.sum()<=m.sum()
