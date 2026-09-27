import numpy as np
from crcbenchmark.metrics import hit_at_k,recall_at_k,reciprocal_rank,iou_xyxy,binary_prf
from crcbenchmark.perturb import nested_fraction_mask

def test_retrieval_metrics():
    ranked=["C","A","B"]; positives={"B"}
    assert recall_at_k(ranked,positives,2)==0 and recall_at_k(ranked,positives,3)==1 and reciprocal_rank(ranked,positives)==1/3
    assert recall_at_k(["A","C","B"], {"A","B","D"}, 2)==1/3
    assert hit_at_k(["A","C","B"], {"A","B","D"}, 2)==1

def test_iou():
    assert iou_xyxy([0,0,10,10],[0,0,10,10])==1
    assert iou_xyxy([0,0,10,10],[10,10,20,20])==0

def test_binary_prf():
    p,r,f=binary_prf({"A","B"},{"B","C"}); assert p==0.5 and r==0.5 and f==0.5

def test_nested_masks():
    m=np.zeros((32,32),bool); m[8:24,8:24]=1
    a=nested_fraction_mask(m,0.25); b=nested_fraction_mask(m,0.75)
    assert a.sum()<=b.sum()<=m.sum()


def test_t3_missing_spacing():
    from crcbenchmark.evaluate import eval_t3

    item = {
        "gt": {
            "positive_labels": ["B", "C"],
            "slice_labels": {"A": 10, "B": 11, "C": 12, "D": 13},
            "boundary_slice": 11,
            "side": "entry",
            "spacing_z_mm": None,
        }
    }
    pred = {"parsed": ["B", "C"], "raw_response": '["B","C"]'}
    out = eval_t3(item, pred)
    assert out["boundary_error_slices"] == 0.0
    assert out["boundary_error_mm"] is None


def test_t3_with_spacing():
    from crcbenchmark.evaluate import eval_t3

    item = {
        "gt": {
            "positive_labels": ["B", "C"],
            "slice_labels": {"A": 10, "B": 11, "C": 12, "D": 13},
            "boundary_slice": 11,
            "side": "entry",
            "spacing_z_mm": 2.5,
        }
    }
    pred = {"parsed": ["C"], "raw_response": '["C"]'}
    out = eval_t3(item, pred)
    assert out["boundary_error_slices"] == 1.0
    assert out["boundary_error_mm"] == 2.5
    assert out["all_slices_f1"] == 2 / 3
