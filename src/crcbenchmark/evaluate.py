from __future__ import annotations
from collections import defaultdict
import numpy as np
from PIL import Image
from .metrics import recall_at_k, reciprocal_rank, iou_xyxy, pointing_hit, binary_prf, normalized_distance


def _pred_map(rows): return {r["item_id"]:r for r in rows}
def _labels_from_parsed(pred):
    p=pred.get("parsed")
    if isinstance(p,list): return [str(x).upper() for x in p]
    raw=str(pred.get("raw_response","")).upper()
    return [c for c in "ABCDEFGHIJKL" if c in raw]

def eval_t1(item,pred):
    ranked=_labels_from_parsed(pred); gt=item["gt"]
    if gt["negative_only"]:
        none="NONE" in str(pred.get("raw_response","")).upper() or (isinstance(pred.get("parsed"),list) and "NONE" in [str(x).upper() for x in pred["parsed"]])
        return {"none_specificity":float(none),"invalid":float(not ranked and not none)}
    positives=set(gt["positive_labels"])
    return {"recall_1":recall_at_k(ranked,positives,1),"recall_3":recall_at_k(ranked,positives,3),"recall_5":recall_at_k(ranked,positives,5),"mrr":reciprocal_rank(ranked,positives),"invalid":float(len(ranked)==0)}

def eval_t2(item,pred):
    p=pred.get("parsed")
    if not isinstance(p,dict): return {"pointing_acc":0.0,"box_iou":0.0,"norm_distance":1.0,"invalid":1.0}
    point=p.get("point"); box=p.get("box")
    if not (isinstance(point,list) and len(point)==2): point=[float("nan"),float("nan")]
    if not (isinstance(box,list) and len(box)==4): box=[0,0,0,0]
    gt_mask=np.asarray(Image.open(item["gt_mask_path"]))>0; h,w=gt_mask.shape
    point_px=[float(point[0])/1000*(w-1),float(point[1])/1000*(h-1)]
    gp=item["gt"]["point_norm"]; gp=[gp[0]/1000*(w-1),gp[1]/1000*(h-1)]
    return {"pointing_acc":pointing_hit(point_px,gt_mask),"box_iou":iou_xyxy([float(x) for x in box],[float(x) for x in item["gt"]["box_norm"]]),"norm_distance":normalized_distance(point_px,gp,w,h),"invalid":float(any(not np.isfinite(float(x)) for x in point))}

def eval_t3(item,pred):
    labels=set(_labels_from_parsed(pred)); truth=set(item["gt"]["positive_labels"]); p,r,f1=binary_prf(labels,truth); mapping=item["gt"]["slice_labels"]; boundary=int(item["gt"]["boundary_slice"])
    sl=[mapping[x] for x in labels if x in mapping]; err=abs((min(sl) if item["gt"]["side"]=="entry" else max(sl))-boundary) if sl else len(mapping)
    return {"slice_precision":p,"slice_recall":r,"slice_f1":f1,"boundary_error_slices":float(err),"boundary_error_mm":float(err)*float(item["gt"]["spacing_z_mm"]),"invalid":float(len(labels)==0)}

def eval_t4(item,pred): return {"pairwise_acc":float(pred.get("choice")==item["gt"]["answer"]),"invalid":float(pred.get("choice") is None)}

def eval_items(manifest,preds):
    pm=_pred_map(preds); out=[]
    for item in manifest:
        pred=pm.get(item["item_id"])
        if pred is None or item["track"]=="t5": continue
        metrics={"t1":eval_t1,"t2":eval_t2,"t3":eval_t3,"t4":eval_t4}[item["track"]](item,pred)
        out.append({"item_id":item["item_id"],"track":item["track"],"dataset":item["dataset"],"case_id":item["case_id"],**metrics,"latency_s":pred.get("latency_s")})
    return out

def eval_t5_groups(manifest,preds):
    pm=_pred_map(preds); groups=defaultdict(list)
    for item in manifest:
        if item["track"]=="t5" and item["item_id"] in pm: groups[item["group_id"]].append((item,pm[item["item_id"]]))
    out=[]
    for gid,pairs in groups.items():
        bc={i["condition"]:(i,p) for i,p in pairs}
        if "original" not in bc: continue
        def pp(cond):
            pred=bc.get(cond,({},{}))[1]; s=pred.get("choice_scores") or {}
            return float(s["PRESENT"]) if "PRESENT" in s else float(pred.get("choice")=="PRESENT")
        p0=pp("original"); pl=pp("lesion_gaussian") if "lesion_gaussian" in bc else float("nan"); pc=pp("control_gaussian") if "control_gaussian" in bc else float("nan")
        lm=pp("lesion_median") if "lesion_median" in bc else float("nan"); cm=pp("control_median") if "control_median" in bc else float("nan")
        fracs=[(float(c.rsplit("_",1)[1]),pp(c)) for c in sorted([c for c in bc if c.startswith("lesion_frac_")],key=lambda x:float(x.rsplit("_",1)[1]))]
        seq=[p0]+[p for _,p in fracs]; mono=np.mean([float(a>=b) for a,b in zip(seq,seq[1:])]) if len(seq)>1 else float("nan")
        fi,fp=bc["original"]; out.append({"item_id":gid,"track":"t5","dataset":fi["dataset"],"case_id":fi["case_id"],"p_original":p0,"p_lesion_gaussian":pl,"p_control_gaussian":pc,"delta_lesion":p0-pl,"delta_control":p0-pc,"faithfulness_gap":pc-pl,"faithfulness_gap_median":cm-lm if np.isfinite(lm) and np.isfinite(cm) else float("nan"),"monotonicity":float(mono),"score_mode":fp.get("score_mode")})
    return out

def patient_aggregate(rows):
    groups=defaultdict(list)
    for r in rows: groups[(r["track"],r["dataset"],r["case_id"])].append(r)
    out=[]
    for (track,dataset,case_id),rs in groups.items():
        agg={"track":track,"dataset":dataset,"case_id":case_id,"n_items":len(rs)}
        keys=sorted({k for r in rs for k,v in r.items() if isinstance(v,(int,float)) and k!="latency_s"})
        for k in keys:
            vals=[float(r[k]) for r in rs if isinstance(r.get(k),(int,float)) and np.isfinite(float(r[k]))]
            if vals: agg[k]=float(np.mean(vals))
        l=[float(r["latency_s"]) for r in rs if isinstance(r.get("latency_s"),(int,float))]
        if l: agg["latency_s"]=float(np.mean(l))
        out.append(agg)
    return out
