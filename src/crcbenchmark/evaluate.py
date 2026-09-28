from __future__ import annotations
from collections import defaultdict
from math import comb
import numpy as np
from PIL import Image
from .metrics import hit_at_k, recall_at_k, reciprocal_rank, iou_xyxy, pointing_hit, binary_prf, normalized_distance
from .utils import parse_choice_response, parse_label_list_response, parse_t2_response


def _pred_map(rows): return {r["item_id"]:r for r in rows}
def _labels_from_parsed(pred, allowed="ABCDEFGHIJKL", max_items=None):
    return parse_label_list_response(pred.get("raw_response", ""), allowed, max_items) or []

def eval_t1(item,pred):
    ranked=_labels_from_parsed(pred,max_items=5); gt=item["gt"]
    if gt["negative_only"]:
        none=ranked==["NONE"]
        return {"none_specificity":float(none),"invalid":float(not ranked and not none)}
    if ranked==["NONE"]: ranked=[]
    positives=set(gt["positive_labels"])
    n = 12
    return {"hit_1":hit_at_k(ranked,positives,1),"hit_3":hit_at_k(ranked,positives,3),"hit_5":hit_at_k(ranked,positives,5),"recall_1":recall_at_k(ranked,positives,1),"recall_3":recall_at_k(ranked,positives,3),"recall_5":recall_at_k(ranked,positives,5),"mrr":reciprocal_rank(ranked,positives),"invalid":float(len(ranked)==0),"random_hit_3":1-comb(n-len(positives),3)/comb(n,3),"random_recall_3":3/n}

def eval_t2(item,pred):
    gt_mask=np.asarray(Image.open(item["gt_mask_path"]))>0; h,w=gt_mask.shape
    baseline={"uniform_pointing":float(gt_mask.mean()),"center_pointing":float(gt_mask[h//2,w//2])}
    p=parse_t2_response(pred.get("raw_response",""))
    if p is None: return {"pointing_acc":0.0,"box_iou":0.0,"norm_distance":1.0,"invalid":1.0,**baseline}
    point=p.get("point"); box=p.get("box")
    valid_point=isinstance(point,list) and len(point)==2 and all(type(x) in (int,float) and np.isfinite(x) and 0<=x<=1000 for x in point)
    valid_box=isinstance(box,list) and len(box)==4 and all(type(x) in (int,float) and np.isfinite(x) and 0<=x<=1000 for x in box)
    if not valid_point: point=[float("nan"),float("nan")]
    if not valid_box: box=[0,0,0,0]
    point_px=[float(point[0])/1000*(w-1),float(point[1])/1000*(h-1)]
    gp=item["gt"]["point_norm"]; gp=[gp[0]/1000*(w-1),gp[1]/1000*(h-1)]
    return {"pointing_acc":pointing_hit(point_px,gt_mask) if valid_point else 0.0,"box_iou":iou_xyxy([float(x) for x in box],[float(x) for x in item["gt"]["box_norm"]]),"norm_distance":normalized_distance(point_px,gp,w,h) if valid_point else 1.0,"invalid":float(not (valid_point and valid_box)),**baseline}

def eval_t3(item,pred):
    labels=set(_labels_from_parsed(pred,allowed="ABCDEFGHI")); labels.discard("NONE"); truth=set(item["gt"]["positive_labels"]); p,r,f1=binary_prf(labels,truth); mapping=item["gt"]["slice_labels"]; boundary=int(item["gt"]["boundary_slice"])
    _,_,all_f1=binary_prf(set(mapping),truth)
    sl=[mapping[x] for x in labels if x in mapping]; err=abs((min(sl) if item["gt"]["side"]=="entry" else max(sl))-boundary) if sl else len(mapping)
    spacing=item["gt"].get("spacing_z_mm")
    boundary_error_mm=float(err)*float(spacing) if spacing is not None else None
    center_label=sorted(mapping)[(len(mapping)-1)//2]
    _,_,center_f1=binary_prf({center_label},truth)
    all_boundary=abs((min(mapping.values()) if item["gt"]["side"]=="entry" else max(mapping.values()))-boundary)
    return {"slice_precision":p,"slice_recall":r,"slice_f1":f1,"all_slices_f1":all_f1,"center_e_f1":center_f1,"boundary_error_slices":float(err),"all_slices_boundary_error_slices":float(all_boundary),"center_e_boundary_error_slices":float(abs(mapping[center_label]-boundary)),"boundary_error_mm":boundary_error_mm,"invalid":float(len(labels)==0)}

def eval_t4(item,pred):
    choice=parse_choice_response(pred.get("raw_response",""),item["choices"])
    return {"pairwise_acc":float(choice==item["gt"]["answer"]),"invalid":float(choice is None),"choice":choice,"swap":item["gt"].get("swap"),"always_a_acc":float(item["gt"]["answer"]=="A")}

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
            if cond not in bc: return float("nan")
            item,pred=bc[cond]; s=pred.get("choice_scores") or {}
            if pred.get("score_mode")=="likelihood" and "PRESENT" in s:
                value=float(s["PRESENT"])
                return value if np.isfinite(value) else float("nan")
            choice=parse_choice_response(pred.get("raw_response",""),item["choices"])
            return float(choice=="PRESENT") if choice is not None else float("nan")
        p0=pp("original"); pl=pp("lesion_gaussian") if "lesion_gaussian" in bc else float("nan"); pc=pp("control_gaussian") if "control_gaussian" in bc else float("nan")
        lm=pp("lesion_median") if "lesion_median" in bc else float("nan"); cm=pp("control_median") if "control_median" in bc else float("nan")
        fracs=[(float(c.rsplit("_",1)[1]),pp(c)) for c in sorted([c for c in bc if c.startswith("lesion_frac_")],key=lambda x:float(x.rsplit("_",1)[1]))]
        seq=[p0]+[p for _,p in fracs]; mono=np.mean([float(a>=b) for a,b in zip(seq,seq[1:])]) if len(seq)>1 and all(np.isfinite(seq)) else float("nan")
        fi,fp=bc["original"]
        valid_fraction=float(np.mean([np.isfinite(pp(c)) for c in bc]))
        original_present=float(p0>=0.5) if np.isfinite(p0) else float("nan")
        likelihood=fp.get("score_mode")=="likelihood"
        interpretable=likelihood or (np.isfinite(p0) and p0==1.0)
        if not interpretable:
            mono=float("nan")
        out.append({"item_id":gid,"track":"t5","dataset":fi["dataset"],"case_id":fi["case_id"],"p_original":p0,"p_lesion_gaussian":pl,"p_control_gaussian":pc,"original_present":original_present,"valid_fraction":valid_fraction,"delta_lesion":p0-pl if interpretable else float("nan"),"delta_control":p0-pc if interpretable else float("nan"),"faithfulness_gap":pc-pl if likelihood else float("nan"),"decision_gap":pc-pl if interpretable and not likelihood else float("nan"),"faithfulness_gap_median":cm-lm if likelihood and np.isfinite(lm) and np.isfinite(cm) else float("nan"),"monotonicity":float(mono) if likelihood else float("nan"),"decision_monotonicity":float(mono) if not likelihood else float("nan"),"score_mode":fp.get("score_mode")})
    return out

def patient_aggregate(rows):
    groups=defaultdict(list)
    for r in rows: groups[(r["track"],r["dataset"],r["case_id"])].append(r)
    out=[]
    for (track,dataset,case_id),rs in groups.items():
        agg={"track":track,"dataset":dataset,"case_id":case_id,"n_items":len(rs)}
        keys=sorted({k for r in rs for k,v in r.items() if isinstance(v,(int,float)) and k not in {"latency_s","swap"}})
        for k in keys:
            vals=[float(r[k]) for r in rs if isinstance(r.get(k),(int,float)) and np.isfinite(float(r[k]))]
            if vals: agg[k]=float(np.mean(vals))
        l=[float(r["latency_s"]) for r in rs if isinstance(r.get("latency_s"),(int,float))]
        if l: agg["latency_s"]=float(np.mean(l))
        if track=="t4":
            by_swap={r.get("swap"):r for r in rs}
            pair_complete=set(by_swap)=={0,1}
            pair_valid=pair_complete and all(by_swap[s].get("choice") in {"A","B"} for s in (0,1))
            agg["pair_valid"]=float(pair_valid)
            agg["pair_both_correct"]=float(pair_valid and all(by_swap[s]["pairwise_acc"]==1 for s in (0,1)))
            agg["swap_consistency"]=float(by_swap[0]["choice"]!=by_swap[1]["choice"]) if pair_valid else float("nan")
            valid_choices=[r["choice"] for r in rs if r.get("choice") in {"A","B"}]
            agg["selected_a_fraction"]=float(np.mean([c=="A" for c in valid_choices])) if valid_choices else float("nan")
        out.append(agg)
    return out
