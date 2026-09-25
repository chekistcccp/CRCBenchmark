from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
from .preprocess import VolumeCase,to_rgb_pil,make_montage,bbox_from_mask,centroid_from_mask,normalize_box,normalize_point,crop_center_physical,crop_center_pixels,body_mask_from_hu
from .perturb import gaussian_suppress,median_replace,choose_matched_control,nested_fraction_mask

LETTERS=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

def _save_image(path,image):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); image.save(path,quality=95); return str(path)
def _save_mask(path,mask):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); Image.fromarray(mask.astype(np.uint8)*255).save(path); return str(path)
def _neighbor_positives(pos,zmax,n=3):
    pos=[int(x) for x in pos]; chosen=[zmax]
    below=sorted([z for z in pos if z<zmax],key=lambda z:zmax-z); above=sorted([z for z in pos if z>zmax],key=lambda z:z-zmax)
    for pool in (below,above):
        if pool and len(chosen)<n: chosen.append(pool[0])
    for z in sorted(pos,key=lambda z:abs(z-zmax)):
        if z not in chosen and len(chosen)<n: chosen.append(z)
    return chosen

def build_t1(case,out_root,cfg,rng):
    tumor=case.tumor_mask(); areas=tumor.reshape(tumor.shape[0],-1).sum(1); pos=np.where(areas>0)[0]
    if len(pos)==0:return []
    zmax=int(np.argmax(areas)); npos=int(cfg["positives"]); nneg=int(cfg["negatives"]); pos_sel=_neighbor_positives(pos,zmax,npos); neg_all=np.where(areas==0)[0]
    if len(neg_all)<nneg:return []
    neg_sorted=sorted([int(z) for z in neg_all],key=lambda z:min(abs(z-int(pos.min())),abs(z-int(pos.max())))); pool=neg_sorted[:max(nneg*4,nneg)]
    neg_sel=list(rng.choice(pool,size=nneg,replace=False)) if len(pool)>=nneg else neg_sorted[:nneg]
    base=[(int(z),True) for z in pos_sel]+[(int(z),False) for z in neg_sel]; rows=[]
    for perm in range(int(cfg["permutations"])):
        order=rng.permutation(len(base)).tolist(); shuffled=[base[i] for i in order]; ims=[to_rgb_pil(case.image[z],intensity_mode=case.intensity_mode,size=256) for z,_ in shuffled]; labs=LETTERS[:len(ims)]
        montage=make_montage(ims,labs,3,4,256); p=Path(out_root)/"t1"/case.dataset/case.case_id/f"positive_perm{perm}.jpg"; _save_image(p,montage)
        positives=[lab for lab,(_,is_pos) in zip(labs,shuffled) if is_pos]
        rows.append({"item_id":f"t1:{case.dataset}:{case.case_id}:pos:{perm}","track":"t1","dataset":case.dataset,"case_id":case.case_id,"image_path":str(p),"prompt":"Twelve axial CT slices from one patient are shown and labeled A-L. Rank up to five slices according to the likelihood that they contain the primary colorectal tumor. Return only a JSON list of labels from most to least suspicious. If no slice contains tumor, return [\"NONE\"].","gt":{"positive_labels":positives,"negative_only":False}})
    neg_sel=list(rng.choice(neg_all,size=npos+nneg,replace=False)); ims=[to_rgb_pil(case.image[int(z)],intensity_mode=case.intensity_mode,size=256) for z in neg_sel]; labs=LETTERS[:len(ims)]
    montage=make_montage(ims,labs,3,4,256); p=Path(out_root)/"t1"/case.dataset/case.case_id/"negative_only.jpg"; _save_image(p,montage)
    rows.append({"item_id":f"t1:{case.dataset}:{case.case_id}:neg","track":"t1","dataset":case.dataset,"case_id":case.case_id,"image_path":str(p),"prompt":"Twelve axial CT slices from one patient are shown and labeled A-L. Rank up to five slices according to the likelihood that they contain the primary colorectal tumor. Return only a JSON list of labels from most to least suspicious. If no slice contains tumor, return [\"NONE\"].","gt":{"positive_labels":[],"negative_only":True}})
    return rows

def build_t2(case,out_root,cfg):
    tumor=case.tumor_mask(); areas=tumor.reshape(tumor.shape[0],-1).sum(1); pos=np.where(areas>0)[0]
    if len(pos)==0:return []
    zmax=int(np.argmax(areas)); candidates=[zmax]
    if len(pos)>=3:candidates+=[int(pos[max(0,len(pos)//4)]),int(pos[min(len(pos)-1,(3*len(pos))//4)])]
    candidates=list(dict.fromkeys(candidates))[:int(cfg["max_slices_per_case"])]; rows=[]
    for z in candidates:
        raw=case.image[z]; m=tumor[z]; h,w=raw.shape; im=to_rgb_pil(raw,intensity_mode=case.intensity_mode,size=int(cfg["canvas_size"]))
        p=Path(out_root)/"t2"/case.dataset/case.case_id/f"z{z}.jpg"; _save_image(p,im); mp=Path(out_root)/"t2"/case.dataset/case.case_id/f"z{z}_mask.png"; _save_mask(mp,m)
        bbox=bbox_from_mask(m); centroid=centroid_from_mask(m)
        rows.append({"item_id":f"t2:{case.dataset}:{case.case_id}:{z}","track":"t2","dataset":case.dataset,"case_id":case.case_id,"slice_index":z,"image_path":str(p),"gt_mask_path":str(mp),"prompt":"Identify the single image location that provides the strongest visual evidence for the primary colorectal tumor. Return only JSON: {\"point\":[x,y],\"box\":[x1,y1,x2,y2]}, where all coordinates are normalized integers from 0 to 1000.","gt":{"point_norm":normalize_point(centroid,w,h),"box_norm":normalize_box(bbox,w,h),"shape_hw":[h,w]}})
    return rows

def _local_source_window(case, center_pos, radius):
    """Return array positions for a truly consecutive source-slice window."""
    n = case.image.shape[0]
    src = getattr(case, "source_slice_indices", getattr(case, "slice_indices", None))
    if src is None:
        positions = list(range(center_pos-radius, center_pos+radius+1))
        if min(positions) < 0 or max(positions) >= n:
            return None, None
        return positions, positions

    src = [int(x) for x in src]
    center_src = src[int(center_pos)]
    lookup = {z:i for i,z in enumerate(src)}
    wanted = list(range(center_src-radius, center_src+radius+1))
    if not all(z in lookup for z in wanted):
        return None, wanted
    return [lookup[z] for z in wanted], wanted


def build_t3(case,out_root,cfg):
    tumor=case.tumor_mask(); areas=tumor.reshape(tumor.shape[0],-1).sum(1); pos=np.where(areas>0)[0]
    if len(pos)==0:return []
    radius=int(cfg["radius"]); n=2*radius+1; rows=[]
    for side,center_pos in (("entry",int(pos.min())),("exit",int(pos.max()))):
        arr_positions,source_slices=_local_source_window(case,center_pos,radius)
        if arr_positions is None:
            continue
        ims=[to_rgb_pil(case.image[z],intensity_mode=case.intensity_mode,size=256) for z in arr_positions]
        labs=LETTERS[:n]; montage=make_montage(ims,labs,3,3,256)
        p=Path(out_root)/"t3"/case.dataset/case.case_id/f"{side}.jpg"; _save_image(p,montage)
        gt_labels=[lab for lab,z in zip(labs,arr_positions) if areas[z]>0]
        spacing_z=float(case.spacing_zyx[0])
        spacing_z=None if not np.isfinite(spacing_z) or spacing_z<=0 else spacing_z
        boundary_source=int(source_slices[radius])
        rows.append({
            "item_id":f"t3:{case.dataset}:{case.case_id}:{side}",
            "track":"t3","dataset":case.dataset,"case_id":case.case_id,
            "image_path":str(p),
            "prompt":"The nine images labeled A-I are truly consecutive axial CT slices. Identify all slices containing visible primary colorectal tumor. Return only a JSON list of labels, for example [\"E\",\"F\"].",
            "gt":{
                "positive_labels":gt_labels,
                "slice_labels":{lab:int(z) for lab,z in zip(labs,source_slices)},
                "boundary_slice":boundary_source,
                "spacing_z_mm":spacing_z,
                "side":side
            }
        })
    return rows


def build_t4(case,out_root,cfg,rng):
    if case.dataset.upper()!="CARE":return []
    tumor=case.tumor_mask(); normal=case.normal_mask()
    same=[z for z in range(case.image.shape[0]) if tumor[z].any() and normal[z].any()]
    if same:
        zt=zn=int(same[int(np.argmax([tumor[z].sum() for z in same]))])
    else:
        tz=np.where(tumor.reshape(tumor.shape[0],-1).sum(1)>0)[0]
        nz=np.where(normal.reshape(normal.shape[0],-1).sum(1)>0)[0]
        if len(tz)==0 or len(nz)==0:return []
        _,zt,zn=min(((abs(int(a)-int(b)),int(a),int(b)) for a in tz for b in nz),key=lambda x:x[0])

    tc=centroid_from_mask(tumor[zt]); nc=centroid_from_mask(normal[zn])
    if tc is None or nc is None:return []

    out_size=int(cfg["output_size"])
    dy,dx=float(case.spacing_zyx[1]),float(case.spacing_zyx[2])
    physical_spacing_known=np.isfinite(dy) and np.isfinite(dx) and dy>0 and dx>0
    if physical_spacing_known:
        fov=float(cfg["fov_mm"])
        ta=crop_center_physical(case.image[zt],tuple(tc),(dy,dx),fov)
        na=crop_center_physical(case.image[zn],tuple(nc),(dy,dx),fov)
        crop_mode="physical_mm"
        crop_value=fov
    else:
        crop_px=int(cfg.get("fallback_crop_px",160))
        ta=crop_center_pixels(case.image[zt],tuple(tc),crop_px)
        na=crop_center_pixels(case.image[zn],tuple(nc),crop_px)
        crop_mode="fixed_pixels"
        crop_value=crop_px

    ti=to_rgb_pil(ta,intensity_mode=case.intensity_mode,size=out_size)
    ni=to_rgb_pil(na,intensity_mode=case.intensity_mode,size=out_size)
    rows=[]
    src=getattr(case,"source_slice_indices",getattr(case,"slice_indices",None))
    source_zt=int(src[zt]) if src is not None else int(zt)
    source_zn=int(src[zn]) if src is not None else int(zn)
    for swap in (0,1):
        pair=[ti,ni] if swap==0 else [ni,ti]
        montage=make_montage(pair,["A","B"],1,2,out_size)
        p=Path(out_root)/"t4"/case.dataset/case.case_id/f"swap{swap}.jpg"; _save_image(p,montage)
        rows.append({
            "item_id":f"t4:{case.dataset}:{case.case_id}:{swap}",
            "track":"t4","dataset":case.dataset,"case_id":case.case_id,
            "image_path":str(p),
            "prompt":"Two rectal CT regions from the same patient are shown as A and B. Which region contains stronger visual evidence of malignant rectal involvement? Answer only A or B.",
            "choices":["A","B"],
            "gt":{
                "answer":"A" if swap==0 else "B",
                "delta_source_slice":int(abs(source_zt-source_zn)),
                "swap":swap,
                "crop_mode":crop_mode,
                "crop_value":crop_value
            }
        })
    return rows

def build_t5(case,out_root,cfg,rng):
    tumor=case.tumor_mask(); areas=tumor.reshape(tumor.shape[0],-1).sum(1)
    if areas.max()<=0:return []
    z=int(np.argmax(areas)); image=case.image[z].astype(np.float32); lesion=tumor[z]; forbidden=binary_dilation(lesion,iterations=5)
    allowed=case.normal_mask()[z] if case.dataset.upper()=="CARE" and case.normal_mask()[z].any() else (body_mask_from_hu(image) if case.intensity_mode=="hu" else np.ones_like(lesion,bool))
    control=choose_matched_control(image,lesion,allowed,forbidden,int(cfg["control_trials"]),rng)
    if control is None:return []
    conditions=[("original",image),("lesion_gaussian",gaussian_suppress(image,lesion,float(cfg["gaussian_sigma"]))),("control_gaussian",gaussian_suppress(image,control,float(cfg["gaussian_sigma"])))]
    ring=binary_dilation(lesion,iterations=8)&~binary_dilation(lesion,iterations=2); cring=binary_dilation(control,iterations=8)&~binary_dilation(control,iterations=2)
    conditions+=[("lesion_median",median_replace(image,lesion,ring)),("control_median",median_replace(image,control,cring))]
    for frac in cfg.get("lesion_fractions",[]): conditions.append((f"lesion_frac_{float(frac):.2f}",gaussian_suppress(image,nested_fraction_mask(lesion,float(frac)),float(cfg["gaussian_sigma"]))))
    rows=[]
    for name,arr in conditions:
        p=Path(out_root)/"t5"/case.dataset/case.case_id/f"{name}.jpg"; _save_image(p,to_rgb_pil(arr,intensity_mode=case.intensity_mode,size=512))
        rows.append({"item_id":f"t5:{case.dataset}:{case.case_id}:{name}","group_id":f"t5:{case.dataset}:{case.case_id}","track":"t5","dataset":case.dataset,"case_id":case.case_id,"image_path":str(p),"condition":name,"prompt":"Does this CT slice contain visible primary colorectal tumor? Answer only PRESENT or ABSENT.","choices":["PRESENT","ABSENT"],"gt":{"original_has_tumor":True,"slice_index":z}})
    return rows

def build_all_tracks(case,out_root,cfg,seed):
    out_root=Path(out_root); rng=np.random.default_rng(abs(hash((seed,case.dataset,case.case_id)))%(2**32)); rows=[]
    rows+=build_t1(case,out_root,cfg["t1"],rng); rows+=build_t2(case,out_root,cfg["t2"]); rows+=build_t3(case,out_root,cfg["t3"]); rows+=build_t4(case,out_root,cfg["t4"],rng); rows+=build_t5(case,out_root,cfg["t5"],rng)
    return rows
