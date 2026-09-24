from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation

@dataclass
class VolumeCase:
    dataset:str; case_id:str; image:np.ndarray; label:np.ndarray; spacing_zyx:tuple[float,float,float]; split:str|None=None; intensity_mode:str="hu"
    @property
    def tumor_label(self): return 2 if self.dataset.upper()=="CARE" else 1
    @property
    def normal_label(self): return 1 if self.dataset.upper()=="CARE" else None
    def tumor_mask(self): return self.label==self.tumor_label
    def normal_mask(self): return np.zeros_like(self.label,dtype=bool) if self.normal_label is None else self.label==self.normal_label

def canonical_nifti(path):
    return nib.as_closest_canonical(nib.load(str(path)))

def load_nifti_case(image_path,mask_path,dataset,case_id,split=None):
    ii,mi=canonical_nifti(image_path),canonical_nifti(mask_path)
    image=np.asarray(ii.dataobj,dtype=np.float32); label=np.asarray(mi.dataobj)
    if image.shape!=label.shape: raise ValueError(f"shape mismatch for {case_id}: {image.shape} vs {label.shape}")
    image=np.moveaxis(image,-1,0); label=np.moveaxis(label,-1,0); z=ii.header.get_zooms()[:3]
    return VolumeCase(dataset,case_id,image,label,(float(z[2]),float(z[1]),float(z[0])),split,"hu")

def infer_case_slice(filename):
    m=re.match(r"^(?P<case>.+?)[_-](?P<slice>\d+)$",Path(filename).stem)
    return None if not m else (m.group("case"),int(m.group("slice")))

def load_care_npz_series(rows,case_id,split=None):
    rows=sorted(rows,key=lambda x:int(x["slice_index"])); images=[]; labels=[]; indices=[]
    for r in rows:
        o=np.load(r["npz_path"]); images.append(np.asarray(o["image"],dtype=np.float32)); labels.append(np.asarray(o["label"])); indices.append(int(r["slice_index"]))
    image=np.stack(images); label=np.stack(labels); mode="normalized" if image.min()>=-1e-4 and image.max()<=1.0001 else "hu"
    dz=float(rows[0].get("slice_spacing",1.0)); dy=float(rows[0].get("pixel_spacing_y",1.0)); dx=float(rows[0].get("pixel_spacing_x",1.0))
    c=VolumeCase("CARE",case_id,image,label,(dz,dy,dx),split,mode); c.slice_indices=indices; return c

def window_to_uint8(image,level=50,width=400,intensity_mode="hu"):
    a=image.astype(np.float32)
    if intensity_mode=="normalized" or (a.min()>=-1e-4 and a.max()<=1.0001): return (np.clip(a,0,1)*255).astype(np.uint8)
    lo,hi=level-width/2,level+width/2; a=np.clip(a,lo,hi); return ((a-lo)/(hi-lo)*255).astype(np.uint8)

def to_rgb_pil(slice2d,level=50,width=400,intensity_mode="hu",size=None):
    im=Image.fromarray(window_to_uint8(slice2d,level,width,intensity_mode),mode="L").convert("RGB")
    if size is not None:
        if isinstance(size,int): size=(size,size)
        im=im.resize(size,Image.Resampling.BILINEAR)
    return im

def make_montage(images,labels,rows,cols,tile_size=256,margin=8):
    canvas=Image.new("RGB",(cols*tile_size,rows*tile_size),"black"); draw=ImageDraw.Draw(canvas)
    for i,(im,lab) in enumerate(zip(images,labels)):
        r,c=divmod(i,cols); canvas.paste(im.resize((tile_size,tile_size),Image.Resampling.BILINEAR),(c*tile_size,r*tile_size))
        draw.rectangle((c*tile_size,r*tile_size,c*tile_size+28,r*tile_size+24),fill="black"); draw.text((c*tile_size+6,r*tile_size+3),lab,fill="white")
    return canvas

def bbox_from_mask(mask):
    ys,xs=np.where(mask>0); return None if len(xs)==0 else [int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]

def centroid_from_mask(mask):
    ys,xs=np.where(mask>0); return None if len(xs)==0 else [float(xs.mean()),float(ys.mean())]

def normalize_point(p,width,height):
    return [int(round(p[0]/max(width-1,1)*1000)),int(round(p[1]/max(height-1,1)*1000))]

def normalize_box(b,width,height):
    x1,y1,x2,y2=b; return [int(round(x1/width*1000)),int(round(y1/height*1000)),int(round(x2/width*1000)),int(round(y2/height*1000))]

def crop_center_physical(image,center_xy,spacing_yx,fov_mm):
    h,w=image.shape; sx=max(int(round((fov_mm/spacing_yx[1])/2)),1); sy=max(int(round((fov_mm/spacing_yx[0])/2)),1); cx,cy=center_xy
    x1,x2=int(round(cx))-sx,int(round(cx))+sx; y1,y2=int(round(cy))-sy,int(round(cy))+sy
    pl,pr=max(0,-x1),max(0,x2-w); pt,pb=max(0,-y1),max(0,y2-h); x1,x2=max(0,x1),min(w,x2); y1,y2=max(0,y1),min(h,y2)
    crop=image[y1:y2,x1:x2]
    return np.pad(crop,((pt,pb),(pl,pr)),mode="edge") if any((pl,pr,pt,pb)) else crop

def body_mask_from_hu(image):
    return binary_dilation(image>-500,iterations=2)
