from __future__ import annotations
from pathlib import Path
import torch
from PIL import Image
from transformers import pipeline
from .base import VLMAdapter

def _dtype(name):
    name=str(name).lower()
    if name in {"bf16","bfloat16"}: return torch.bfloat16
    if name in {"fp16","float16","half"}: return torch.float16
    return torch.float32

class PipelineVLM(VLMAdapter):
    def __init__(self,model_path,dtype="bfloat16",trust_remote_code=True,max_new_tokens=96):
        self.model_path=str(model_path); self.max_new_tokens=int(max_new_tokens)
        self.pipe=pipeline("image-text-to-text",model=self.model_path,torch_dtype=_dtype(dtype),device_map="auto",trust_remote_code=trust_remote_code,local_files_only=True)
    def generate(self,image_path,prompt,max_new_tokens=None):
        image=Image.open(image_path).convert("RGB")
        messages=[{"role":"user","content":[{"type":"image"},{"type":"text","text":prompt}]}]
        out=self.pipe(text=messages,images=[image],max_new_tokens=max_new_tokens or self.max_new_tokens,do_sample=False,return_full_text=False)
        item=out[0] if isinstance(out,list) else out; text=item.get("generated_text",item) if isinstance(item,dict) else item
        if isinstance(text,list):
            for msg in reversed(text):
                if isinstance(msg,dict) and msg.get("role")=="assistant":
                    content=msg.get("content","")
                    if isinstance(content,list): return "".join(x.get("text","") for x in content if isinstance(x,dict))
                    return str(content)
        return str(text).strip()
    def score_choices(self,image_path,prompt,choices):
        try:
            processor=getattr(self.pipe,"processor",None) or getattr(self.pipe,"image_processor",None); model=self.pipe.model
            if processor is None or not hasattr(processor,"apply_chat_template"): return None
            image=Image.open(image_path).convert("RGB")
            msgs=[{"role":"user","content":[{"type":"image"},{"type":"text","text":prompt}]}]
            base=processor.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True); scores={}
            base_inputs=processor(text=[base],images=[image],return_tensors="pt"); plen=int(base_inputs["input_ids"].shape[-1]); device=next(model.parameters()).device
            for candidate in choices:
                full=processor(text=[base+candidate],images=[image],return_tensors="pt")
                full={k:v.to(device) if hasattr(v,"to") else v for k,v in full.items()}; labels=full["input_ids"].clone(); labels[:,:plen]=-100
                with torch.inference_mode(): outputs=model(**full,labels=labels)
                valid=int((labels!=-100).sum().item()); scores[candidate]=float(-outputs.loss.item()*max(valid,1))
            vals=torch.tensor([scores[c] for c in choices],dtype=torch.float64); probs=torch.softmax(vals,dim=0).tolist()
            return {c:float(p) for c,p in zip(choices,probs)}
        except Exception:
            return None
