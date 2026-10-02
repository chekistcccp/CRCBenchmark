#!/usr/bin/env python3
"""Create a blinded, self-contained browser form plus a private scoring key."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from crcbenchmark.inference import manifest_fingerprint
from crcbenchmark.io import read_jsonl, write_json, write_jsonl
from crcbenchmark.publication import relocate_items, select_human_items


FORM = r'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>CT benchmark 独立盲评</title>
<style>body{font:17px system-ui;max-width:1150px;margin:25px auto;padding:15px}img{max-width:100%;height:auto}textarea{width:98%;height:90px}button,input,select{font:inherit;padding:8px;margin:5px}#prompt{white-space:pre-wrap}#status{color:#465}</style>
<h1>CT benchmark 独立盲评</h1><p>请独立作答，不参考其他读者或模型输出。仅评价当前图像与提示是否足以完成任务；不要补充外部影像。此界面不展示参考答案。</p>
<label>读者编号 <input id="reviewer" placeholder="例如 reader_1"></label><p id="counter"></p><p id="prompt"></p><img id="scan" alt="待评价 CT 图像">
<p>T2 可单击图像获取归一化坐标；手动填写 point 与 box 的 JSON。框坐标也使用 0–1000。其他任务按提示输入 JSON 或 A/B。</p><p id="coordinate"></p>
<textarea id="answer" aria-label="回答" placeholder="按英文提示要求作答"></textarea><br>
<label>当前图像和提示足以完成任务？ <select id="valid"><option value="">请选择</option><option value="yes">是</option><option value="no">否</option><option value="uncertain">不确定</option></select></label>
<input id="note" style="width:70%" placeholder="问题说明：例如病灶不可辨、裁剪不完整、题目歧义"><br>
<button id="previous">上一题</button><button id="next">下一题</button><button id="download">导出完整 JSONL</button><p id="status"></p>
<script>const items=__ITEMS__, fingerprint=__FINGERPRINT__; let index=0;
const fields=['answer','valid','note'], $=id=>document.getElementById(id);
const key='crc-review:'+fingerprint; let answers=JSON.parse(localStorage.getItem(key)||'{}');
$('reviewer').value=localStorage.getItem(key+':reader')||'';
function save(){const item=items[index];answers[item.review_id]={review_id:item.review_id,packet_sha256:fingerprint,reviewer_id:$('reviewer').value.trim(),raw_response:$('answer').value,task_valid:$('valid').value,reason:$('note').value};localStorage.setItem(key,JSON.stringify(answers));localStorage.setItem(key+':reader',$('reviewer').value);}
function show(){const item=items[index],a=answers[item.review_id]||{};$('counter').textContent=(index+1)+' / '+items.length+' — '+item.review_id;$('prompt').textContent=item.prompt;$('scan').src=item.image;fields.forEach(f=>$(f).value=a[{answer:'raw_response',valid:'task_valid',note:'reason'}[f]]||'');$('coordinate').textContent='';}
fields.forEach(f=>$(f).addEventListener('input',save));$('reviewer').addEventListener('input',save);
$('previous').onclick=()=>{save();index=Math.max(0,index-1);show();};$('next').onclick=()=>{save();index=Math.min(items.length-1,index+1);show();};
$('scan').onclick=event=>{const r=event.target.getBoundingClientRect();$('coordinate').textContent='点击坐标 ['+Math.round((event.clientX-r.left)/r.width*1000)+','+Math.round((event.clientY-r.top)/r.height*1000)+']';};
$('download').onclick=()=>{save();const reader=$('reviewer').value.trim();if(!reader){$('status').textContent='请填写读者编号。';return;}const result=items.map(item=>({...answers[item.review_id],reviewer_id:reader}));const missing=result.filter(a=>!a.raw_response?.trim()||!a.task_valid);if(missing.length){$('status').textContent='还有 '+missing.length+' 题未完成；请填写回答和可判读性。';return;}const link=document.createElement('a'),url=URL.createObjectURL(new Blob([result.map(a=>JSON.stringify(a)).join('\n')+'\n'],{type:'application/x-ndjson'}));link.href=url;link.download=reader.replace(/[^a-zA-Z0-9_-]/g,'_')+'.jsonl';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);$('status').textContent='已导出 '+result.length+' 条回答。';};show();</script></html>'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default="runs/protocol_v2_5")
    parser.add_argument("--artifact-root")
    parser.add_argument("--output-dir", default="runs/protocol_v2_5/supplement/human_review")
    parser.add_argument("--patients-per-group", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args()
    root, output = Path(args.run_root), Path(args.output_dir)
    manifest = []
    evaluation = []
    for ds in ("msd", "care"):
        manifest.extend(read_jsonl(root / "manifests" / f"benchmark_{ds}_dev.jsonl"))
        evaluation.extend(read_jsonl(root / "manifests" / f"benchmark_{ds}.jsonl"))
    if {(r["dataset"], r["case_id"]) for r in manifest} & {(r["dataset"], r["case_id"]) for r in evaluation}:
        raise ValueError("Development/evaluation overlap")
    selected = relocate_items(select_human_items(manifest, args.patients_per_group, args.seed), args.artifact_root)
    public, private = [], []
    for item in selected:
        review_id = hashlib.sha256(f'{args.seed}:{item["item_id"]}'.encode()).hexdigest()[:16]
        path = Path(item["image_path"])
        encoded = base64.b64encode(path.read_bytes()).decode()
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        public.append({"review_id": review_id, "prompt": item["prompt"], "image": f"data:{mime};base64,{encoded}"})
        private.append({"review_id": review_id, "item": item})
    fingerprint = manifest_fingerprint(public)
    key_path = output / "private" / "review_key.json"
    key = {"packet_sha256": fingerprint, "scope": "development_only", "seed": args.seed,
           "source_manifest_sha256": manifest_fingerprint(manifest), "items": private}
    if key_path.exists() and json.loads(key_path.read_text(encoding="utf-8")) != key:
        raise ValueError("Review packet changed; use a fresh output directory")
    write_json(key_path, key)
    reviewer_dir = output / "reviewer"
    reviewer_dir.mkdir(parents=True, exist_ok=True)
    page = FORM.replace("__ITEMS__", json.dumps(public, ensure_ascii=False).replace("<", "\\u003c"))
    page = page.replace("__FINGERPRINT__", json.dumps(fingerprint))
    (reviewer_dir / "review.html").write_text(page, encoding="utf-8")
    write_jsonl(reviewer_dir / "answer_template.jsonl", [
        {"review_id": r["review_id"], "packet_sha256": fingerprint, "reviewer_id": "",
         "raw_response": "", "task_valid": "", "reason": ""} for r in public])
    print(f"Prepared {len(public)} blinded questions -> {reviewer_dir / 'review.html'}")
    print("Give only reviewer/ to independent doctors; keep private/ for scoring. No clinical reviews have been performed by this script.")


if __name__ == "__main__":
    main()
