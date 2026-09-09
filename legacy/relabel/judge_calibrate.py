"""Measure Qwen3-VL judge agreement vs human decisions on test candidates.
Tries prompt variants; prints agreement/precision/recall per variant."""
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import pole_env  # noqa
import os, json, random
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS","1")
from PIL import Image

PROMPTS = {
 "current": ("Is this image centered on a single utility or power pole "
             "(a tall vertical pole carrying power or telecom lines)? "
             "Answer with only one word: YES or NO."),
 "strict":  ("You are labeling utility poles for a detection dataset with a STRICT standard: "
             "count a pole only if it is a clearly visible, prominent utility/power/light pole "
             "in the foreground or midground. Distant background poles, partial/cut-off poles, "
             "sign posts, masts, and tree trunks do NOT count. "
             "Does this crop show a countable pole by that standard? Answer only YES or NO."),
 "prominent": ("Is the main object in this crop a prominent, clearly visible utility or power pole "
             "that a human surveyor would definitely record? If it is distant, tiny, ambiguous, "
             "or partially cut off, answer NO. Answer only YES or NO."),
}

def crop(im, box, margin=0.30):
    W,H=im.size; x0,y0,x1,y1=box
    mw,mh=(x1-x0)*margin,(y1-y0)*margin
    return im.crop((max(0,int(x0-mw)),max(0,int(y0-mh)),min(W,int(x1+mw)),min(H,int(y1+mh))))

def main():
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    MID=os.environ.get("JUDGE_MODEL","Qwen/Qwen3-VL-8B-Instruct")
    proc=AutoProcessor.from_pretrained(MID)
    model=AutoModelForImageTextToText.from_pretrained(MID, torch_dtype="auto", device_map="cuda").eval()
    def ask(image, prompt):
        msgs=[{"role":"user","content":[{"type":"image","image":image},{"type":"text","text":prompt}]}]
        text=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
        inp=proc(text=[text],images=[image],return_tensors="pt").to(model.device)
        with torch.no_grad():
            out=model.generate(**inp,max_new_tokens=4,do_sample=False)
        return proc.batch_decode(out[:,inp.input_ids.shape[1]:],skip_special_tokens=True)[0].strip().upper().startswith("Y")

    rows=json.load(open("relabel/judge_calib.json"))
    pool=[r for r in rows if r["tag"]!="human"]      # human anchors stay auto-kept
    random.seed(0); random.shuffle(pool)
    sample=pool[:500]
    print(f"sample {len(sample)} (keep rate {sum(r['human_keep'] for r in sample)/len(sample):.1%})",flush=True)
    cache={}
    for name,pr in PROMPTS.items():
        tp=fp=fn=tn=0
        for i,r in enumerate(sample):
            key=(r["stem"],tuple(r["box"]))
            if key not in cache: cache[key]=Image.open(f"Final_Dataset/test/images/{r['stem']}.jpg").convert("RGB")
            v=ask(crop(cache[key],r["box"]),pr)
            if v and r["human_keep"]: tp+=1
            elif v: fp+=1
            elif r["human_keep"]: fn+=1
            else: tn+=1
            if (i+1)%100==0: print(f"  [{name}] {i+1}/{len(sample)}",flush=True)
        acc=(tp+tn)/len(sample); P=tp/max(tp+fp,1); R=tp/max(tp+fn,1)
        print(f"{name:9s} acc={acc:.3f} P={P:.3f} R={R:.3f} judgeYES={(tp+fp)/len(sample):.1%} (human {sum(r['human_keep'] for r in sample)/len(sample):.1%})",flush=True)

main()
