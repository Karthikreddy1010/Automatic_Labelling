"""Context judge: draw the candidate box on the FULL image, ask if a surveyor would count it."""
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import pole_env  # noqa
import os, json, random
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS","1")
from PIL import Image, ImageDraw

PROMPT=("The red box in this street photo marks one candidate object. "
        "Would a utility surveyor count the object in the red box as a distinct, clearly "
        "identifiable utility/power/light pole worth recording? Distant barely-visible poles, "
        "sign posts, tree trunks, and duplicate marks of an already obvious pole do not count. "
        "Answer only YES or NO.")

def main():
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    MID=os.environ.get("JUDGE_MODEL","Qwen/Qwen3-VL-8B-Instruct")
    proc=AutoProcessor.from_pretrained(MID)
    model=AutoModelForImageTextToText.from_pretrained(MID, torch_dtype="auto", device_map="cuda").eval()
    def ask(image):
        msgs=[{"role":"user","content":[{"type":"image","image":image},{"type":"text","text":PROMPT}]}]
        text=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
        inp=proc(text=[text],images=[image],return_tensors="pt").to(model.device)
        with torch.no_grad():
            out=model.generate(**inp,max_new_tokens=4,do_sample=False)
        return proc.batch_decode(out[:,inp.input_ids.shape[1]:],skip_special_tokens=True)[0].strip().upper().startswith("Y")

    rows=json.load(open("relabel/judge_calib.json"))
    pool=[r for r in rows if r["tag"]!="human"]
    random.seed(0); random.shuffle(pool)
    sample=pool[:500]
    tp=fp=fn=tn=0
    for i,r in enumerate(sample):
        im=Image.open(f"Final_Dataset/test/images/{r['stem']}.jpg").convert("RGB")
        d=ImageDraw.Draw(im); d.rectangle(r["box"],outline=(255,0,0),width=4)
        v=ask(im)
        if v and r["human_keep"]: tp+=1
        elif v: fp+=1
        elif r["human_keep"]: fn+=1
        else: tn+=1
        if (i+1)%100==0: print(f"  {i+1}/{len(sample)} interim acc={(tp+tn)/(i+1):.3f}",flush=True)
    n=len(sample); acc=(tp+tn)/n; P=tp/max(tp+fp,1); R=tp/max(tp+fn,1)
    F=2*P*R/max(P+R,1e-9)
    print(f"context-judge acc={acc:.3f} P={P:.3f} R={R:.3f} F1={F:.3f} judgeYES={(tp+fp)/n:.1%} (human 19.8%)",flush=True)
main()
