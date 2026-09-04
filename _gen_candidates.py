import os, glob, json
import pole_env  # offline HF
import numpy as np
from pathlib import Path
from PIL import Image
from sam3_poles import Sam3Poles
from refine_labels import read_labels

sam3=Sam3Poles()
splitdir="Final_Dataset/test"
files=sorted(glob.glob(f"{splitdir}/images/*.jpg"))
out={}
for i,fp in enumerate(files):
    stem=Path(fp).stem
    w,h=Image.open(fp).size
    human=read_labels(f"{splitdir}/labels/{stem}.txt",w,h)
    sb,ss,_=sam3.detect_segment(Image.open(fp).convert("RGB"),threshold=0.45)
    out[stem]={
        "w":w,"h":h,
        "human":[[round(x,1) for x in b] for b in human],
        "sam3":[[round(x,1) for x in b]+[round(float(s),3)] for b,s in zip(sb,ss)]
    }
    if (i+1)%100==0: print(f"{i+1}/{len(files)}")
json.dump(out, open("review_tool/candidates.json","w"))
print("DONE", len(out), "images -> review_tool/candidates.json")
