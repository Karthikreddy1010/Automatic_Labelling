"""Draw disputed-FP overlays so a human can judge: is the model's extra box a
real pole the GT missed (KEEP -> relabel) or a false detection (REJECT)?

Green = human GT box. Red = model prediction that matched NO GT (the dispute).
Yellow = model prediction that DID match a GT (for context).

Output: output/gold/disputed/<img>.jpg  (open the folder, eyeball each).
"""
from __future__ import annotations
import os, json, glob
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
from pathlib import Path
from PIL import Image, ImageDraw
from eval_detector import load_gt, match

IMAGES = "Final_Dataset/test/images"
LABELS = "Final_Dataset/test/labels"
WEIGHTS = "models/pole_detector_v2.pt"
CONF, IMGSZ = 0.25, 704
OUT = Path("output/gold/disputed"); OUT.mkdir(parents=True, exist_ok=True)

names = [n for n in Path("output/gold/v2_disputed_fp.txt").read_text().splitlines() if n]

from ultralytics import YOLO
model = YOLO(WEIGHTS)

for stem in names:
    ip = next((p for p in glob.glob(f"{IMAGES}/{stem}.*")), None)
    if not ip:
        print("missing", stem); continue
    im = Image.open(ip).convert("RGB"); W, H = im.size
    gt = load_gt(f"{LABELS}/{stem}.txt", W, H)
    r = model(ip, conf=CONF, imgsz=IMGSZ, verbose=False)[0]
    preds = []
    if r.boxes is not None:
        for b in r.boxes:
            x0, y0, x1, y1 = (float(v) for v in b.xyxy[0].tolist())
            preds.append([x0, y0, x1, y1, float(b.conf[0])])
    mg, mp = match(gt, preds, 0.5)
    d = ImageDraw.Draw(im)
    for gb in gt:
        d.rectangle(gb, outline=(0, 255, 0), width=4)          # GT green
    for i, pb in enumerate(preds):
        col = (255, 255, 0) if i in mp else (255, 0, 0)        # matched yellow / disputed red
        d.rectangle(pb[:4], outline=col, width=3)
        if i not in mp:
            d.text((pb[0] + 2, pb[1] + 2), f"{pb[4]:.2f}", fill=(255, 0, 0))
    im.save(OUT / f"{stem[:24]}.jpg")

print(f"{len(names)} overlays -> {OUT}/  (green=GT, red=disputed model box, yellow=matched)")
