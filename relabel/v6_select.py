"""v6 active learning: run v5 over unused Dataset/ images, pick review batch."""
import json, os, re
from ultralytics import YOLO

used = set()
for f, pre in [("relabel/old_dataset_reviews.json", None),
               ("relabel/spotcheck_reviews.json", None)]:
    for k in json.load(open(f)):
        used.add(re.sub(r"^(eval_|check_|audit_|old_)", "", k))
pool = sorted(s[:-4] for s in os.listdir("Dataset") if s.endswith(".jpg") and s[:-4] not in used)
print("pool:", len(pool), "| used:", len(used), flush=True)

m = YOLO("models/pole_detector_v5.pt")
out = {}
B = 64
for i in range(0, len(pool), B):
    batch = [f"Dataset/{s}.jpg" for s in pool[i:i+B]]
    for s, r in zip(pool[i:i+B], m.predict(batch, imgsz=704, conf=0.15, verbose=False)):
        out[s] = [[*map(float, b.xyxy[0]), float(b.conf[0])] for b in r.boxes]
    if i % 640 == 0:
        print(f"{i}/{len(pool)}", flush=True)
json.dump(out, open("relabel/v6_preds.json", "w"))

# selection: uncertain (any box conf 0.15-0.55) or crowded (3+ boxes) or empty
sel = []
for s, boxes in out.items():
    confs = [b[4] for b in boxes]
    unc = sum(1 for c in confs if c < 0.55)
    score = unc * 2 + (len(boxes) >= 3) * 1 + (len(boxes) == 0) * 1
    if score > 0:
        sel.append((score, s))
sel.sort(reverse=True)
picked = [s for _, s in sel[:400]]
open("relabel/v6_batch.txt", "w").write("\n".join(picked) + "\n")
n_unc = sum(1 for sc, _ in sel if sc >= 2)
print(f"DONE v6 select: candidates {len(sel)}, picked {len(picked)} (uncertain-heavy {n_unc})", flush=True)
