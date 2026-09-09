"""Phase 1 — all-pole candidate boxes from the transformer ensemble (local H200).

For each image: SAM3 ("utility pole", multi-instance) + Grounding DINO (multi-synonym)
propose boxes; the existing human primary box is kept as a trusted anchor. Boxes are
clustered by IoU; each cluster is tagged by which engines agree:

  both  = SAM3 and DINO both fired here     (high trust)
  human = matches / is the human primary box (highest trust)
  sam3  = SAM3 only                          (ambiguous -> Phase 2 judge)
  dino  = DINO only                          (ambiguous -> Phase 2 judge)

Writes relabel/candidates_<split>.json : {stem: [{box:[x0,y0,x1,y1], tag, score}]}.

  python relabel/gen_candidates.py                 # all splits
  python relabel/gen_candidates.py --splits test   # one split
"""
from __future__ import annotations
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import pole_env  # noqa: F401  HF offline
import os, json, argparse, glob
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
from pathlib import Path
import numpy as np
from src.auto_annotate import iou_xyxy

ROOT = Path("Final_Dataset")
SAM3_THR, DINO_BOX_THR, DINO_TXT_THR, CLUSTER_IOU = 0.4, 0.30, 0.25, 0.5


def load_human(label_fp: Path, W: int, H: int):
    """Primary GT: 'cls cx cy w h' normalized -> [x0,y0,x1,y1] px."""
    out = []
    if not label_fp.exists():
        return out
    for ln in label_fp.read_text().splitlines():
        p = ln.split()
        if len(p) < 5:
            continue
        cx, cy, w, h = (float(v) for v in p[1:5])
        out.append(np.array([(cx - w / 2) * W, (cy - h / 2) * H,
                             (cx + w / 2) * W, (cy + h / 2) * H], dtype=np.float32))
    return out


def cluster(dets):
    """dets: list of (box, score, src). Greedy IoU cluster, score-sorted.
    Returns list of {box, tag, score} with tag 'both' if sam3+dino agree else src."""
    order = sorted(range(len(dets)), key=lambda i: -dets[i][1])
    used = [False] * len(dets)
    clusters = []
    for i in order:
        if used[i]:
            continue
        used[i] = True
        members = [i]
        for j in order:
            if not used[j] and iou_xyxy(dets[i][0], dets[j][0]) >= CLUSTER_IOU:
                used[j] = True
                members.append(j)
        srcs = {dets[m][2] for m in members}
        tag = "both" if {"sam3", "dino"} <= srcs else next(iter(srcs))
        clusters.append({"box": [round(float(v), 1) for v in dets[i][0].tolist()],
                         "tag": tag, "score": round(float(dets[i][1]), 3)})
    return clusters


def merge(sam3_boxes, sam3_scores, dino_boxes, dino_scores, human_boxes):
    dets = [(b, s, "sam3") for b, s in zip(sam3_boxes, sam3_scores)]
    dets += [(b, s, "dino") for b, s in zip(dino_boxes, dino_scores)]
    cands = cluster(dets)
    # Human anchors override any matching cluster; add unmatched as new 'human'.
    for hb in human_boxes:
        hit = None
        for c in cands:
            if iou_xyxy(np.array(c["box"], dtype=np.float32), hb) >= CLUSTER_IOU:
                hit = c
                break
        box = [round(float(v), 1) for v in hb.tolist()]
        if hit:
            hit["tag"], hit["box"], hit["score"] = "human", box, 1.0
        else:
            cands.append({"box": box, "tag": "human", "score": 1.0})
    return cands


def demo():
    """Self-check: cluster tagging + human override."""
    a = np.array([0, 0, 10, 100], np.float32)
    a2 = np.array([1, 1, 11, 101], np.float32)   # overlaps a (sam3+dino -> both)
    b = np.array([50, 0, 60, 100], np.float32)   # dino only
    cands = merge([a], [.9], [a2, b], [.8, .7], [])
    tags = sorted(c["tag"] for c in cands)
    assert tags == ["both", "dino"], tags
    cands = merge([a], [.9], [], [], [a2])        # human overrides the sam3 box
    assert len(cands) == 1 and cands[0]["tag"] == "human", cands
    print("demo OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=0, help="0=all (debug cap)")
    args = ap.parse_args()

    from sam3_poles import Sam3Poles
    from src.auto_annotate import OpenVocabAnnotator
    from PIL import Image
    sam3 = Sam3Poles()
    dino = OpenVocabAnnotator(use_sam=False)

    for split in args.splits:
        imgs = sorted(glob.glob(str(ROOT / split / "images" / "*.jpg")))
        if args.limit:
            imgs = imgs[:args.limit]
        out = {}
        tot = {"human": 0, "both": 0, "sam3": 0, "dino": 0}
        for n, fp in enumerate(imgs, 1):
            stem = Path(fp).stem
            im = Image.open(fp).convert("RGB")
            W, H = im.size
            sb, ss, _ = sam3.detect_segment(im, threshold=SAM3_THR)
            _, db, ds = dino.detect(im, DINO_BOX_THR, DINO_TXT_THR)
            sb = [np.array(x, np.float32) for x in sb]
            hb = load_human(ROOT / split / "labels" / f"{stem}.txt", W, H)
            cands = merge(sb, ss, db, ds, hb)
            out[stem] = cands
            for c in cands:
                tot[c["tag"]] += 1
            if n % 100 == 0:
                print(f"  [{split}] {n}/{len(imgs)}  {tot}", flush=True)
        Path("relabel").mkdir(exist_ok=True)
        json.dump(out, open(f"relabel/candidates_{split}.json", "w"))
        print(f"DONE [{split}]: {len(imgs)} imgs -> relabel/candidates_{split}.json  {tot}",
              flush=True)


if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv:
        demo()
    else:
        main()
