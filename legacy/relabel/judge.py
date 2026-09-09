"""Phase 2 — VLM judge (Qwen2.5-VL-7B, local H200) cleans ambiguous candidates.

Trusted candidates are kept without a VLM call:
  human            -> keep (human primary box)
  both, score>=0.6 -> keep (SAM3 and DINO agree, confidently)
Everything else (sam3 / dino / low-score both) is cropped (box + 30% margin) and
shown to Qwen2.5-VL: "is this a utility pole? YES/NO". Kept iff YES.

Writes YOLO labels to Final_Dataset_allpole/<split>/labels/<stem>.txt (class 0,
normalized cx cy w h) and symlinks the images dir. Judges train+val only; the test
split is human-reviewed in Phase 4, so it is not passed here.

  python relabel/judge.py                    # train + val
  python relabel/judge.py --splits val --limit 20
"""
from __future__ import annotations
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import pole_env  # noqa: F401  HF offline
import os, json, argparse
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
from pathlib import Path
from PIL import Image

SRC = Path("Final_Dataset")
OUT = Path("Final_Dataset_allpole")
MODEL_ID = os.environ.get("JUDGE_MODEL", "Qwen/Qwen3-VL-30B-A3B-Instruct")
BOTH_KEEP = 0.6          # 'both' at/above this score skips the VLM
MARGIN = 0.30            # crop context around the box
PROMPT = ("Is this image centered on a single utility or power pole "
          "(a tall vertical pole carrying power or telecom lines)? "
          "Answer with only one word: YES or NO.")


def crop(im, box, margin=MARGIN):
    W, H = im.size
    x0, y0, x1, y1 = box
    mw, mh = (x1 - x0) * margin, (y1 - y0) * margin
    return im.crop((max(0, int(x0 - mw)), max(0, int(y0 - mh)),
                    min(W, int(x1 + mw)), min(H, int(y1 + mh))))


def to_yolo(box, W, H):
    x0, y0, x1, y1 = box
    return f"0 {((x0+x1)/2)/W:.6f} {((y0+y1)/2)/H:.6f} {(x1-x0)/W:.6f} {(y1-y0)/H:.6f}"


class QwenJudge:
    def __init__(self):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(MODEL_ID)
        kw = dict(torch_dtype="auto", device_map="cuda")
        if os.environ.get("JUDGE_4BIT") == "1":  # fit a 32B judge in a 48GB MIG slice
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        self.model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, **kw).eval()

    def is_pole(self, image) -> bool:
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": PROMPT}]}]
        text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = self.proc(text=[text], images=[image], return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**inp, max_new_tokens=4, do_sample=False)
        resp = self.proc.batch_decode(out[:, inp.input_ids.shape[1]:],
                                      skip_special_tokens=True)[0].strip().upper()
        return resp.startswith("Y")


def demo():
    """Self-check: crop clamps to image, YOLO formatting round-trips."""
    im = Image.new("RGB", (100, 200))
    c = crop(im, [40, 80, 60, 120])
    assert c.size[0] <= 100 and c.size[1] <= 200
    line = to_yolo([0, 0, 100, 200], 100, 200)
    assert line == "0 0.500000 0.500000 1.000000 1.000000", line
    print("demo OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    judge = QwenJudge()

    for split in args.splits:
        cands = json.load(open(f"relabel/candidates_{split}.json"))
        img_dir = SRC / split / "images"
        lab_dir = OUT / split / "labels"
        lab_dir.mkdir(parents=True, exist_ok=True)
        link = OUT / split / "images"
        if not link.exists():
            link.symlink_to(_os.path.relpath(img_dir, link.parent))
        stems = list(cands)[:args.limit] if args.limit else list(cands)
        st = {"human": 0, "both_auto": 0, "judged_yes": 0, "judged_no": 0}
        for n, stem in enumerate(stems, 1):
            im = Image.open(img_dir / f"{stem}.jpg").convert("RGB")
            W, H = im.size
            lines = []
            for c in cands[stem]:
                keep = False
                if c["tag"] == "human":
                    keep, st["human"] = True, st["human"] + 1
                elif c["tag"] == "both" and c["score"] >= BOTH_KEEP:
                    keep, st["both_auto"] = True, st["both_auto"] + 1
                else:
                    if judge.is_pole(crop(im, c["box"])):
                        keep, st["judged_yes"] = True, st["judged_yes"] + 1
                    else:
                        st["judged_no"] += 1
                if keep:
                    lines.append(to_yolo(c["box"], W, H))
            (lab_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            if n % 100 == 0:
                print(f"  [{split}] {n}/{len(stems)}  {st}", flush=True)
        print(f"DONE [{split}]: {len(stems)} imgs  {st}", flush=True)


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        main()
