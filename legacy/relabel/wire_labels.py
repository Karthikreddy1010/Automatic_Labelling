"""Pull traced wire polylines from the Fly tool, build a YOLO-seg dataset.

A wire is traced as a polyline (a line, zero width). YOLO-seg needs a closed
polygon, so each polyline is inflated to a ribbon of THICK pixels by offsetting
every vertex along the local perpendicular, forward then back.

THICK is the key experiment knob, not a cosmetic choice: real wires are 2-4 px
and masks that thin make IoU brutally unstable (2 px of drift ~= IoU 0). Fatter
ribbons are easier to learn and score higher, but a fat mask is a looser claim
about where the wire actually is. Report which THICK any wire mAP was measured
at -- the number is meaningless without it.

Usage:
    python3 relabel/wire_labels.py [--thick 6] [--val-frac 0.3]
"""
import argparse, json, math, os, random, shutil, sys, urllib.request
from pathlib import Path
from PIL import Image

T = os.environ.get("REVIEW_TOKEN") or sys.exit("set REVIEW_TOKEN")
BASE = os.environ.get("REVIEW_BASE", "https://pole-review-abhi.fly.dev")
SRC = Path("Dataset_new")
OUT = Path("Final_Dataset_wire")


def fetch_wires():
    """{stem_without_prefix: [polyline, ...]} for every traced image."""
    url = f"{BASE}/api/wire/traced?t={T}"
    stems = json.load(urllib.request.urlopen(url, timeout=60))
    out = {}
    for s in stems:
        if not s.startswith("wire_"):
            continue
        item = json.load(urllib.request.urlopen(
            f"{BASE}/api/wire/item?stem={s}&t={T}", timeout=60))
        out[s[len("wire_"):]] = item.get("wires", [])
    return out


def polyline_to_polygon(pts, thick):
    """Inflate a polyline into a closed ribbon polygon of width `thick`."""
    if len(pts) < 2:
        return None
    half = thick / 2.0
    normals = []
    for i, _ in enumerate(pts):
        # average the perpendiculars of the segments touching this vertex
        segs = []
        if i > 0:
            segs.append((pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]))
        if i < len(pts) - 1:
            segs.append((pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]))
        nx = ny = 0.0
        for dx, dy in segs:
            ln = math.hypot(dx, dy) or 1.0
            nx += -dy / ln
            ny += dx / ln
        ln = math.hypot(nx, ny) or 1.0
        normals.append((nx / ln, ny / ln))
    fwd = [(p[0] + n[0] * half, p[1] + n[1] * half) for p, n in zip(pts, normals)]
    bwd = [(p[0] - n[0] * half, p[1] - n[1] * half) for p, n in zip(pts, normals)]
    return fwd + bwd[::-1]


def write_split(stems, wires, split, thick):
    img_d = OUT / split / "images"
    lbl_d = OUT / split / "labels"
    img_d.mkdir(parents=True, exist_ok=True)
    lbl_d.mkdir(parents=True, exist_ok=True)
    n_poly = 0
    for stem in stems:
        src = SRC / f"{stem}.jpg"
        dst = img_d / f"{stem}.jpg"
        if not dst.exists():
            os.link(src, dst)          # hardlink, never symlink (ultralytics
                                       # resolves labels from the realpath)
        W, H = Image.open(src).size
        lines = []
        for poly in wires.get(stem, []):
            ring = polyline_to_polygon(poly, thick)
            if not ring:
                continue
            flat = []
            for x, y in ring:
                flat.append(min(max(x / W, 0.0), 1.0))
                flat.append(min(max(y / H, 0.0), 1.0))
            lines.append("0 " + " ".join(f"{v:.6f}" for v in flat))
            n_poly += 1
        (lbl_d / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    return n_poly


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thick", type=float, default=6.0, help="ribbon width in px")
    ap.add_argument("--val-frac", type=float, default=0.3)
    args = ap.parse_args()

    wires = fetch_wires()
    if not wires:
        print("no traced images yet — trace some at /wire first")
        return
    stems = sorted(wires)
    n_wires = sum(len(v) for v in wires.values())
    n_empty = sum(1 for v in wires.values() if not v)
    print(f"traced images: {len(stems)} | wires: {n_wires} "
          f"({n_wires / max(1, len(stems)):.2f}/img) | no-wire images: {n_empty}")

    if OUT.exists():
        shutil.rmtree(OUT)
    random.seed(0)
    shuffled = stems[:]
    random.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * args.val_frac))
    val, train = shuffled[:n_val], shuffled[n_val:]

    n_tr = write_split(train, wires, "train", args.thick)
    n_va = write_split(val, wires, "val", args.thick)

    root = OUT.resolve()
    (OUT / "data_local.yaml").write_text(
        f"train: {root}/train/images\n"
        f"val: {root}/val/images\n\n"
        f"nc: 1\nnames: ['wire']\n")
    for c in OUT.rglob("*.cache"):
        c.unlink()
    print(f"train {len(train)} imgs / {n_tr} polys · val {len(val)} imgs / {n_va} polys "
          f"· thick={args.thick}px")
    print(f"wrote {OUT}/data_local.yaml")


if __name__ == "__main__":
    main()
