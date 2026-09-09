"""Poll the Fly live-camera page for new frames, run v6 on the H200, post boxes back.

Compute-here/host-there: same pattern as eval_upload/score_and_post.py, just
looped forever instead of run-once. Single-slot server state means a slow
poll cycle here just drops frames, not queues them, so falling behind never
compounds latency.
"""
import io, json, os, sys, time, urllib.request
from PIL import Image
from ultralytics import YOLO

T = os.environ.get("REVIEW_TOKEN") or sys.exit("set REVIEW_TOKEN")
BASE = os.environ.get("REVIEW_BASE", "https://pole-review-abhi.fly.dev")
MODEL = "models/pole_detector_v6.pt"
CONF = 0.4


def get(path):
    return urllib.request.urlopen(f"{BASE}{path}?t={T}", timeout=15)


def post_json(path, obj):
    body = json.dumps(obj).encode()
    req = urllib.request.Request(f"{BASE}{path}?t={T}", data=body,
                                  headers={"Content-Type": "application/json"}, method="POST")
    urllib.request.urlopen(req, timeout=15).read()


def main():
    model = YOLO(MODEL)
    print(f"live_infer: watching {BASE} with {MODEL}", flush=True)
    last_seen = -1
    while True:
        try:
            pending = json.load(get("/api/live/pending"))["frame_id"]
            if pending is None or pending == last_seen:
                time.sleep(0.3)
                continue
            last_seen = pending
            frame = get("/api/live/frame").read()
            img = Image.open(io.BytesIO(frame)).convert("RGB")
            r = model.predict(img, imgsz=704, conf=CONF, verbose=False)[0]
            boxes = [[*map(float, b.xyxy[0]), round(float(b.conf[0]), 2)] for b in r.boxes]
            post_json("/api/live/result", {"frame_id": pending, "boxes": boxes})
            print(f"frame {pending}: {len(boxes)} poles", flush=True)
        except Exception as e:
            print("err:", e, flush=True)
            time.sleep(1)


if __name__ == "__main__":
    main()
