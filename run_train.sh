#!/usr/bin/env bash
set -euo pipefail
cd "/home/jovyan/work/Electric Utility"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TMPDIR="/home/jovyan/work/Electric Utility/.tmp"
mkdir -p "$TMPDIR"
exec python3 train_seg.py \
  --epochs 60 --batch 32 --workers 0 --imgsz 640 \
  --name pole_seg_v1 --project runs/segment/runs/pole
