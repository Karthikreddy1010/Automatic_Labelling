#!/usr/bin/env bash
set -euo pipefail
cd "/home/jovyan/work/Electric Utility"
export HF_HUB_DISABLE_PROGRESS_BARS=1 TMPDIR="/home/jovyan/work/Electric Utility/.tmp"
mkdir -p "$TMPDIR"
exec python3 train_detector.py --model yolo26m.pt --epochs 2 --batch 32 --name pole_det_smoke
