#!/usr/bin/env bash
# deploy/hawk_start_sam3_service.sh - Start the isolated SAM3 service on HAWK
# ==============================================================================
#
# Runs services/sam3_service.py inside the SAM3 Singularity sandbox
# (Python 3.11, PyTorch 2.7.0+cu126, Transformers 5.17.0, facebook/sam3 --
# a completely separate environment from the main PoleAnnotator/Qwen one).
# Never imports SAM3 into the main application's Python process.
#
# Run this in its own terminal/systemd unit, kept running -- the main app
# talks to it over HTTP at http://127.0.0.1:8801 (see deploy/hawk.env).

set -euo pipefail

SANDBOX="/home/maska/containers/sam3-pytorch27-sandbox"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

singularity exec --nv "$SANDBOX" \
  bash -c "cd '$REPO_DIR' && python -m uvicorn services.sam3_service:app --host 0.0.0.0 --port 8801"
