# PoleAnnotator AI -- annotation workspace container
# ===================================================
#
# Builds the ANNOTATION-ONLY image (requirements-core.txt): the complete
# manual labeling workflow -- import, draw/edit OBBs, review, active-learning
# tags, coverage, YOLO-OBB export, guided tour -- at ~330 MB of dependencies
# instead of the several GB the AI pipeline needs.
#
# The AI buttons (AI Label, Run All, Batch Engine, Refine SAM) report that the
# model is not installed; they never crash the server. To build an image with
# the AI pipeline, swap requirements-core.txt for requirements.txt below and
# expect a much larger image, a CUDA base and model weights to mount.
#
#   docker build -t poleannotator .
#   docker run -p 8000:8000 -v poleannotator-data:/data poleannotator
#
# Then open http://localhost:8000

FROM python:3.13-slim

# - PYTHONDONTWRITEBYTECODE: no .pyc litter in the image or the data volume
# - PYTHONUNBUFFERED: logs reach `docker logs` immediately rather than sitting
#   in a buffer, which matters when the only view of a container is its log
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OBB_DATA_DIR=/data/datasets

WORKDIR /app

# Dependencies first, as their own layer: application edits then rebuild in
# seconds instead of re-resolving and re-downloading every package.
COPY requirements-core.txt ./
RUN pip install --no-cache-dir -r requirements-core.txt

COPY . .

# Run as a non-root user, and give it the data directory. Datasets are written
# at runtime, so the volume mount point has to be owned by that user up front
# or the first import fails on a permission error.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /data/datasets \
    && chown -R appuser:appuser /data /app
USER appuser

EXPOSE 8000

# 0.0.0.0, not the launcher's 127.0.0.1 default: a container binding only to
# loopback is unreachable from the host no matter how the port is published.
# --no-browser because there is no browser in here to open.
CMD ["python", "run_backend.py", "--host", "0.0.0.0", "--port", "8000", "--no-browser"]

# Cheap, model-free probe. An annotation-only container reports
# mode=annotation-only and is healthy -- that must not trigger a restart loop.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status == 200 else 1)"
