# Deploying PoleAnnotator AI

> **Read this first: the app has no authentication.** Anyone who can reach the
> port can view, edit, export and **delete** your datasets. Never publish it
> straight to the internet. Put it behind something that authenticates — a
> reverse proxy with basic auth or SSO, a VPN, or an SSH tunnel. The shipped
> `docker-compose.yml` publishes to `127.0.0.1` only for exactly this reason.

---

## 1. Which build do you want?

| | Annotation only | Full stack |
|---|---|---|
| Install | `requirements-core.txt` | `requirements.txt` |
| Size | ~330 MB deps, **626 MB image** | several GB + model weights |
| Import, draw / edit / rotate OBBs | yes | yes |
| Review, active learning, coverage, YOLO-OBB export | yes | yes |
| AI Label / Run All / Batch Engine / Refine SAM | **no** | yes |
| Needs a GPU | no | practically, yes |

The shipped `Dockerfile` builds the **annotation-only** image. The AI buttons
report that the model is not installed; they never crash the server, and
`/api/health` reports `"mode": "annotation-only"`, which is a healthy state —
do not let an orchestrator restart on it.

To build the full stack, swap `requirements-core.txt` for `requirements.txt`
in the `Dockerfile`, start from a CUDA base image, and mount your model
weights. That is a different deployment with different hardware requirements;
this guide covers the annotation image.

---

## 2. Docker Compose (recommended)

```bash
docker compose up -d          # build and start
docker compose logs -f        # watch it come up
```

Open <http://localhost:8000>.

```bash
docker compose down           # stop; datasets survive in the named volume
docker compose down -v        # stop AND DELETE all datasets
```

### Plain Docker

```bash
docker build -t poleannotator .
docker run -d --name poleannotator \
  -p 127.0.0.1:8000:8000 \
  -v poleannotator-data:/data \
  --restart unless-stopped \
  poleannotator
```

---

## 3. Without Docker

```bash
pip install -r requirements-core.txt
python run_backend.py --host 0.0.0.0 --port 8000 --no-browser
```

The launcher defaults to `127.0.0.1`, which is right for a desktop and wrong
for a server or container — nothing outside the machine can reach it. Pass
`--host 0.0.0.0` only when something in front of it handles authentication.

For a long-running service, put it under systemd (or supervisor) so it starts
on boot and restarts on failure:

```ini
# /etc/systemd/system/poleannotator.service
[Unit]
Description=PoleAnnotator AI
After=network.target

[Service]
User=poleannotator
WorkingDirectory=/opt/poleannotator
Environment=OBB_DATA_DIR=/var/lib/poleannotator/datasets
ExecStart=/opt/poleannotator/.venv/bin/python run_backend.py --host 127.0.0.1 --port 8000 --no-browser
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

---

## 4. Configuration

All optional; the defaults are what runs locally.

| Variable | Default | What it does |
|---|---|---|
| `OBB_DATA_DIR` | `data/datasets` | Where datasets, annotations, predictions, masks, history and exports live. **This is the only state worth backing up.** The image sets it to `/data/datasets`. |
| `OBB_CORS_ORIGINS` | unset | Comma-separated origin allowlist for credentialed cross-origin requests, e.g. `https://labels.example.com`. Leave unset for normal same-origin use: the app serves its own frontend, so it needs no CORS. |
| `QWEN_MODEL_PATH`, `QWEN_OLLAMA_MODEL`, `QWEN_API_KEY` | unset | Full-stack only — see the README's Qwen section. |

Command-line: `python run_backend.py --host H --port P [--no-browser] [--reload]`.

---

## 5. Health and monitoring

`GET /api/health` is cheap, touches no model, and returns:

```json
{
  "status": "ok",
  "version": "2.0.0",
  "data_dir_writable": true,
  "dataset_count": 3,
  "models_loaded": [],
  "mode": "annotation-only"
}
```

Use it as both liveness and readiness probe. Treat **only** a non-200 or a
missing response as unhealthy:

- `"models_loaded": []` with `"mode": "annotation-only"` is **normal** for this
  image. Alerting or restarting on it will restart a perfectly healthy service.
- `"data_dir_writable": false` is a real problem — the volume is missing,
  read-only, or owned by the wrong user, and imports will fail.

Kubernetes:

```yaml
livenessProbe:
  httpGet: { path: /api/health, port: 8000 }
  initialDelaySeconds: 20
  periodSeconds: 30
readinessProbe:
  httpGet: { path: /api/health, port: 8000 }
  initialDelaySeconds: 5
```

Interactive API docs are at `/docs`.

---

## 6. Reverse proxy with authentication

Nginx, terminating TLS and requiring basic auth:

```nginx
server {
    listen 443 ssl;
    server_name labels.example.com;

    ssl_certificate     /etc/letsencrypt/live/labels.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/labels.example.com/privkey.pem;

    auth_basic           "PoleAnnotator";
    auth_basic_user_file /etc/nginx/.htpasswd;

    # Image uploads are many MB per batch; the 1 MB default rejects them.
    client_max_body_size 512M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host              $host;
        proxy_set_header X-Real-IP         $remote_addr;
        proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # AI inference and batch jobs can run for minutes on CPU.
        proxy_read_timeout 600s;
        proxy_send_timeout 600s;
    }
}
```

The frontend resolves API paths relative to where its own script was loaded,
so serving the app under a sub-path (`/labels/`) works without rebuilding —
just proxy that prefix through intact.

---

## 7. Backups

Everything that matters is under `OBB_DATA_DIR`. Nothing else in the container
holds state.

```bash
# Docker named volume -> dated tarball
docker run --rm \
  -v poleannotator-data:/data:ro \
  -v "$(pwd)":/backup \
  alpine tar czf "/backup/poleannotator-$(date +%F).tar.gz" -C /data .
```

Restore by extracting back into the volume with the service stopped.

Back up before upgrading. **Deletions in the UI are immediate and permanent** —
removing an image takes its annotations, masks and history with it, "Clear all"
empties a dataset, and deleting a dataset removes everything in it. There is no
recycle bin; your backups are the undo.

---

## 8. Upgrading

```bash
git pull
docker compose build
docker compose up -d
```

The data volume is untouched by a rebuild. Dataset metadata is forward
compatible: fields added by newer versions (such as the `import_batch` tag
behind "Undo upload") are absent on older datasets and simply read as missing,
so existing datasets keep working — they just cannot undo an upload that
predates the feature.

---

## 9. Troubleshooting

| Symptom | Cause |
|---|---|
| `ModuleNotFoundError: No module named 'torch'` at startup | A dependency outside `requirements-core.txt` is being imported at module scope on the startup path. The CI workflow guards this; see `tests/test_core_install.py`. |
| Container healthy, but nothing on the published port | The app bound to `127.0.0.1` inside the container. Use `--host 0.0.0.0` — the shipped `CMD` already does. |
| Imports fail, `data_dir_writable: false` | The volume is read-only or owned by another uid. The image runs as uid **10001**; `chown -R 10001:10001` the host directory if you bind-mount one instead of using a named volume. |
| Uploads fail at the proxy with 413 | `client_max_body_size` is too small — see the Nginx block above. |
| AI buttons say "model not installed" | Expected on this image. Build the full stack (§1) to enable them. |
| Batch job shows `FAILED` | The reason is shown in the batch modal; with the annotation-only image it is the missing AI models. |
