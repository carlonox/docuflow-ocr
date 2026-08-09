# Deployment Guide

This guide covers all ways to deploy DocuFlow: pip install, Docker, and production patterns.

## Table of Contents

- [Option 1: pip install (simplest)](#option-1-pip-install-simplest)
- [Option 2: Docker (recommended for reproducibility)](#option-2-docker-recommended-for-reproducibility)
- [Option 3: Docker Compose (development)](#option-3-docker-compose-development)
- [Production Patterns](#production-patterns)
- [GPU Acceleration](#gpu-acceleration)
- [Ollama Integration](#ollama-integration)
- [Troubleshooting](#troubleshooting)

---

## Option 1: pip install (simplest)

### Minimal installation

```bash
pip install -r requirements.txt
```

This installs only the core framework (numpy + Pillow). No OCR providers yet.

### With providers (recommended cascade)

```bash
# Install the recommended cascade stack
pip install -r requirements.txt
pip install rapidocr-onnxruntime surya-ocr scikit-image
```

### Full installation

```bash
pip install -e ".[rapid,surya,preprocessing,vision,ollama]"
```

### Verify installation

```bash
python -c "from docuflow.ocr.cascade import CascadeOCR; print('OK')"
```

---

## Option 2: Docker (recommended for reproducibility)

Docker solves the #1 problem with OCR projects: **dependency hell**. One build, runs anywhere.

### Build the image

```bash
# Minimal (framework only)
docker build -t docuflow .

# With recommended cascade (RapidOCR + Surya)
docker build \
  --build-arg INSTALL_RAPID=true \
  --build-arg INSTALL_SURYA=true \
  -t docuflow:cascade .

# Everything (all providers + GPU)
docker build \
  --build-arg INSTALL_RAPID=true \
  --build-arg INSTALL_SURYA=true \
  --build-arg INSTALL_VISION=true \
  --build-arg INSTALL_PREPROCESS=true \
  --build-arg WITH_GPU=true \
  -t docuflow:full .
```

### Build arguments

| Argument | Default | Effect | Size impact |
|---|---|---|---|
| `INSTALL_RAPID` | `false` | RapidOCR (fast, CPU-only) | +500MB |
| `INSTALL_SURYA` | `false` | Surya OCR 2 (accurate, CPU+GPU) | +2GB |
| `INSTALL_VISION` | `false` | Google Cloud Vision SDK | +100MB |
| `INSTALL_PREPROCESS` | `false` | scikit-image (Sauvola, deskew) | +200MB |
| `INSTALL_TESSERACT` | `false` | pytesseract (legacy fallback) | +50MB |
| `INSTALL_BROWSER` | `false` | Selenium (browser automation) | +100MB |
| `WITH_GPU` | `false` | CUDA-enabled PyTorch for Surya | +1GB |

### Run the image

```bash
# Process a single document
docker run --rm \
  -v $(pwd)/documents:/input:ro \
  -v $(pwd)/output:/output \
  docuflow:cascade \
  python examples/quick_start.py /input/my_document.png

# Interactive shell for experimentation
docker run --rm -it docuflow:cascade bash

# Run with custom script
docker run --rm \
  -v $(pwd)/my_script.py:/app/script.py:ro \
  docuflow:cascade \
  python /app/script.py
```

### Persistent data

DocuFlow defines three volumes for persistent state:

| Volume | Purpose |
|---|---|
| `/data` | Learning engine state, checkpoints, feedback history |
| `/models` | Downloaded OCR model weights (Surya, RapidOCR) |
| `/output` | Processed documents, extracted fields |

Mount named volumes to preserve state across runs:

```bash
docker run --rm \
  -v docuflow_data:/data \
  -v docuflow_models:/models \
  -v docuflow_output:/output \
  -v $(pwd)/documents:/input:ro \
  docuflow:cascade \
  python examples/quick_start.py /input/my_document.png
```

---

## Option 3: Docker Compose (development)

For iterative development with hot-reload and easy environment management.

### Start the stack

```bash
# Build and start with default config (RapidOCR + Surya)
docker compose up --build

# Run a specific script
docker compose run --rm docuflow \
  python examples/cascade_with_llm.py /app/input/sample_id_card.png

# Interactive shell
docker compose run --rm docuflow bash

# View logs
docker compose logs -f docuflow
```

### Customize the stack

Edit `docker-compose.yml` to change:
- Which providers to install (build args)
- Resource limits (memory, CPU)
- Volume mounts
- Environment variables

---

## Production Patterns

### Pattern 1: Batch processing service

For processing large document batches (e.g., nightly invoice processing):

```yaml
# docker-compose.prod.yml
services:
  docuflow-worker:
    image: docuflow:cascade
    restart: always
    volumes:
      - docuflow_data:/data
      - docuflow_models:/models
      - ./inbox:/app/inbox
      - ./processed:/app/processed
    environment:
      - DOCUFLOW_MODE=batch
    command: ["python", "-m", "docuflow.batch_worker"]
```

### Pattern 2: API service

For exposing DocuFlow as an HTTP API:

```python
# api_service.py
from fastapi import FastAPI, UploadFile, File
from docuflow.ocr.cascade import CascadeOCR, CascadeConfig
from docuflow.ocr.providers.rapid_ocr import RapidOCRProvider
from docuflow.ocr.providers.surya import SuryaProvider

app = FastAPI()
cascade = CascadeOCR(
    fast_provider=RapidOCRProvider(),
    accurate_provider=SuryaProvider(),
    config=CascadeConfig(),
)

@app.post("/ocr")
async def process(file: UploadFile = File(...)):
    contents = await file.read()
    with open("/tmp/upload.png", "wb") as f:
        f.write(contents)
    result = cascade.extract("/tmp/upload.png")
    return {"fields": result.fields, "confidence": result.confidence}
```

Run with:
```bash
docker run --rm -p 8000:8000 docuflow:cascade \
  uvicorn api_service:app --host 0.0.0.0 --port 8000
```

### Pattern 3: Kubernetes deployment

For horizontal scaling:

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: docuflow
spec:
  replicas: 3
  selector:
    matchLabels:
      app: docuflow
  template:
    metadata:
      labels:
        app: docuflow
    spec:
      containers:
      - name: docuflow
        image: docuflow:cascade
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "6Gi"
            cpu: "4"
        volumeMounts:
        - name: data
          mountPath: /data
        - name: models
          mountPath: /models
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: docuflow-data
      - name: models
        persistentVolumeClaim:
          claimName: docuflow-models
```

---

## GPU Acceleration

### Docker with NVIDIA GPU

Build with GPU support:

```bash
docker build \
  --build-arg INSTALL_SURYA=true \
  --build-arg WITH_GPU=true \
  -t docuflow:gpu .
```

Run with NVIDIA runtime:

```bash
docker run --rm --gpus all \
  -v $(pwd)/documents:/input:ro \
  docuflow:gpu \
  python examples/quick_start.py /input/my_document.png
```

### Docker Compose with GPU

Uncomment the GPU section in `docker-compose.yml`:

```yaml
services:
  docuflow:
    # ... existing config ...
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

Then:
```bash
docker compose up --build
```

---

## Ollama Integration

DocuFlow can use a local Ollama instance for LLM-based correction (e.g., `glm-ocr` for vision, `qwen2.5` for text correction).

### Option A: Ollama on host (recommended)

Install Ollama on your host machine, then DocuFlow connects via `host.docker.internal`:

```bash
# Start Ollama on host
ollama serve

# Pull the models you need
ollama pull glm-ocr:latest
ollama pull qwen2.5:3b

# Run DocuFlow (it will auto-connect to host.docker.internal:11434)
docker compose up --build
```

### Option B: Ollama in container

Uncomment the `ollama` service in `docker-compose.yml`:

```yaml
services:
  ollama:
    image: ollama/ollama:latest
    container_name: docuflow-ollama
    restart: unless-stopped
    volumes:
      - ollama_data:/root/.ollama
    ports:
      - "11434:11434"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

Then:
```bash
docker compose --profile ollama up --build
```

Update `OLLAMA_HOST` in DocuFlow:
```bash
docker compose run --rm docuflow \
  -e OLLAMA_HOST=http://ollama:11434 \
  python examples/cascade_with_llm.py /input/sample.png
```

---

## Troubleshooting

### "No module named 'surya'" / "No module named 'rapidocr_onnxruntime'"

The provider wasn't included in the build. Rebuild with the right flag:

```bash
docker build --build-arg INSTALL_SURYA=true --build-arg INSTALL_RAPID=true -t docuflow .
```

### "libGL.so.1: cannot open shared object file"

Missing system library. The Dockerfile already includes it, but if you're running outside Docker:

```bash
# Ubuntu/Debian
sudo apt-get install libgl1

# macOS
brew install libgl
```

### "Ollama connection refused"

Ollama isn't running or isn't accessible from the container.

**Check Ollama is running:**
```bash
curl http://localhost:11434/api/tags
```

**Check container can reach host:**
```bash
docker exec -it docuflow curl http://host.docker.internal:11434/api/tags
```

**If using Docker Desktop on Linux**, `host.docker.internal` may not work. Use the host's LAN IP instead:

```bash
OLLAMA_HOST=http://192.168.1.100:11434 docker compose up
```

### Surya downloads models on first run (~2GB)

This is normal. Models are cached in `/models` (or `~/.cache/surya` outside Docker). Mount the volume to persist across runs:

```bash
docker run --rm -v docuflow_models:/models docuflow:cascade ...
```

### Out of memory during Surya inference

Surya wants ~4GB RAM. Increase container memory limit:

```yaml
# docker-compose.yml
services:
  docuflow:
    deploy:
      resources:
        limits:
          memory: 6G
```

Or disable Surya and use only RapidOCR:

```bash
DOCUFLOW_USE_SURYA=false docker compose up
```

### Permission denied on /data or /models

The non-root user `docuflow` needs write access. Fix by setting ownership:

```bash
docker run --rm -v docuflow_data:/data -v docuflow_models:/models \
  --user root \
  docuflow:cascade \
  chown -R docuflow:docuflow /data /models
```

---

## Image Size Reference

| Build configuration | Approximate size |
|---|---|
| Minimal (framework only) | ~350MB |
| + RapidOCR | ~850MB |
| + Surya (CPU) | ~2.8GB |
| + Surya (GPU) | ~3.8GB |
| + All providers (CPU) | ~3.2GB |
| + All providers (GPU) | ~4.2GB |

---

## Security Considerations

- **Never bake credentials into the image.** Use environment variables or secrets management (AWS Secrets Manager, HashiCorp Vault, Kubernetes Secrets).
- **Run as non-root.** The Dockerfile defaults to user `docuflow`.
- **Read-only input mounts.** Mount input directories with `:ro` to prevent accidental writes.
- **Scan the image.** Use `docker scout` or Trivy to check for vulnerabilities:
  ```bash
  docker scout cves docuflow:cascade
  ```

---

## Next Steps

- [Architecture](ARCHITECTURE.md) — understand the pipeline
- [Learning System](LEARNING_SYSTEM.md) — how DocuFlow improves over time
- [Custom Extractors](CUSTOM_EXTRACTORS.md) — write extractors for your documents
- [Security](SECURITY.md) — credential and data handling
