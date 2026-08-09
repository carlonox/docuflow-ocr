# syntax=docker/dockerfile:1.7
# ============================================================================
# DocuFlow — Multi-stage Docker build
# ============================================================================
#
# Build arguments control which OCR providers to include:
#   --build-arg INSTALL_SURYA=true      (adds Surya OCR 2, ~2GB, CPU+GPU)
#   --build-arg INSTALL_RAPID=true      (adds RapidOCR ONNX, ~500MB, CPU only)
#   --build-arg INSTALL_TESSERACT=true  (adds Tesseract, requires system binary)
#   --build-arg INSTALL_VISION=true     (adds Google Cloud Vision SDK)
#   --build-arg INSTALL_PREPROCESS=true (adds scikit-image for advanced preprocessing)
#   --build-arg INSTALL_BROWSER=true    (adds Selenium for browser automation)
#   --build-arg WITH_GPU=true           (enables CUDA support for Surya)
#
# Examples:
#   # Minimal (framework only, no OCR providers)
#   docker build -t docuflow .
#
#   # With RapidOCR + Surya (the recommended cascade)
#   docker build --build-arg INSTALL_RAPID=true --build-arg INSTALL_SURYA=true -t docuflow .
#
#   # Everything, GPU-enabled
#   docker build \
#     --build-arg INSTALL_RAPID=true \
#     --build-arg INSTALL_SURYA=true \
#     --build-arg INSTALL_VISION=true \
#     --build-arg INSTALL_PREPROCESS=true \
#     --build-arg WITH_GPU=true \
#     -t docuflow:gpu .
#
# ============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Builder (install dependencies, compile wheels)
# ---------------------------------------------------------------------------
ARG PYTHON_VERSION=3.11
ARG WITH_GPU=false

FROM python:${PYTHON_VERSION}-slim AS builder

ARG INSTALL_SURYA=false
ARG INSTALL_RAPID=false
ARG INSTALL_TESSERACT=false
ARG INSTALL_VISION=false
ARG INSTALL_PREPROCESS=false
ARG INSTALL_BROWSER=false
ARG WITH_GPU=false

# Build dependencies for C extensions (numpy, scikit-image, onnxruntime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy only dependency files first (Docker cache optimization)
COPY pyproject.toml requirements.txt ./

# Install core dependencies
RUN pip install --no-cache-dir --upgrade pip wheel setuptools \
 && pip install --no-cache-dir -r requirements.txt

# Install optional providers based on build args
RUN if [ "$INSTALL_RAPID" = "true" ]; then \
        echo "Installing RapidOCR (ONNX)..." \
        && pip install --no-cache-dir "rapidocr-onnxruntime>=1.3"; \
    fi

RUN if [ "$INSTALL_SURYA" = "true" ]; then \
        echo "Installing Surya OCR 2..." \
        && if [ "$WITH_GPU" = "true" ]; then \
            pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 torch torchvision; \
        else \
            pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision; \
        fi \
        && pip install --no-cache-dir "surya-ocr>=0.5"; \
    fi

RUN if [ "$INSTALL_VISION" = "true" ]; then \
        echo "Installing Google Cloud Vision..." \
        && pip install --no-cache-dir "google-cloud-vision>=3.7" "gspread>=5.10" "google-auth>=2.23"; \
    fi

RUN if [ "$INSTALL_PREPROCESS" = "true" ]; then \
        echo "Installing scikit-image..." \
        && pip install --no-cache-dir "scikit-image>=0.21"; \
    fi

RUN if [ "$INSTALL_TESSERACT" = "true" ]; then \
        echo "Installing pytesseract..." \
        && pip install --no-cache-dir "pytesseract>=0.3.10"; \
    fi

RUN if [ "$INSTALL_BROWSER" = "true" ]; then \
        echo "Installing Selenium..." \
        && pip install --no-cache-dir "selenium>=4.15"; \
    fi

# Install the package itself
COPY . .
RUN pip install --no-cache-dir -e .

# ---------------------------------------------------------------------------
# Stage 2: Runtime (minimal image)
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

ARG WITH_GPU=false

LABEL org.opencontainers.image.title="DocuFlow" \
      org.opencontainers.image.description="Self-learning document processing framework with cascade OCR" \
      org.opencontainers.image.source="https://github.com/carlonox/docuflow-ocr" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.authors="Carlos Javier Cuervo Baracaldo"

# Runtime system dependencies
# - libgl1: OpenCV/RapidOCR needs it
# - libglib2.0-0: ONNX Runtime needs it
# - tesseract-ocr: only if INSTALL_TESSERACT was true (handled by builder)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r docuflow && useradd -r -g docuflow -d /app -s /sbin/nologin docuflow

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --from=builder /build/src ./src
COPY --from=builder /build/examples ./examples
COPY --from=builder /build/scripts ./scripts
COPY --from=builder /build/tests ./tests
COPY --from=builder /build/docs ./docs
COPY --from=builder /build/pyproject.toml ./pyproject.toml

# Create data directory for persistent storage
RUN mkdir -p /data /models /output \
 && chown -R docuflow:docuflow /app /data /models /output

# Default environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    DOCUFLOW_DATA_DIR=/data \
    DOCUFLOW_MODELS_DIR=/models \
    OLLAMA_HOST=http://host.docker.internal:11434 \
    # Providers are lazy-loaded — set to false to disable at runtime
    DOCUFLOW_USE_SURYA=true \
    DOCUFLOW_USE_RAPID=true \
    DOCUFLOW_USE_TESSERACT=false \
    DOCUFLOW_USE_VISION=false \
    DOCUFLOW_USE_OLLAMA=true

# Health check — verify Python environment is functional
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import docuflow; print('docuflow OK')" || exit 1

USER docuflow

# Volumes for persistent data
VOLUME ["/data", "/models", "/output"]

# Default command: run quick_start on a synthetic sample document
CMD ["python", "examples/quick_start.py", "tests/fixtures/sample_id_card_1.png"]
