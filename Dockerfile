# ============================================================
# G-MASS · MediSafe-GH · Dockerfile
# Africa AI Safety Prize 2026 · Apache 2.0
#
# Supports two build targets:
#   cpu   — Kaggle, CI, lightweight inference (default)
#   gpu   — RTX / CUDA, local model inference (LLaMA, RoBERTA, AfroLM, BioMistral)
#
# Build (CPU):
#   docker build --target cpu -t medisafe-gh:cpu .
#
# Build (GPU):
#   docker build --target gpu -t medisafe-gh:gpu .
#
# Run evaluation (CPU):
#   docker run --rm \
#     -v $(pwd)/data:/app/data \
#     -v $(pwd)/configs:/app/configs \
#     -v $(pwd)/logs:/app/logs \
#     --env-file .env \
#     medisafe-gh:cpu \
#     gmass evaluate --model gpt-4o-mini --language english
#
# Run in background (nohup equivalent inside container):
#   docker run -d \
#     --name gmass-eval \
#     -v $(pwd)/data:/app/data \
#     -v $(pwd)/configs:/app/configs \
#     -v $(pwd)/logs:/app/logs \
#     --env-file .env \
#     medisafe-gh:cpu \
#     gmass evaluate --all-models
#
# Follow logs:
#   docker logs -f gmass-eval
#
# Run GPU (RTX):
#   docker run --rm --gpus all \
#     -v $(pwd)/data:/app/data \
#     -v $(pwd)/configs:/app/configs \
#     -v $(pwd)/logs:/app/logs \
#     --env-file .env \
#     medisafe-gh:gpu \
#     gmass evaluate --model llama-3.2-3b --language twi
# ============================================================

# ── Stage 0: shared base ─────────────────────────────────────────────────────
FROM python:3.11-slim AS base

# Metadata
LABEL org.opencontainers.image.title="MediSafe-GH G-MASS"
LABEL org.opencontainers.image.description="Ghana Medical AI Safety Screen — cross-lingual safety evaluation for medical AI"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.source="https://github.com/Armstrong66/medisafe-gh"

# System dependencies
# - curl: healthcheck and fasttext model download
# - git: HuggingFace datasets with git-lfs blobs
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security — never run ML workloads as root
RUN useradd --create-home --shell /bin/bash gmass
WORKDIR /app

# ── Stage 1: dependency installation ─────────────────────────────────────────
FROM base AS deps

# Copy only dependency files first (Docker layer cache: deps rebuild only when
# pyproject.toml or requirements.txt change, not on every code change)
COPY pyproject.toml ./
COPY requirements.txt ./

# Install Python dependencies
# --no-cache-dir keeps image size down
# fasttext installed separately (requires C build tools briefly)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir fasttext-wheel

# Download fasttext language identification model (176 languages, ~1MB)
# Stored inside the image so containers work fully offline after build
RUN mkdir -p /app/models/fasttext && \
    curl -sL https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin \
    -o /app/models/fasttext/lid.176.bin

# ── Stage 2a: CPU target (default) ───────────────────────────────────────────
FROM deps AS cpu

# Install CPU-only PyTorch (much smaller than CUDA build: ~250MB vs ~2GB)
RUN pip install --no-cache-dir \
    torch==2.3.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Copy package source
COPY medisafe_gh/ ./medisafe_gh/
COPY configs/     ./configs/
COPY scripts/     ./scripts/
COPY README.md    ./

# Install the package in editable mode so `gmass` CLI is available
RUN pip install --no-cache-dir -e . && \
    chown -R gmass:gmass /app

# Create mount-point directories (populated at runtime via -v flags)
RUN mkdir -p /app/data/probes \
             /app/data/eval_outputs/raw \
             /app/data/eval_outputs/scored \
             /app/data/eval_outputs/combined \
             /app/data/simulation \
             /app/logs && \
    chown -R gmass:gmass /app/data /app/logs

USER gmass

# Environment defaults (override at runtime via --env-file .env)
ENV GMASS_LOG_LEVEL=INFO
ENV FASTTEXT_LID_PATH=/app/models/fasttext/lid.176.bin
# API keys must be injected at runtime — never baked into the image
# OPENAI_API_KEY, GOOGLE_API_KEY, KHAYA_API_KEY, HF_TOKEN

# Healthcheck: verify the gmass CLI is available and config loads correctly
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD gmass --version || exit 1

ENTRYPOINT ["gmass"]
CMD ["--help"]

# ── Stage 2b: GPU target (RTX / CUDA) ────────────────────────────────────────
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS gpu

# Re-install Python in CUDA base image (it uses Ubuntu, not python:slim)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    libsndfile1 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* && \
    ln -sf python3.11 /usr/bin/python3 && \
    ln -sf python3 /usr/bin/python

RUN useradd --create-home --shell /bin/bash gmass
WORKDIR /app

# Copy pre-built deps from the deps stage
COPY --from=deps /usr/local/lib/python3.11/site-packages \
                  /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin
COPY --from=deps /app/models    /app/models

# Install CUDA-enabled PyTorch on top of CPU deps
RUN pip install --no-cache-dir \
    torch==2.3.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Copy package source
COPY medisafe_gh/ ./medisafe_gh/
COPY configs/     ./configs/
COPY scripts/     ./scripts/
COPY requirements.txt pyproject.toml README.md ./

RUN pip install --no-cache-dir -e . && \
    chown -R gmass:gmass /app

RUN mkdir -p /app/data/probes \
             /app/data/eval_outputs/raw \
             /app/data/eval_outputs/scored \
             /app/data/eval_outputs/combined \
             /app/data/simulation \
             /app/logs && \
    chown -R gmass:gmass /app/data /app/logs

USER gmass

ENV GMASS_LOG_LEVEL=INFO
ENV FASTTEXT_LID_PATH=/app/models/fasttext/lid.176.bin

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD gmass --version || exit 1

ENTRYPOINT ["gmass"]
CMD ["--help"]