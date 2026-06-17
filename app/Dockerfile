# syntax=docker/dockerfile:1.6
# ─────────────────────────────────────────────────────────────────────────────
# NaeviScan — production Docker image for Hugging Face Spaces (Docker SDK)
#
# Design decisions:
#   • python:3.11-slim-bookworm  — Debian-based slim; ~50 MB base, glibc so
#     the pre-built PyTorch CPU wheels install without compilation.
#   • CPU-only torch/torchvision — keeps the image ~1.5 GB smaller than the
#     default CUDA build.  HF free Spaces are CPU-only anyway.
#   • BuildKit cache mount for pip — layer cache survives across rebuilds even
#     when requirements.txt changes partially.
#   • Single gunicorn worker, NO --preload — with --preload the master loads
#     the model before fork(); PyTorch's internal threading locks end up
#     half-initialized in the forked worker, deadlocking the first /predict
#     call at 0% CPU.  With --workers 1 there's only one worker anyway, so
#     loading the model post-fork costs nothing and avoids the deadlock.
#   • UID 1000 — matches the user HF Spaces runs containers as.  All writable
#     paths (HOME, pip cache, matplotlib config, torch hub) are redirected to
#     /home/user so they survive the non-root context.
#   • gunicorn --timeout 240 — TTA path runs (1 + N_TTA_AUGMENTED) forward
#     passes on CPU. With config.N_TTA_AUGMENTED=2 the typical request lands
#     in ~50–70 s, but cold model warm-up + Grad-CAM can stretch the first
#     prediction longer, so the timeout is set to 240 s as safety margin.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim-bookworm

# ── System packages ────────────────────────────────────────────────────────
# libgomp1  — OpenMP runtime used by PyTorch CPU kernels
# curl      — used only by the HEALTHCHECK, removed from final runtime layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Non-root user (UID 1000 = HF Spaces default) ─────────────────────────
RUN useradd --uid 1000 --create-home --shell /bin/bash user

# ── Writable cache directories under /home/user ───────────────────────────
# These must be set BEFORE pip/torch/matplotlib run for the first time so
# none of them attempt to write to /root or other unwritable paths.
ENV HOME=/home/user \
    PIP_CACHE_DIR=/home/user/.cache/pip \
    MPLCONFIGDIR=/home/user/.config/matplotlib \
    TORCH_HOME=/home/user/.cache/torch \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN mkdir -p \
        /home/user/.cache/pip \
        /home/user/.config/matplotlib \
        /home/user/.cache/torch && \
    chown -R user:user /home/user

# ── Python dependencies ────────────────────────────────────────────────────
# Copy only requirements first so this layer is cached until requirements.txt
# changes.  The BuildKit cache mount keeps the pip download cache across
# builds so repeated installs do not re-download wheels.
COPY requirements.txt /tmp/requirements.txt

# Install gunicorn (WSGI server; not in requirements.txt as it's ops-only).
# Install torch/torchvision from the CPU-only index to avoid the 2 GB CUDA
# wheels.  All other packages come from PyPI as normal.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip --quiet && \
    pip install gunicorn==23.0.0 --quiet && \
    pip install \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        --quiet \
        -r /tmp/requirements.txt

# ── Application code ────────────────────────────────────────────────────
# Copy application source.  .dockerignore excludes notebooks, extra model
# checkpoints, __pycache__, .DS_Store, and other dev artefacts.
WORKDIR /app
COPY --chown=user:user . /app

# ── Switch to non-root user ───────────────────────────────────────────────
USER user

# ── Expose HF Spaces required port ───────────────────────────────────────
EXPOSE 7860

# ── Runtime environment ────────────────────────────────────────────────────
# PORT is the canonical binding port.  Setting it here (default 7860) means
# gunicorn reads it from the environment in the CMD shell expansion below.
# HF Spaces always injects PORT=7860; this default is for local `docker run`.
ENV PORT=7860

# ── Health check ──────────────────────────────────────────────────────────
# Polls "/" after a 60 s start period (model cold-load on CPU).
# 3 retries × 10 s interval before marking unhealthy.
# Uses $PORT so the check stays in sync if PORT is overridden at runtime.
HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/ || exit 1

# ── Entry point ───────────────────────────────────────────────────────────
# Shell form so ${PORT} is expanded from the environment at container start.
# --workers 1   : single worker keeps peak RAM minimal on the free HF Spaces
#                 CPU tier (model weights + activations ~300 MB).
# --timeout 240 : ceiling for the cold TTA request (3 forward passes + Grad-CAM).
# NO --preload  : see header note — preloading the model in the master and
#                 then forking deadlocks PyTorch's internal locks in the
#                 worker.  Single worker => no benefit to preloading anyway.
CMD gunicorn \
    --bind "0.0.0.0:${PORT}" \
    --workers 1 \
    --timeout 240 \
    --access-logfile - \
    --error-logfile - \
    wsgi:application
