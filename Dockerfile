# ─────────────────────────────────────────────
# Stage 1: base image
# Using a slim PyTorch image with CUDA support.
# Switch to "cpu" tag if you don't have a GPU:
#   pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime  ← GPU
#   pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime  ← change to cpu build if needed
# ─────────────────────────────────────────────
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

# Keeps Python from buffering stdout (easier to read logs)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# ─────────────────────────────────────────────
# System dependencies
# ─────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# ─────────────────────────────────────────────
# Working directory inside the container
# ─────────────────────────────────────────────
WORKDIR /app

# ─────────────────────────────────────────────
# Install Python dependencies
# Copy requirements first so Docker caches this
# layer and skips re-installing on every code change
# ─────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# ─────────────────────────────────────────────
# Copy application source code
# ─────────────────────────────────────────────
COPY app/ ./app/

# ─────────────────────────────────────────────
# Pre-create directories the app expects
# (data is mounted at runtime via docker-compose)
# ─────────────────────────────────────────────
RUN mkdir -p \
        /app/app/model \
        /app/app/prediction_results/OK \
        /app/app/prediction_results/NG \
        /app/app/uploaded_images \
        /data

# ─────────────────────────────────────────────
# Expose FastAPI port
# ─────────────────────────────────────────────
EXPOSE 8000

# ─────────────────────────────────────────────
# Start the server
# ─────────────────────────────────────────────
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
