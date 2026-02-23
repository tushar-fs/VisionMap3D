# Multi-stage Dockerfile for VisionMap API
#
# Stage 1 installs all dependencies (including build tools like gcc)
# Stage 2 copies only the runtime — no compilers in the final image.
# This cuts the image size roughly in half.

# --- Builder stage ---
FROM python:3.11-slim AS builder
WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install CPU-only PyTorch (saves ~2 GB vs the default CUDA build)
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Runtime stage ---
FROM python:3.11-slim AS runtime
WORKDIR /app

# Only the runtime libraries we actually need
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libjpeg62-turbo \
    libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
