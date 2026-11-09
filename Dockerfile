FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install essential system build tools and OpenCV/dlib dependencies
# libgl1 and libglib2.0-0 are required for OpenCV headless execution
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libgl1 \
    libglib2.0-0 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python package dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "setuptools<81" wheel && \
    pip install --no-cache-dir -r requirements.txt

# Install offline model weights for face_recognition
RUN pip install --no-cache-dir git+https://github.com/ageitgey/face_recognition_models || true

# Create application directories for storage, database, symlink results, and offline AI models
RUN mkdir -p /app/data/raw /app/data/results /app/data/db /app/data/models/insightface /app/templates /app/static /app/img

# Set InsightFace offline model root path
ENV INSIGHTFACE_ROOT=/app/data/models/insightface

# Copy application source code
COPY . /app

# Ensure data folders are writable
RUN chmod -R 777 /app/data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
