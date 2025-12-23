# ---- Base image ----
FROM python:3.11-slim

# Prevent Python from writing pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Workdir
WORKDIR /app

# System deps (optional but useful for some libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (better caching)
COPY requirements.txt /app/requirements.txt

# Install deps
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy project
COPY . /app

# Expose port (Render uses $PORT, local uses 8000)
EXPOSE 8000

# Start server (Render sets PORT env var; fallback to 8000)
CMD ["bash", "-lc", "uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
