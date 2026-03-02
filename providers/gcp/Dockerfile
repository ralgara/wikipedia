# Dockerfile — Wikipedia pipeline (GCP Cloud Run Job)
# Build context: repo root
#   docker build -f providers/gcp/Dockerfile -t wikipedia-pipeline .

FROM python:3.12-slim

WORKDIR /app

# System deps:
#   libgomp1       — OpenMP, required by numpy/scipy parallel ops
#   fonts-dejavu-core — fonts for matplotlib chart text rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (separate layer for cache efficiency)
COPY providers/gcp/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared library and scripts
COPY shared/ shared/
COPY scripts/ scripts/

# Copy GCP app code
COPY providers/gcp/app/ providers/gcp/app/

# Force non-interactive matplotlib backend
ENV MPLBACKEND=Agg

# Cloud Run Job: run to completion and exit
ENTRYPOINT ["python", "providers/gcp/app/pipeline.py"]
