# Dockerfile — Wikipedia pipeline (GCP Cloud Run Job)
# Build context: repo root
#   docker build --platform linux/amd64 -f providers/gcp/Dockerfile -t wikipedia-pipeline .

# amd64 is pinned in the FROM line, not left to a --platform build flag: the image is built
# on an arm64 Mac as often as on the x86 PC, and an arm64 image silently fails on the target.
FROM --platform=linux/amd64 python:3.12-alpine

WORKDIR /app

# Runtime deps:
#   libgomp    — OpenMP, required by numpy/scipy parallel ops
#   font-dejavu — fonts for matplotlib chart text rendering
RUN apk add --no-cache \
    libgomp \
    font-dejavu

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
