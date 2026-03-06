#!/usr/bin/env bash
# Run the Wikipedia GCP pipeline locally via Docker.
#
# Prerequisites:
#   1. Build the image:
#        docker build --platform linux/amd64 \
#          -f providers/gcp/Dockerfile \
#          -t wikipedia-pipeline .
#   2. A GCP service account JSON key at $SA_KEY_PATH.
#      Generate one if needed:
#        gcloud iam service-accounts keys create /path/to/key.json \
#          --iam-account wikipedia-pipeline@wikipedia-cortex.iam.gserviceaccount.com
#
# Usage:
#   SA_KEY_PATH=/path/to/key.json ./providers/gcp/ops/run-local.sh
#   REPORT_DAYS=90 SA_KEY_PATH=/path/to/key.json ./providers/gcp/ops/run-local.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"

SA_KEY_PATH="${SA_KEY_PATH:-}"
if [[ -z "$SA_KEY_PATH" ]]; then
  echo "Error: SA_KEY_PATH is not set." >&2
  echo "  export SA_KEY_PATH=/path/to/wikipedia-pipeline-key.json" >&2
  exit 1
fi

if [[ ! -f "$SA_KEY_PATH" ]]; then
  echo "Error: SA key not found at $SA_KEY_PATH" >&2
  exit 1
fi

GCS_BUCKET="${GCS_BUCKET:-wikipedia-cortex-data}"
REPORT_DAYS="${REPORT_DAYS:-30}"

docker run --rm \
  --platform linux/amd64 \
  -v "$SA_KEY_PATH":/secrets/sa-key.json:ro \
  -e GOOGLE_APPLICATION_CREDENTIALS=/secrets/sa-key.json \
  -e GCS_BUCKET="$GCS_BUCKET" \
  -e REPORT_DAYS="$REPORT_DAYS" \
  wikipedia-pipeline
