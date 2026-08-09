#!/usr/bin/env bash
# deploy-vm.sh — build and push the pipeline image for the VM path.
#
# Sibling to deploy.sh, which does the same build then creates/updates a Cloud Run Job.
# The VM pulls this image directly, so there is no job to update — build and push is all.
#
# Run from repo root:
#   ./providers/gcp/iac/deploy-vm.sh
#   ./providers/gcp/iac/deploy-vm.sh --build-only

set -euo pipefail

PROJECT_ID="wikipedia-cortex"
REGION="us-east4"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/wikipedia/pipeline"

echo "==> Building container image"
# --platform is also pinned in the Dockerfile's FROM line; kept here so the intent is visible
# at the call site too. arm64 images built on the Mac fail silently on the VM.
docker build \
  --platform linux/amd64 \
  -f providers/gcp/Dockerfile \
  -t "${IMAGE}:latest" \
  .

if [[ "${1:-}" == "--build-only" ]]; then
  echo "  Build complete. Skipping push."
  exit 0
fi

echo "==> Authenticating Docker with Artifact Registry"
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

echo "==> Pushing image"
docker push "${IMAGE}:latest"

echo ""
echo "==> Pushed ${IMAGE}:latest"
echo "    Roll it out to the VM with:"
echo "      ./providers/gcp/ops/vm.sh run-now     # pulls latest, then runs"
