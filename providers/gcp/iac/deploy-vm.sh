#!/usr/bin/env bash
# deploy-vm.sh — build and push the pipeline image for the VM path.
#
# Sibling to deploy.sh, which does the same build then creates/updates a Cloud Run Job.
# The VM pulls this image directly, so there is no job to update — build and push is all.
#
# Run from repo root:
#   ./providers/gcp/iac/deploy-vm.sh
#   ./providers/gcp/iac/deploy-vm.sh --build-only
#   ./providers/gcp/iac/deploy-vm.sh --impersonate   # push from warp-host
#
# ON AUTH. The default path assumes the caller's own credentials can push — true on a
# workstation where you are logged in as yourself, false on warp-host, whose attached SA
# holds artifactregistry.reader and nothing more. That is deliberate: the systemd unit runs
# the pulled image as root, so an identity that can both run the pipeline and republish its
# image could rewrite what root executes tomorrow.
#
# --impersonate keeps that separation while still allowing a push from the host: it mints a
# short-lived token for the builder SA (which holds writer and which nothing runs as) and
# logs Docker in with that token instead of the caller's own identity. The unattended
# container never takes this path, so it still cannot publish.
#
# Run providers/gcp/ops/grant-builder-access.sh once before using --impersonate.

set -euo pipefail

PROJECT_ID="wikipedia-cortex"
REGION="us-east4"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/wikipedia/pipeline"
BUILDER_SA="${WIKI_BUILDER_SA:-wikipedia-builder@${PROJECT_ID}.iam.gserviceaccount.com}"

BUILD_ONLY=0
IMPERSONATE=0
for arg in "$@"; do
  case "$arg" in
    --build-only)  BUILD_ONLY=1 ;;
    --impersonate) IMPERSONATE=1 ;;
    -h|--help)     sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "Unknown flag: $arg" >&2; exit 1 ;;
  esac
done

echo "==> Building container image"
# --platform is also pinned in the Dockerfile's FROM line; kept here so the intent is visible
# at the call site too. arm64 images built on the Mac fail silently on the VM.
docker build \
  --platform linux/amd64 \
  -f providers/gcp/Dockerfile \
  -t "${IMAGE}:latest" \
  .

if [[ $BUILD_ONLY -eq 1 ]]; then
  echo "  Build complete. Skipping push."
  exit 0
fi

if [[ $IMPERSONATE -eq 1 ]]; then
  echo "==> Authenticating Docker as ${BUILDER_SA} (impersonated)"
  # The token is short-lived and never written to disk beyond Docker's own config; it is
  # piped straight in rather than passed as an argument so it stays out of the process list.
  if ! gcloud auth print-access-token --impersonate-service-account="$BUILDER_SA" 2>/dev/null \
       | docker login -u oauth2accesstoken --password-stdin "https://${REGION}-docker.pkg.dev"; then
    echo "  FAILED to impersonate ${BUILDER_SA}." >&2
    echo "  Run providers/gcp/ops/grant-builder-access.sh first (from a machine with IAM" >&2
    echo "  admin on ${PROJECT_ID}), then retry. Fresh bindings can take a few minutes to" >&2
    echo "  take effect — wait and retry before re-granting." >&2
    exit 1
  fi
else
  echo "==> Authenticating Docker with Artifact Registry"
  gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet
fi

echo "==> Pushing image"
docker push "${IMAGE}:latest"

echo ""
echo "==> Pushed ${IMAGE}:latest"
echo "    Roll it out to the VM with:"
echo "      ./providers/gcp/ops/vm.sh run-now     # pulls latest, then runs"
