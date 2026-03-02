#!/usr/bin/env bash
# deploy.sh — Build, push, and deploy the Wikipedia pipeline container.
#
# On first run: builds image, pushes to Artifact Registry, creates Cloud Run Job.
# On subsequent runs: rebuilds image, pushes, updates the job to the new image.
#
# Run from repo root:
#   ./providers/gcp/iac/deploy.sh              # full build + push + deploy
#   ./providers/gcp/iac/deploy.sh --build-only  # local build only (no push)

set -euo pipefail

PROJECT_ID="wikipedia-cortex"
REGION="us-east4"
IMAGE="us-east4-docker.pkg.dev/${PROJECT_ID}/wikipedia/pipeline"
JOB_NAME="wikipedia-pipeline"
SA="wikipedia-pipeline@${PROJECT_ID}.iam.gserviceaccount.com"

# ── Build ─────────────────────────────────────────────────────────────────────

echo "==> Building container image"
docker build \
  -f providers/gcp/Dockerfile \
  -t "${IMAGE}:latest" \
  .

if [[ "${1:-}" == "--build-only" ]]; then
  echo "  Build complete. Skipping push and deploy."
  exit 0
fi

# ── Push ──────────────────────────────────────────────────────────────────────

echo "==> Authenticating Docker with Artifact Registry"
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

echo "==> Pushing image"
docker push "${IMAGE}:latest"

# ── Deploy ────────────────────────────────────────────────────────────────────

if gcloud run jobs describe "${JOB_NAME}" \
     --region="${REGION}" \
     --project="${PROJECT_ID}" &>/dev/null; then

  echo "==> Updating Cloud Run Job: ${JOB_NAME}"
  gcloud run jobs update "${JOB_NAME}" \
    --image="${IMAGE}:latest" \
    --region="${REGION}" \
    --project="${PROJECT_ID}"

else

  echo "==> Creating Cloud Run Job: ${JOB_NAME}"
  gcloud run jobs create "${JOB_NAME}" \
    --image="${IMAGE}:latest" \
    --region="${REGION}" \
    --service-account="${SA}" \
    --memory=2Gi \
    --cpu=1 \
    --task-timeout=600 \
    --max-retries=2 \
    --set-env-vars="BUCKET_NAME=wikipedia-cortex-data,REPORT_DAYS=30,GCP_PROJECT=${PROJECT_ID}" \
    --project="${PROJECT_ID}"

fi

echo ""
echo "==> Done. To trigger a manual run:"
echo "    gcloud run jobs execute ${JOB_NAME} --region=${REGION} --wait"
