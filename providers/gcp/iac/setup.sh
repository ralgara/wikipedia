#!/usr/bin/env bash
# setup.sh — Provision wikipedia-cortex GCP infrastructure
#
# Idempotent: safe to re-run. Resources that already exist will be skipped
# or produce a benign error.
#
# Prerequisites:
#   gcloud auth login
#   gcloud auth application-default login
#
# Usage:
#   ./providers/gcp/iac/setup.sh

set -euo pipefail

PROJECT_ID="wikipedia-cortex"
REGION="us-east4"
BUCKET="wikipedia-cortex-data"
REGISTRY="wikipedia"
SA_NAME="wikipedia-pipeline"
SA="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
BILLING_ACCOUNT="014455-79A88E-0F4116"

echo "==> Creating project ${PROJECT_ID}"
gcloud projects create "${PROJECT_ID}" --name="Wikipedia" || echo "  (already exists)"

echo "==> Linking billing account"
gcloud billing projects link "${PROJECT_ID}" --billing-account="${BILLING_ACCOUNT}"

echo "==> Setting active project"
gcloud config set project "${PROJECT_ID}"

echo "==> Enabling APIs"
gcloud services enable \
  run.googleapis.com \
  cloudscheduler.googleapis.com \
  artifactregistry.googleapis.com \
  storage.googleapis.com \
  cloudbuild.googleapis.com \
  --project="${PROJECT_ID}"

echo "==> Creating GCS bucket: gs://${BUCKET}"
gsutil mb -p "${PROJECT_ID}" -l "${REGION}" -b on "gs://${BUCKET}" || echo "  (already exists)"

echo "==> Creating Artifact Registry repository: ${REGISTRY}"
gcloud artifacts repositories create "${REGISTRY}" \
  --repository-format=docker \
  --location="${REGION}" \
  --description="Wikipedia pipeline container images" \
  --project="${PROJECT_ID}" || echo "  (already exists)"

echo "==> Creating service account: ${SA_NAME}"
gcloud iam service-accounts create "${SA_NAME}" \
  --display-name="Wikipedia Pipeline" \
  --project="${PROJECT_ID}" || echo "  (already exists)"

echo "==> Granting bucket access to service account"
gsutil iam ch "serviceAccount:${SA}:roles/storage.objectAdmin" "gs://${BUCKET}"

echo "==> Granting Cloud Run invoker role to service account"
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SA}" \
  --role="roles/run.invoker"

# Cloud Run Job execution URL — deterministic, no need to wait for job to exist
JOB_NAME="wikipedia-pipeline"
JOB_URI="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT_ID}/jobs/${JOB_NAME}:run"

echo "==> Creating Cloud Scheduler job: wikipedia-daily"
if gcloud scheduler jobs describe wikipedia-daily \
     --location="${REGION}" \
     --project="${PROJECT_ID}" &>/dev/null; then
  echo "  (already exists)"
else
  gcloud scheduler jobs create http wikipedia-daily \
    --location="${REGION}" \
    --schedule="0 6 * * *" \
    --uri="${JOB_URI}" \
    --message-body="{}" \
    --oauth-service-account-email="${SA}" \
    --project="${PROJECT_ID}"
fi

echo ""
echo "==> Infrastructure provisioned. Next step:"
echo "    ./providers/gcp/iac/deploy.sh    # build + push + create Cloud Run Job"
