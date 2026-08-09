#!/usr/bin/env bash
# setup-vm.sh — one-time project setup for the cortex runner VM.
#
# Sibling to setup.sh, which provisioned the Cloud Run path (bucket, SA, Artifact Registry,
# Cloud Scheduler). That script is left alone — it remains the record of how the bucket and
# service account were created. This one adds only what the VM path needs.
#
# Idempotent: safe to re-run.
#
# Usage:
#   ANTHROPIC_API_KEY=... ./providers/gcp/iac/setup-vm.sh
#   # or, to reuse the key already on this machine:
#   set -a && source ~/.env.wikipedia && set +a && ./providers/gcp/iac/setup-vm.sh

set -euo pipefail

PROJECT_ID="wikipedia-cortex"
REGION="us-east4"
SA="wikipedia-pipeline@${PROJECT_ID}.iam.gserviceaccount.com"
SECRET_NAME="anthropic-api-key"

echo "==> Enabling APIs (compute and secretmanager were both off)"
gcloud services enable \
  compute.googleapis.com \
  secretmanager.googleapis.com \
  --project="${PROJECT_ID}"

echo "==> Creating secret: ${SECRET_NAME}"
if gcloud secrets describe "${SECRET_NAME}" --project="${PROJECT_ID}" &>/dev/null; then
  echo "  (already exists)"
else
  gcloud secrets create "${SECRET_NAME}" \
    --replication-policy=automatic \
    --project="${PROJECT_ID}"
fi

if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "==> Adding a new secret version from \$ANTHROPIC_API_KEY"
  # printf, not echo: a trailing newline would become part of the key.
  printf '%s' "${ANTHROPIC_API_KEY}" | \
    gcloud secrets versions add "${SECRET_NAME}" --data-file=- --project="${PROJECT_ID}"
else
  echo "==> \$ANTHROPIC_API_KEY not set — skipping version upload."
  echo "    Without a version the pipeline still runs; narratives degrade to placeholders."
  echo "    Add one later with:"
  echo "      printf '%s' \"\$ANTHROPIC_API_KEY\" | gcloud secrets versions add ${SECRET_NAME} \\"
  echo "        --data-file=- --project=${PROJECT_ID}"
fi

echo "==> Granting the VM's service account what it needs"
# Read the Anthropic key at boot.
gcloud secrets add-iam-policy-binding "${SECRET_NAME}" \
  --member="serviceAccount:${SA}" \
  --role="roles/secretmanager.secretAccessor" \
  --project="${PROJECT_ID}" >/dev/null

# Pull the pipeline image from Artifact Registry.
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SA}" \
  --role="roles/artifactregistry.reader" \
  --condition=None >/dev/null

# Write logs from the VM.
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SA}" \
  --role="roles/logging.logWriter" \
  --condition=None >/dev/null

echo ""
echo "==> Project ready. Next steps:"
echo "    ./providers/gcp/iac/deploy-vm.sh    # build + push the pipeline image"
echo "    ./providers/gcp/ops/vm.sh create    # provision the Spot VM"
echo ""
echo "    The service account needs NO key file — it is attached to the VM, so Application"
echo "    Default Credentials work inside the container via the metadata server."
echo ""
echo "    To let this VM serve other cortex projects later, grant this same SA roles in those"
echo "    projects. Still keyless; no second credential to manage."
