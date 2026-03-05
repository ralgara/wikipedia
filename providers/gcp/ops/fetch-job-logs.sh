#!/usr/bin/env bash
# fetch-job-logs.sh — Fetch Cloud Logging output for a Cloud Run Job execution.
#
# Usage:
#   ./providers/gcp/ops/fetch-job-logs.sh [EXECUTION_NAME]
#   ./providers/gcp/ops/fetch-job-logs.sh [EXECUTION_NAME] [--project P] [--region R] [--job J]
#
# Defaults (this project):
#   --project  wikipedia-cortex
#   --region   us-east4
#   --job      wikipedia-pipeline
#
# Examples:
#   ./providers/gcp/ops/fetch-job-logs.sh                              # latest execution, defaults
#   ./providers/gcp/ops/fetch-job-logs.sh wikipedia-pipeline-kcs6t     # specific execution
#   ./providers/gcp/ops/fetch-job-logs.sh --project my-project --job my-job  # different project

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────

PROJECT_ID="wikipedia-cortex"
REGION="us-east4"
JOB_NAME="wikipedia-pipeline"
EXECUTION=""

# ── Argument parsing ──────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
  case "${1}" in
    --project) PROJECT_ID="${2}"; shift 2 ;;
    --region)  REGION="${2}";     shift 2 ;;
    --job)     JOB_NAME="${2}";   shift 2 ;;
    --*)       echo "Unknown option: ${1}" >&2; exit 1 ;;
    *)         EXECUTION="${1}";  shift ;;
  esac
done

# ── Resolve execution name ────────────────────────────────────────────────────

if [[ -z "${EXECUTION}" ]]; then
  echo "==> Fetching most recent execution for job: ${JOB_NAME} (${PROJECT_ID})"
  EXECUTION=$(gcloud run jobs executions list \
    --job="${JOB_NAME}" \
    --region="${REGION}" \
    --project="${PROJECT_ID}" \
    --limit=1 \
    --format="value(name)" \
    | xargs basename)
  echo "    Execution: ${EXECUTION}"
fi

# ── Fetch logs ────────────────────────────────────────────────────────────────

echo ""
echo "==> Logs for execution: ${EXECUTION}"
echo "──────────────────────────────────────────────────────────────────────────"

gcloud logging read \
  "resource.type=\"cloud_run_job\" \
   AND resource.labels.job_name=\"${JOB_NAME}\" \
   AND labels.\"run.googleapis.com/execution_name\"=\"${EXECUTION}\"" \
  --project="${PROJECT_ID}" \
  --limit=200 \
  --order=asc \
  --format="value(timestamp,textPayload,jsonPayload.message)" \
  2>/dev/null | grep -v '^[[:space:]]*$' || true

echo "──────────────────────────────────────────────────────────────────────────"
echo ""
echo "==> Console URL:"
echo "    https://console.cloud.google.com/logs/viewer?project=${PROJECT_ID}&advancedFilter=resource.type%3D%22cloud_run_job%22%0Aresource.labels.job_name%3D%22${JOB_NAME}%22%0Alabels.%22run.googleapis.com%2Fexecution_name%22%3D%22${EXECUTION}%22"
