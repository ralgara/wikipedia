#!/bin/sh
# daily-run.sh — the daily pipeline, as run inside the container on the VM.
#
# Delivered to the VM via instance metadata and bind-mounted into the container, so the
# stage order can change without rebuilding the image.
#
# Deliberately drives the five decoupled scripts rather than providers/gcp/app/pipeline.py:
# that script predates the decoupled design and still drives the old single-window
# generate-report.py. It stays in the tree, dormant, as the Cloud Run path.
#
# Every stage is idempotent, which is what makes Spot preemption safe — a half-finished
# run costs nothing to repeat:
#   - download-pageviews.py defaults to gap-fill; already-present dates are skipped
#   - upload-to-gcs.py skips data blobs already in the bucket, overwrites only reports
#
# POSIX sh, not bash: the image is python:3.12-alpine.

set -eu

cd /app

# Pre-flight: daily-run.sh and the scripts it calls arrive on the VM through TWO different
# channels — this file via instance metadata (vm.sh update), scripts/ baked into the image
# (iac/deploy-vm.sh). Push metadata without rebuilding the image and a stage silently refers
# to a script that isn't there; the reverse leaves a stage that never runs at all. That is
# exactly how the dashboard stage went missing for two days after ee26d77: the image predated
# the commit, the metadata was never pushed, and the run "succeeded" every morning.
# Fail loudly and name the fix rather than letting either half drift unnoticed.
for s in sync-from-gcs download-pageviews generate-year-report generate-all-time-report \
         analyze-longitudinal generate-dashboard upload-to-gcs; do
  if [ ! -f "scripts/$s.py" ]; then
    echo "FATAL: scripts/$s.py is missing from this image." >&2
    echo "       The image is older than daily-run.sh. Rebuild and push it first:" >&2
    echo "         ./providers/gcp/iac/deploy-vm.sh && ./providers/gcp/ops/vm.sh update" >&2
    exit 1
  fi
done

echo "==> [1/7] Seed local archive from GCS"
python scripts/sync-from-gcs.py --execute

echo "==> [2/7] Fill archive gaps"
python scripts/download-pageviews.py

echo "==> [3/7] Per-year reports"
python scripts/generate-year-report.py --all

echo "==> [4/7] All-time report + index"
python scripts/generate-all-time-report.py

echo "==> [5/7] Longitudinal analysis"
python scripts/analyze-longitudinal.py --no-cache

echo "==> [6/7] Operational dashboard"
python scripts/generate-dashboard.py --days 90

echo "==> [7/7] Publish to GCS"
python scripts/upload-to-gcs.py --all --execute

echo "==> Daily run complete"
