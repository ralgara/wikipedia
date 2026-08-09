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
