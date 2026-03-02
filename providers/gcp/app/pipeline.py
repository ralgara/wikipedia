#!/usr/bin/env python3
"""GCP Cloud Run Job — Wikipedia pageviews pipeline.

Steps (run once daily via Cloud Scheduler):
  1. Download yesterday's pageviews from Wikimedia API
  2. Upload JSON to GCS  (Hive-partitioned, same layout as AWS)
  3. Sync recent JSON files from GCS → local /app/data/
  4. Generate HTML report via scripts/generate-report.py
  5. Upload HTML report to GCS (public)

Environment variables:
  BUCKET_NAME   GCS bucket (default: wikipedia-cortex-data)
  REPORT_DAYS   Days of history to include in report (default: 30)
  GCP_PROJECT   GCP project ID (default: wikipedia-cortex)
"""

import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

from google.cloud import storage

# Repo root → /app in the container; needed for shared library imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from shared.wikipedia.client import download_pageviews
from shared.wikipedia.storage import generate_storage_key

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

BUCKET_NAME = os.environ.get("BUCKET_NAME", "wikipedia-cortex-data")
REPORT_DAYS = int(os.environ.get("REPORT_DAYS", "30"))
GCP_PROJECT = os.environ.get("GCP_PROJECT", "wikipedia-cortex")

# These match what generate-report.py expects (Path(__file__).parent.parent / ...)
DATA_DIR = Path("/app/data")
REPORTS_DIR = Path("/app/reports")

# ── Step 1 & 2: Download + upload JSON ───────────────────────────────────────

def download_and_store(gcs: storage.Client, date: datetime) -> str:
    """Fetch pageviews from Wikimedia API and store to GCS. Returns GCS key."""
    logger.info(f"Fetching pageviews for {date.strftime('%Y-%m-%d')}")
    data = download_pageviews(date)
    logger.info(f"  {len(data)} articles")

    key = generate_storage_key(date)
    blob = gcs.bucket(BUCKET_NAME).blob(key)
    blob.upload_from_string(
        json.dumps(data),
        content_type="application/json",
    )
    logger.info(f"  Uploaded → gs://{BUCKET_NAME}/{key}")
    return key

# ── Step 3: Sync recent JSON from GCS → local ────────────────────────────────

def sync_recent(gcs: storage.Client, days: int) -> int:
    """Download the last N days of JSON files from GCS to DATA_DIR.

    Skips dates already present locally (idempotent).
    Returns number of files downloaded.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    bucket = gcs.bucket(BUCKET_NAME)
    today = datetime.utcnow()
    downloaded = 0

    for i in range(days):
        date = today - timedelta(days=i)
        local = DATA_DIR / f"pageviews_{date.strftime('%Y%m%d')}.json"
        if local.exists():
            continue

        key = generate_storage_key(date)
        blob = bucket.blob(key)
        if blob.exists():
            blob.download_to_filename(str(local))
            downloaded += 1

    logger.info(f"Synced {downloaded} files from GCS (last {days} days)")
    return downloaded

# ── Step 4: Generate HTML report ─────────────────────────────────────────────

def generate_report() -> Path:
    """Run scripts/generate-report.py and return path to the output file."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [sys.executable, "scripts/generate-report.py", "--days", str(REPORT_DAYS)],
        cwd="/app",
        capture_output=True,
        text=True,
    )

    if result.stdout:
        logger.info(result.stdout.strip())
    if result.returncode != 0:
        logger.error(result.stderr.strip())
        raise RuntimeError(f"generate-report.py exited with code {result.returncode}")

    report = REPORTS_DIR / "latest.html"
    if not report.exists():
        raise FileNotFoundError(f"Expected report not found: {report}")

    logger.info(f"Report generated: {report} ({report.stat().st_size:,} bytes)")
    return report

# ── Step 5: Upload report to GCS ─────────────────────────────────────────────

def upload_report(gcs: storage.Client, report: Path) -> str:
    """Upload HTML report to GCS, make public, return public URL."""
    key = f"reports/{report.name}"
    blob = gcs.bucket(BUCKET_NAME).blob(key)
    blob.upload_from_filename(str(report), content_type="text/html")
    blob.make_public()
    url = blob.public_url
    logger.info(f"Report live: {url}")
    return url

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    gcs = storage.Client(project=GCP_PROJECT)
    yesterday = datetime.utcnow() - timedelta(days=1)

    logger.info("==> [1/5] Download pageviews")
    download_and_store(gcs, yesterday)

    logger.info(f"==> [2/5] Sync last {REPORT_DAYS} days from GCS")
    sync_recent(gcs, REPORT_DAYS)

    logger.info("==> [3/5] Generate HTML report")
    report = generate_report()

    logger.info("==> [4/5] Upload report to GCS")
    url = upload_report(gcs, report)

    logger.info(f"==> Done — {url}")


if __name__ == "__main__":
    main()
