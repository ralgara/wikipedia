#!/usr/bin/env python3
"""GCP pipeline: download pageviews → upload JSON to GCS → generate report → upload report.

Environment variables:
    GCS_BUCKET      GCS bucket name (default: wikipedia-cortex-data)
    REPORT_DAYS     Days of history to include in report (default: 30)
    GOOGLE_APPLICATION_CREDENTIALS  Path to SA key JSON (mounted by run-local.sh)
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from google.cloud import storage

# Shared library is on PYTHONPATH=/app
sys.path.insert(0, '/app')
from shared.wikipedia.client import download_pageviews
from shared.wikipedia.storage import generate_storage_key

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

BUCKET_NAME = os.environ.get('GCS_BUCKET', 'wikipedia-cortex-data')
REPORT_DAYS = int(os.environ.get('REPORT_DAYS', '30'))
DATA_DIR = Path('/app/data')
REPORTS_DIR = Path('/app/reports')


def gcs_client():
    return storage.Client()


def blob_exists(bucket, key: str) -> bool:
    return bucket.blob(key).exists()


def upload_json(bucket, date: datetime, data: list[dict]) -> str:
    key = generate_storage_key(date)
    blob = bucket.blob(key)
    blob.upload_from_string(json.dumps(data), content_type='application/json')
    blob.make_public()
    logger.info(f"Uploaded {key} ({len(data)} articles)")
    return key


def download_recent_json(bucket, days: int) -> None:
    """Sync the most recent N days of JSON from GCS to local DATA_DIR."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.utcnow().date()
    synced = 0
    for i in range(days):
        date = datetime.combine(today - timedelta(days=i), datetime.min.time())
        key = generate_storage_key(date)
        local_path = DATA_DIR / f"pageviews_{date.strftime('%Y%m%d')}.json"
        if local_path.exists():
            continue
        blob = bucket.blob(key)
        if blob.exists():
            blob.download_to_filename(str(local_path))
            synced += 1
    logger.info(f"Synced {synced} JSON files from GCS (last {days} days)")


def upload_report(bucket, local_path: Path, gcs_path: str) -> None:
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(str(local_path), content_type='text/html')
    blob.make_public()
    logger.info(f"Uploaded report → gs://{BUCKET_NAME}/{gcs_path}")


def run_report_generator() -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    script = Path('/app/scripts/generate-report.py')
    date_str = datetime.utcnow().strftime('%Y%m%d')
    output_file = REPORTS_DIR / f"report_{date_str}.html"
    result = subprocess.run(
        [sys.executable, str(script),
         '--days', str(REPORT_DAYS),
         '--output', output_file.name],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error(f"Report generator failed:\n{result.stderr}")
        raise RuntimeError("Report generation failed")
    logger.info(result.stdout.strip())
    return output_file


def main():
    logger.info("=== Wikipedia pipeline starting ===")
    client = gcs_client()
    bucket = client.bucket(BUCKET_NAME)

    # --- Step 1: Download today's pageviews (Wikimedia lags 2-5 days) ---
    # Try today-2 first; fall back up to today-5
    uploaded_date = None
    for lag in range(2, 6):
        target = datetime.utcnow() - timedelta(days=lag)
        key = generate_storage_key(target)
        if blob_exists(bucket, key):
            logger.info(f"Already have {target.date()} in GCS, skipping download")
            uploaded_date = target
            break
        logger.info(f"Downloading pageviews for {target.date()}...")
        try:
            data = download_pageviews(target)
            upload_json(bucket, target, data)
            uploaded_date = target
            break
        except Exception as e:
            logger.warning(f"  Failed for {target.date()}: {e}")

    if not uploaded_date:
        logger.error("Could not download pageviews for any recent date")
        sys.exit(1)

    # --- Step 2: Sync recent JSON from GCS for report generation ---
    download_recent_json(bucket, REPORT_DAYS + 5)

    # --- Step 3: Generate HTML report ---
    report_path = run_report_generator()

    # --- Step 4: Upload report to GCS ---
    date_str = uploaded_date.strftime('%Y%m%d')
    upload_report(bucket, report_path, f"reports/daily/report_{date_str}.html")
    upload_report(bucket, report_path, "reports/latest.html")

    logger.info("=== Pipeline complete ===")


if __name__ == '__main__':
    main()
