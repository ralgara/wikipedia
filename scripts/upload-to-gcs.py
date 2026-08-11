#!/usr/bin/env python3
"""Sync local data/ and reports/ to GCS.

This is the decoupled publish step of the local pipeline: download → generate → upload,
three independent invocations. Nothing here generates anything; it only mirrors what is
already on disk.

    uv run python scripts/upload-to-gcs.py                   # dry run (the default)
    uv run python scripts/upload-to-gcs.py --reports         # publish HTML reports
    uv run python scripts/upload-to-gcs.py --data            # publish pageviews JSON
    uv run python scripts/upload-to-gcs.py --all --execute   # both, for real

Nothing uploads unless --execute is passed.

Data blobs are immutable: a pageviews file for a given date is written once and skipped
forever after. Reports are always overwritten — they are regenerated from the full archive
each run, so the newest copy is always the correct one.

Relationship to backfill-gcs.py: that script shells out to `gsutil -m cp` and is the right
tool for a first bulk upload of the whole archive (thousands of files, parallel, no Python
deps). This script uses the storage SDK and is the right tool for incremental daily syncs
and for reports. Both write the same Hive-partitioned layout and are safe to interleave.

Environment:
    GCS_BUCKET                      target bucket (default: wikipedia-cortex-data)
    GOOGLE_APPLICATION_CREDENTIALS  path to a service-account JSON key; if unset, falls
                                    back to Application Default Credentials
"""

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from google.cloud import storage

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from shared.wikipedia.storage import generate_storage_key

DATA_DIR = ROOT / 'data'
REPORTS_DIR = ROOT / 'reports'

DEFAULT_BUCKET = 'wikipedia-cortex-data'
DATA_PREFIX = 'wikipedia/pageviews'
REPORTS_PREFIX = 'reports'

# Derived artifacts that are expensive to rebuild and NOT reproducible. narratives_cache.json is
# 242 LLM-generated spike narratives; regenerating it costs 242 API calls and — because the model
# is non-deterministic — produces different prose for the same events, silently rewriting the
# narrative text in all 12 year reports. Before 2026-08-11 this file lived only on whichever host
# happened to run the pipeline, so it diverged unnoticed between the Mac and cortex-runner, and
# again when the job moved to warp-host. The bucket is the durable copy of the archive; it has to
# be the durable copy of this too.
DERIVED_PREFIX = 'wikipedia/derived'
DERIVED_FILES = ['narratives_cache.json']

PAGEVIEWS_RE = re.compile(r'pageviews_(\d{8})\.json$')

# The bucket grants allUsers:objectViewer at the IAM level with uniform bucket-level access,
# so per-object ACLs are disabled. Never call blob.make_public() here — it raises.
PUBLIC_URL = 'https://storage.googleapis.com/{bucket}/{key}'


def existing_keys(bucket, prefix: str) -> set[str]:
    """List a prefix once rather than probing each blob individually.

    A per-file blob.exists() check would be one API round trip per file; the archive is
    ~4,000 files, so listing the prefix in bulk is the difference between seconds and
    several minutes.
    """
    return {blob.name for blob in bucket.list_blobs(prefix=prefix)}


def collect_data(bucket, skip_existing: bool) -> list[tuple[Path, str]]:
    """Local pageviews JSON → (path, key) pairs that need uploading."""
    present = existing_keys(bucket, DATA_PREFIX) if skip_existing else set()
    pending = []
    skipped = 0

    for path in sorted(DATA_DIR.glob('pageviews_*.json')):
        match = PAGEVIEWS_RE.search(path.name)
        if not match:
            continue
        date = datetime.strptime(match.group(1), '%Y%m%d')
        key = generate_storage_key(date, prefix=DATA_PREFIX)
        if key in present:
            skipped += 1
            continue
        pending.append((path, key))

    print(f'  data:    {len(pending)} to upload, {skipped} already in bucket')
    return pending


def collect_reports() -> list[tuple[Path, str]]:
    """Local HTML reports → (path, key) pairs. Reports always overwrite."""
    pending = [
        (path, f'{REPORTS_PREFIX}/{path.relative_to(REPORTS_DIR).as_posix()}')
        for path in sorted(REPORTS_DIR.rglob('*.html'))
    ]
    print(f'  reports: {len(pending)} to upload (always overwritten)')
    return pending


def collect_derived() -> list[tuple[Path, str]]:
    """Derived artifacts → (path, key) pairs. Always overwrite, like reports."""
    pending = []
    for name in DERIVED_FILES:
        path = DATA_DIR / name
        if path.exists():
            pending.append((path, f'{DERIVED_PREFIX}/{name}'))
        else:
            print(f'  derived: {name} absent locally — skipping')
    print(f'  derived: {len(pending)} to upload (always overwritten)')
    return pending


def upload(bucket, items: list[tuple[Path, str]], content_type: str, execute: bool) -> int:
    total = 0
    for i, (path, key) in enumerate(items, 1):
        size = path.stat().st_size
        total += size
        if not execute:
            print(f'    [dry-run] {key}  ({size:,} bytes)')
            continue
        bucket.blob(key).upload_from_filename(str(path), content_type=content_type)
        if i % 100 == 0 or i == len(items):
            print(f'    {i}/{len(items)} uploaded', end='\r' if i < len(items) else '\n')
    return total


def main():
    parser = argparse.ArgumentParser(description='Sync local data/ and reports/ to GCS')
    parser.add_argument('--data', action='store_true', help='Sync data/*.json (skips dates already in the bucket)')
    parser.add_argument('--reports', action='store_true', help='Sync reports/**/*.html (always overwrites)')
    parser.add_argument('--derived', action='store_true',
                        help='Sync data/narratives_cache.json (always overwrites)')
    parser.add_argument('--all', action='store_true', help='--data, --reports, and --derived')
    parser.add_argument('--execute', action='store_true',
                        help='Actually upload. Without this the script only reports what it would do.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Explicitly request a dry run (this is already the default)')
    parser.add_argument('--bucket', default=os.environ.get('GCS_BUCKET', DEFAULT_BUCKET),
                        help=f'Target bucket (default: $GCS_BUCKET or {DEFAULT_BUCKET})')
    args = parser.parse_args()

    do_data = args.data or args.all
    do_reports = args.reports or args.all
    do_derived = args.derived or args.all
    if not (do_data or do_reports or do_derived):
        parser.error('nothing selected — pass --data, --reports, --derived, or --all')

    execute = args.execute and not args.dry_run

    client = storage.Client()
    bucket = client.bucket(args.bucket)

    print(f'Bucket: gs://{args.bucket}   mode: {"UPLOAD" if execute else "dry run"}')

    data_items = collect_data(bucket, skip_existing=True) if do_data else []
    report_items = collect_reports() if do_reports else []
    derived_items = collect_derived() if do_derived else []

    total = 0
    if data_items:
        print('\nData:')
        total += upload(bucket, data_items, 'application/json', execute)
    if report_items:
        print('\nReports:')
        total += upload(bucket, report_items, 'text/html', execute)
    if derived_items:
        print('\nDerived:')
        total += upload(bucket, derived_items, 'application/json', execute)

    n = len(data_items) + len(report_items) + len(derived_items)
    verb = 'Uploaded' if execute else 'Would upload'
    print(f'\n{verb} {n} file(s), {total/1024/1024:.1f} MB')

    if not execute:
        print('\nNothing was written. Re-run with --execute to publish.')
    elif do_reports:
        index = PUBLIC_URL.format(bucket=args.bucket, key=f'{REPORTS_PREFIX}/index.html')
        print(f'\nLive: {index}')


if __name__ == '__main__':
    main()
