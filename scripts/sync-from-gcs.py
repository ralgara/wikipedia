#!/usr/bin/env python3
"""Seed the local archive from GCS — the mirror image of upload-to-gcs.py.

The bucket is the durable copy of the archive; a machine's local data/ is a cache. On a fresh
host (or a recreated Spot VM) data/ is empty, and download-pageviews.py would happily re-fetch
all ~4,000 days from the Wikimedia API. Pulling them from GCS instead is faster, free within
the same region, and does not hammer a public API for data we already have.

    uv run python scripts/sync-from-gcs.py              # dry run (the default)
    uv run python scripts/sync-from-gcs.py --execute

Idempotent and cheap once warm: one bucket listing, then downloads only what is missing.

Environment:
    GCS_BUCKET                      source bucket (default: wikipedia-cortex-data)
    GOOGLE_APPLICATION_CREDENTIALS  optional; Application Default Credentials are used otherwise
"""

import argparse
import os
import sys
from pathlib import Path

from google.cloud import storage

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from shared.wikipedia.storage import parse_storage_key

DATA_DIR = ROOT / 'data'
DEFAULT_BUCKET = 'wikipedia-cortex-data'
DATA_PREFIX = 'wikipedia/pageviews'


def main():
    parser = argparse.ArgumentParser(description='Seed local data/ from GCS')
    parser.add_argument('--execute', action='store_true',
                        help='Actually download. Without this the script only reports what it would do.')
    parser.add_argument('--bucket', default=os.environ.get('GCS_BUCKET', DEFAULT_BUCKET))
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    args = parser.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)

    client = storage.Client()
    bucket = client.bucket(args.bucket)

    print(f'Source: gs://{args.bucket}/{DATA_PREFIX}   mode: {"DOWNLOAD" if args.execute else "dry run"}')

    pending = []
    seen = 0
    for blob in bucket.list_blobs(prefix=DATA_PREFIX):
        date = parse_storage_key(blob.name)
        if date is None:
            continue
        seen += 1
        local = args.data_dir / f'pageviews_{date.strftime("%Y%m%d")}.json'
        if not local.exists():
            pending.append((blob, local))

    print(f'  {seen} in bucket, {seen - len(pending)} already local, {len(pending)} to download')

    if not pending:
        print('\nLocal archive is already in sync.')
        return

    if not args.execute:
        for _, local in pending[:10]:
            print(f'    [dry-run] {local.name}')
        if len(pending) > 10:
            print(f'    … and {len(pending) - 10} more')
        print('\nNothing was written. Re-run with --execute to download.')
        return

    for i, (blob, local) in enumerate(pending, 1):
        blob.download_to_filename(str(local))
        if i % 100 == 0 or i == len(pending):
            print(f'    {i}/{len(pending)} downloaded', end='\r' if i < len(pending) else '\n')

    print(f'\nDownloaded {len(pending)} file(s) → {args.data_dir}')


if __name__ == '__main__':
    main()
