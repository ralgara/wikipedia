#!/usr/bin/env python3
"""Download Wikipedia pageviews for local testing.

Usage:
    ./scripts/download-pageviews.py                    # Yesterday
    ./scripts/download-pageviews.py 2025-01-15         # Specific date
    ./scripts/download-pageviews.py 2025-01-01 2025-01-07  # Date range
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta

# Add shared library to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
from wikipedia import download_pageviews, generate_storage_key

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def download_date(date: datetime) -> str:
    """Download pageviews for a single date, save to data/ directory."""
    print(f"Downloading pageviews for {date.strftime('%Y-%m-%d')}...")

    data = download_pageviews(date)

    filename = f"pageviews_{date.strftime('%Y%m%d')}.json"
    filepath = os.path.join(DATA_DIR, filename)

    with open(filepath, 'w') as f:
        json.dump(data, f)

    print(f"  Saved {len(data)} articles to {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser(description='Download Wikipedia pageviews')
    parser.add_argument('start_date', nargs='?', help='Start date (YYYY-MM-DD), default: yesterday')
    parser.add_argument('end_date', nargs='?', help='End date (YYYY-MM-DD), default: same as start')
    parser.add_argument('--preview', '-p', action='store_true', help='Preview top 5 articles instead of saving')
    args = parser.parse_args()

    # Parse dates
    if args.start_date:
        start = datetime.strptime(args.start_date, '%Y-%m-%d')
    else:
        start = datetime.utcnow() - timedelta(days=1)

    end = datetime.strptime(args.end_date, '%Y-%m-%d') if args.end_date else start

    # Validate date order
    if start > end:
        print(f"Error: start date ({start.date()}) is after end date ({end.date()})", file=sys.stderr)
        sys.exit(1)

    # Preview mode - just print top articles
    if args.preview:
        data = download_pageviews(start)
        print(json.dumps(data[:5], indent=2))
        return

    # Download mode
    os.makedirs(DATA_DIR, exist_ok=True)

    current = start
    files = []
    errors = []
    while current <= end:
        try:
            filepath = download_date(current)
            files.append(filepath)
        except Exception as e:
            print(f"  Error: {e}", file=sys.stderr)
            errors.append((current.strftime('%Y-%m-%d'), str(e)))
        current += timedelta(days=1)

    print(f"\nDownloaded {len(files)} file(s) to data/")
    if errors:
        print(f"Failed: {len(errors)} date(s)", file=sys.stderr)


if __name__ == '__main__':
    main()
