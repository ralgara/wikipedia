#!/usr/bin/env python3
"""Download Wikipedia pageviews to local data directory.

Modes:
    Fill gaps (default — no positional args):
        Scans data/ for existing files, downloads every missing date from
        START to (today - lag). Use this to keep the local archive current.

        ./scripts/download-pageviews.py
        ./scripts/download-pageviews.py --start 2020-01-01
        ./scripts/download-pageviews.py --lag 5

    Explicit range:
        ./scripts/download-pageviews.py 2025-01-15
        ./scripts/download-pageviews.py 2025-01-01 2025-01-07

    Preview:
        ./scripts/download-pageviews.py --preview 2025-01-15
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timedelta

# Add shared library to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
from wikipedia import download_pageviews, generate_storage_key

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

# Wikimedia REST API top-1000 pageviews start date
ARCHIVE_START = datetime(2015, 7, 1)

# Default lag: Wikimedia API typically lags 2-3 days; use 3 to be safe
DEFAULT_LAG = 3


def get_filepath(date: datetime) -> str:
    filename = f"pageviews_{date.strftime('%Y%m%d')}.json"
    return os.path.join(DATA_DIR, filename)


def local_dates() -> set:
    """Return the set of dates already present in DATA_DIR."""
    pattern = re.compile(r'pageviews_(\d{8})\.json$')
    dates = set()
    if not os.path.isdir(DATA_DIR):
        return dates
    for name in os.listdir(DATA_DIR):
        m = pattern.match(name)
        if m:
            try:
                dates.add(datetime.strptime(m.group(1), '%Y%m%d'))
            except ValueError:
                pass
    return dates


def missing_dates(start: datetime, end: datetime) -> list:
    """Return sorted list of dates in [start, end] not present locally."""
    have = local_dates()
    missing = []
    current = start
    while current <= end:
        if current not in have:
            missing.append(current)
        current += timedelta(days=1)
    return missing


def download_date(date: datetime) -> str:
    """Download pageviews for a single date and save to data/."""
    print(f"Downloading {date.strftime('%Y-%m-%d')}...")
    data = download_pageviews(date)
    filepath = get_filepath(date)
    with open(filepath, 'w') as f:
        json.dump(data, f)
    print(f"  {len(data)} articles → {os.path.basename(filepath)}")
    return filepath


def run_downloads(dates: list, force: bool = False) -> tuple:
    """Download a list of dates. Returns (downloaded, skipped, errors)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    downloaded, skipped, errors = [], 0, []
    for date in dates:
        filepath = get_filepath(date)
        if not force and os.path.exists(filepath):
            skipped += 1
            continue
        try:
            downloaded.append(download_date(date))
        except Exception as e:
            print(f"  Error {date.strftime('%Y-%m-%d')}: {e}", file=sys.stderr)
            errors.append((date.strftime('%Y-%m-%d'), str(e)))
    return downloaded, skipped, errors


def print_summary(downloaded, skipped, errors):
    print(f"\nDownloaded: {len(downloaded)}  Skipped: {skipped}  Failed: {len(errors)}")
    if errors:
        print("Failed dates:", file=sys.stderr)
        for date_str, err in errors:
            print(f"  {date_str}: {err}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description='Download Wikipedia pageviews')
    parser.add_argument('start_date', nargs='?',
                        help='Start date (YYYY-MM-DD). Omit to fill all gaps.')
    parser.add_argument('end_date', nargs='?',
                        help='End date (YYYY-MM-DD). Defaults to start_date.')
    parser.add_argument('--start', default=None,
                        help=f'Gap-fill start date (default: {ARCHIVE_START.date()})')
    parser.add_argument('--lag', type=int, default=DEFAULT_LAG,
                        help=f'Days before today to stop (default: {DEFAULT_LAG}, Wikimedia lag)')
    parser.add_argument('--preview', '-p', action='store_true',
                        help='Print top 5 articles for start_date without saving')
    parser.add_argument('--force', '-f', action='store_true',
                        help='Re-download even if file already exists')
    parser.add_argument('--dry-run', action='store_true',
                        help='List missing dates without downloading')
    args = parser.parse_args()

    # --- Preview mode ---
    if args.preview:
        date = datetime.strptime(args.start_date, '%Y-%m-%d') if args.start_date \
            else datetime.utcnow() - timedelta(days=1)
        data = download_pageviews(date)
        print(json.dumps(data[:5], indent=2))
        return

    # --- Explicit range mode ---
    if args.start_date:
        start = datetime.strptime(args.start_date, '%Y-%m-%d')
        end = datetime.strptime(args.end_date, '%Y-%m-%d') if args.end_date else start
        if start > end:
            print(f"Error: start ({start.date()}) is after end ({end.date()})", file=sys.stderr)
            sys.exit(1)
        current = start
        dates = []
        while current <= end:
            dates.append(current)
            current += timedelta(days=1)
        downloaded, skipped, errors = run_downloads(dates, force=args.force)
        print_summary(downloaded, skipped, errors)
        return

    # --- Gap-fill mode ---
    fill_start = datetime.strptime(args.start, '%Y-%m-%d') if args.start else ARCHIVE_START
    fill_end = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0) \
        - timedelta(days=args.lag)

    gaps = missing_dates(fill_start, fill_end)

    if not gaps:
        print(f"Archive complete: {fill_start.date()} → {fill_end.date()} (no gaps)")
        return

    print(f"Gap fill: {fill_start.date()} → {fill_end.date()}")
    print(f"Missing: {len(gaps)} date(s)")
    if gaps:
        print(f"  First: {gaps[0].date()}  Last: {gaps[-1].date()}")

    if args.dry_run:
        for d in gaps:
            print(f"  {d.date()}")
        return

    downloaded, skipped, errors = run_downloads(gaps, force=args.force)
    print_summary(downloaded, skipped, errors)


if __name__ == '__main__':
    main()
