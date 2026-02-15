#!/usr/bin/env python3
"""Find gaps in collected Wikipedia pageview data.

This script processes a list of filenames (usually from 'ls' or 'aws s3 ls')
and identifies missing dates in the sequence.

Usage:
    ls data/ | ./scripts/find-date-gaps.py
    aws s3 ls s3://bucket/data/ --recursive | ./scripts/find-date-gaps.py
"""

import sys
import re
from datetime import datetime, timedelta

def parse_date(date_str):
    """Parse date from various formats."""
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None

def main():
    """
    Process a list of filenames from stdin, extract dates, and find gaps in the sequence.
    """
    # Regular expressions to extract date from filename
    # Supports:
    # - pageviews_20250115.json
    # - pageviews-2025-01-15.json
    patterns = [
        re.compile(r'pageviews_(\d{8})'),
        re.compile(r'pageviews-(\d{4}-\d{2}-\d{2})'),
        re.compile(r'(\d{4}-\d{2}-\d{2})'),
        re.compile(r'(\d{8})'),
    ]

    found_dates = set()

    # Process each line of input
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        current_date = None
        for pattern in patterns:
            match = pattern.search(line)
            if match:
                current_date = parse_date(match.group(1))
                if current_date:
                    found_dates.add(current_date.date())
                    break

    if not found_dates:
        print("No dates found in input.")
        return

    sorted_dates = sorted(list(found_dates))
    first_date = sorted_dates[0]
    last_date = sorted_dates[-1]

    print(f"Range: {first_date} to {last_date}")
    print(f"Total unique dates found: {len(sorted_dates)}")

    expected_count = (last_date - first_date).days + 1
    if len(sorted_dates) == expected_count:
        print("No gaps found!")
        return

    print(f"Expected {expected_count} dates, found {len(sorted_dates)}. Missing {expected_count - len(sorted_dates)} dates.")
    print("\nGaps found:")

    current = first_date
    while current <= last_date:
        if current not in found_dates:
            # Found start of a gap
            gap_start = current
            while current <= last_date and current not in found_dates:
                current += timedelta(days=1)
            gap_end = current - timedelta(days=1)

            if gap_start == gap_end:
                print(f"  Missing: {gap_start}")
            else:
                print(f"  Missing: {gap_start} to {gap_end} ({(gap_end - gap_start).days + 1} days)")
        else:
            current += timedelta(days=1)

if __name__ == "__main__":
    main()
