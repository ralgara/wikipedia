#!/usr/bin/env python3
"""Convert JSON pageview files to SQLite database.

Usage:
    ./scripts/convert-to-sqlite.py                    # Create database
    ./scripts/convert-to-sqlite.py --force            # Overwrite existing
"""

import argparse
import json
import os
import sqlite3
import sys
from glob import glob
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / 'data'
DB_PATH = DATA_DIR / 'pageviews.db'

# Dark theme colors for terminal output (matching other scripts)
COLORS = {
    'info': '\033[36m',      # Cyan
    'success': '\033[32m',   # Green
    'warning': '\033[33m',   # Yellow
    'error': '\033[31m',     # Red
    'muted': '\033[90m',     # Gray
    'reset': '\033[0m',
    'bold': '\033[1m',
}


def color(text: str, style: str) -> str:
    """Apply color to text for terminal output."""
    return f"{COLORS.get(style, '')}{text}{COLORS['reset']}"


def format_bytes(size: int) -> str:
    """Format byte size for human readability."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def create_database(db_path: Path) -> sqlite3.Connection:
    """Create SQLite database with pageviews table."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pageviews (
            article TEXT NOT NULL,
            views INTEGER NOT NULL,
            rank INTEGER NOT NULL,
            date TEXT NOT NULL
        )
    ''')

    conn.commit()
    return conn


def create_indexes(conn: sqlite3.Connection) -> dict:
    """Create indexes for optimal query performance."""
    cursor = conn.cursor()

    indexes = {
        'idx_article_date': '(article, date)',      # Article X over time
        'idx_date': '(date)',                        # Top articles on date Y
        'idx_date_views': '(date, views DESC)',      # Ranking queries
        'idx_article': '(article)',                  # Article lookups
    }

    print(f"\n{color('Creating indexes...', 'info')}")

    for idx_name, columns in indexes.items():
        print(f"  Creating {color(idx_name, 'muted')} on {columns}...")
        cursor.execute(f'CREATE INDEX IF NOT EXISTS {idx_name} ON pageviews {columns}')

    conn.commit()
    return indexes


def get_index_sizes(conn: sqlite3.Connection) -> dict:
    """Get size information for each index."""
    cursor = conn.cursor()

    # Get page size and page counts for indexes
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='pageviews'")
    index_names = [row[0] for row in cursor.fetchall()]

    # SQLite doesn't directly expose index sizes, but we can get page counts
    cursor.execute("PRAGMA page_size")
    page_size = cursor.fetchone()[0]

    # Analyze to get accurate stats
    cursor.execute("ANALYZE")

    sizes = {}
    for idx_name in index_names:
        # Use sqlite_stat1 if available
        try:
            cursor.execute(f"SELECT stat FROM sqlite_stat1 WHERE idx=?", (idx_name,))
            row = cursor.fetchone()
            if row:
                sizes[idx_name] = row[0]
        except sqlite3.OperationalError:
            sizes[idx_name] = 'N/A'

    return sizes


def load_json_files() -> list:
    """Find all pageview JSON files."""
    pattern = str(DATA_DIR / 'pageviews_*.json')
    files = sorted(glob(pattern))
    return files


def main():
    parser = argparse.ArgumentParser(description='Convert pageview JSON files to SQLite')
    parser.add_argument('--force', '-f', action='store_true', help='Overwrite existing database')
    args = parser.parse_args()

    print(f"\n{color('Wikipedia Pageviews - JSON to SQLite Converter', 'bold')}")
    print(color('=' * 50, 'muted'))

    # Check for existing database
    if DB_PATH.exists():
        if not args.force:
            print(f"\n{color('Error:', 'error')} Database already exists at {DB_PATH}")
            print(f"Use {color('--force', 'warning')} to overwrite")
            sys.exit(1)
        else:
            print(f"\n{color('Removing existing database...', 'warning')}")
            os.remove(DB_PATH)

    # Find JSON files
    json_files = load_json_files()
    if not json_files:
        print(f"\n{color('Error:', 'error')} No pageview files found in {DATA_DIR}")
        print("Run ./scripts/download-pageviews.py first")
        sys.exit(1)

    print(f"\n{color('Found', 'info')} {len(json_files)} JSON files to convert")

    # Create database
    print(f"{color('Creating database...', 'info')}")
    conn = create_database(DB_PATH)
    cursor = conn.cursor()

    # Process files
    total_records = 0
    print(f"\n{color('Loading data:', 'info')}")

    for i, filepath in enumerate(json_files, 1):
        filename = os.path.basename(filepath)

        with open(filepath) as f:
            data = json.load(f)

        # Insert records
        cursor.executemany(
            'INSERT INTO pageviews (article, views, rank, date) VALUES (?, ?, ?, ?)',
            [(r['article'], r['views'], r['rank'], r['date']) for r in data]
        )

        total_records += len(data)

        # Progress reporting
        progress = i / len(json_files) * 100
        bar_width = 30
        filled = int(bar_width * i / len(json_files))
        bar = '=' * filled + '-' * (bar_width - filled)

        print(f"\r  [{color(bar, 'info')}] {progress:5.1f}% | {filename} ({len(data):,} records)", end='')

    conn.commit()
    print(f"\n\n{color('Inserted', 'success')} {total_records:,} total records")

    # Create indexes
    indexes = create_indexes(conn)

    # Get stats
    print(f"\n{color('Analyzing database...', 'info')}")
    cursor.execute("SELECT COUNT(*) FROM pageviews")
    record_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT date) FROM pageviews")
    date_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT article) FROM pageviews")
    article_count = cursor.fetchone()[0]

    cursor.execute("SELECT MIN(date), MAX(date) FROM pageviews")
    date_range = cursor.fetchone()

    # Get index stats
    index_sizes = get_index_sizes(conn)

    conn.close()

    # File size
    db_size = DB_PATH.stat().st_size

    # Print summary
    print(f"\n{color('=' * 50, 'muted')}")
    print(f"{color('Conversion Complete!', 'success')}")
    print(f"{color('=' * 50, 'muted')}")

    print(f"\n{color('Database:', 'bold')} {DB_PATH}")
    print(f"{color('File size:', 'bold')} {format_bytes(db_size)}")

    print(f"\n{color('Statistics:', 'bold')}")
    print(f"  Total records:    {record_count:,}")
    print(f"  Unique articles:  {article_count:,}")
    print(f"  Days covered:     {date_count}")
    print(f"  Date range:       {date_range[0]} to {date_range[1]}")

    print(f"\n{color('Indexes:', 'bold')}")
    for idx_name, columns in indexes.items():
        stat = index_sizes.get(idx_name, 'N/A')
        print(f"  {color(idx_name, 'info'):40} {columns:25} {color(f'[{stat}]', 'muted')}")

    print(f"\n{color('Example queries:', 'muted')}")
    print(f"  # Article over time (uses idx_article_date)")
    print(f"  sqlite3 {DB_PATH.name} \"SELECT date, views FROM pageviews WHERE article='Python_(programming_language)' ORDER BY date\"")
    print(f"\n  # Top articles on a date (uses idx_date_views)")
    print(f"  sqlite3 {DB_PATH.name} \"SELECT article, views FROM pageviews WHERE date='2025-01-01' ORDER BY views DESC LIMIT 10\"")

    print()


if __name__ == '__main__':
    main()
