#!/usr/bin/env python3
"""Review articles flagged for manual review.

Usage:
    ./scripts/review-flagged.py list              # Show all flagged articles
    ./scripts/review-flagged.py unhide ARTICLE    # Mark article as legitimate
    ./scripts/review-flagged.py hide ARTICLE      # Manually hide an article
    ./scripts/review-flagged.py hide ARTICLE "reason"  # Hide with custom reason
"""

import argparse
import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / 'data' / 'pageviews.db'

# Dark theme colors for terminal output
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


def list_flagged():
    """Show all flagged articles."""
    if not DB_PATH.exists():
        print(f"{color('Error:', 'error')} Database not found at {DB_PATH}")
        print("Run ./scripts/convert-to-sqlite.py first")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get flagged articles
    cursor.execute("""
        SELECT article, hide_reason, COUNT(*) as appearances, SUM(views) as total_views
        FROM pageviews
        WHERE hide=1 AND hide_reason='flagged_for_review'
        GROUP BY article
        ORDER BY total_views DESC
    """)

    rows = cursor.fetchall()

    if not rows:
        print(f"\n{color('No articles flagged for review', 'success')}")
        conn.close()
        return

    print(f"\n{color('Articles Flagged for Review', 'bold')}")
    print(color('=' * 110, 'muted'))
    print(f"{'Article':<60} {'Appearances':>12} {'Total Views':>15}")
    print(color('-' * 110, 'muted'))

    for article, reason, appearances, total_views in rows:
        print(f"{article:<60} {appearances:>12,} {total_views:>15,}")

    print(color('=' * 110, 'muted'))
    print(f"\n{color('Total:', 'bold')} {len(rows)} articles flagged")

    # Get summary by hide reason
    cursor.execute("""
        SELECT hide_reason, COUNT(DISTINCT article) as article_count, COUNT(*) as record_count
        FROM pageviews
        WHERE hide=1
        GROUP BY hide_reason
        ORDER BY record_count DESC
    """)

    print(f"\n{color('Hide Reason Summary:', 'bold')}")
    for reason, article_count, record_count in cursor.fetchall():
        print(f"  {reason:<25} {article_count:>6} articles, {record_count:>8,} records")

    conn.close()


def unhide_article(article: str):
    """Mark article as legitimate (unhide)."""
    if not DB_PATH.exists():
        print(f"{color('Error:', 'error')} Database not found at {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Check if article exists
    cursor.execute("SELECT COUNT(*) FROM pageviews WHERE article=?", (article,))
    count = cursor.fetchone()[0]

    if count == 0:
        print(f"{color('Error:', 'error')} Article not found: {article}")
        conn.close()
        sys.exit(1)

    # Unhide
    cursor.execute(
        "UPDATE pageviews SET hide=0, hide_reason=NULL WHERE article=?",
        (article,)
    )

    conn.commit()
    rows_updated = cursor.rowcount

    print(f"{color('Success:', 'success')} Unhidden {rows_updated} records for: {color(article, 'info')}")

    conn.close()


def hide_article(article: str, reason: str = "manual_block"):
    """Manually hide an article."""
    if not DB_PATH.exists():
        print(f"{color('Error:', 'error')} Database not found at {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Check if article exists
    cursor.execute("SELECT COUNT(*) FROM pageviews WHERE article=?", (article,))
    count = cursor.fetchone()[0]

    if count == 0:
        print(f"{color('Error:', 'error')} Article not found: {article}")
        conn.close()
        sys.exit(1)

    # Hide
    cursor.execute(
        "UPDATE pageviews SET hide=1, hide_reason=? WHERE article=?",
        (reason, article)
    )

    conn.commit()
    rows_updated = cursor.rowcount

    print(f"{color('Success:', 'success')} Hidden {rows_updated} records for: {color(article, 'info')}")
    print(f"Reason: {color(reason, 'muted')}")

    conn.close()


def main():
    parser = argparse.ArgumentParser(
        description='Review and manage flagged articles',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ./scripts/review-flagged.py list
  ./scripts/review-flagged.py unhide "Article_Name"
  ./scripts/review-flagged.py hide "Article_Name"
  ./scripts/review-flagged.py hide "Article_Name" "spam_content"
        """
    )

    parser.add_argument('command', choices=['list', 'unhide', 'hide'],
                        help='Command to execute')
    parser.add_argument('article', nargs='?',
                        help='Article name (required for unhide/hide)')
    parser.add_argument('reason', nargs='?', default='manual_block',
                        help='Hide reason (optional for hide command)')

    args = parser.parse_args()

    if args.command == 'list':
        list_flagged()
    elif args.command == 'unhide':
        if not args.article:
            print(f"{color('Error:', 'error')} Article name required for unhide command")
            sys.exit(1)
        unhide_article(args.article)
    elif args.command == 'hide':
        if not args.article:
            print(f"{color('Error:', 'error')} Article name required for hide command")
            sys.exit(1)
        hide_article(args.article, args.reason)


if __name__ == '__main__':
    main()
