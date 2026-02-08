#!/usr/bin/env python3
"""Enrich Wikipedia article metadata with Wikidata and DBpedia information.

This script queries external knowledge graphs to add semantic metadata to articles,
creating enriched JSON files that can be used by analysis scripts.

Usage:
    ./scripts/enrich-metadata.py                    # Enrich all articles
    ./scripts/enrich-metadata.py --top-n 1000       # Enrich top 1000 articles
    ./scripts/enrich-metadata.py --resume           # Resume interrupted enrichment
    ./scripts/enrich-metadata.py --dry-run          # Show what would be enriched
"""

import argparse
import json
import logging
import sqlite3
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.wikipedia.ontology import get_enriched_metadata

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.table import Table
    HAVE_RICH = True
except ImportError:
    HAVE_RICH = False
    print("Tip: Install 'rich' for better progress display: pip install rich")

# Configuration
DATA_DIR = Path(__file__).parent.parent / 'data'
ENRICHED_DIR = DATA_DIR / 'enriched_metadata'
DB_PATH = DATA_DIR / 'pageviews.db'

# Rate limiting
REQUESTS_PER_SECOND = 2  # Conservative rate limit
DELAY_BETWEEN_REQUESTS = 1.0 / REQUESTS_PER_SECOND

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_articles_from_db(top_n: Optional[int] = None,
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> List[str]:
    """Get list of articles from SQLite database.
    
    Args:
        top_n: Limit to top N articles by total views
        start_date: Filter by date range start (YYYY-MM-DD)
        end_date: Filter by date range end (YYYY-MM-DD)
    
    Returns:
        List of unique article names
    """
    if not DB_PATH.exists():
        logger.error(f"Database not found: {DB_PATH}")
        logger.error("Run ./scripts/convert-to-sqlite.py first")
        return []
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Build query
    query = "SELECT article, SUM(views) as total_views FROM pageviews"
    params = []
    
    # Add date filters if specified
    if start_date or end_date:
        conditions = []
        if start_date:
            conditions.append("date >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("date <= ?")
            params.append(end_date)
        query += " WHERE " + " AND ".join(conditions)
    
    # Group and order
    query += " GROUP BY article ORDER BY total_views DESC"
    
    # Add limit if specified
    if top_n:
        query += " LIMIT ?"
        params.append(top_n)
    
    cursor.execute(query, params)
    articles = [row[0] for row in cursor.fetchall()]
    
    conn.close()
    return articles


def get_already_enriched() -> set:
    """Get set of articles that have already been enriched.
    
    Returns:
        Set of article names with existing enriched metadata files
    """
    if not ENRICHED_DIR.exists():
        return set()
    
    enriched = set()
    for json_file in ENRICHED_DIR.glob('*.json'):
        # Article name is filename without .json extension
        article = json_file.stem
        enriched.add(article)
    
    return enriched


def save_enriched_metadata(article: str, metadata: Dict) -> None:
    """Save enriched metadata to JSON file.
    
    Args:
        article: Wikipedia article name
        metadata: Enriched metadata dictionary
    """
    ENRICHED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Add enrichment timestamp
    metadata['enriched_at'] = datetime.utcnow().isoformat() + 'Z'
    metadata['article'] = article
    
    # Save to JSON file
    output_path = ENRICHED_DIR / f"{article}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def enrich_articles(articles: List[str],
                    resume: bool = False,
                    dry_run: bool = False) -> Dict[str, int]:
    """Enrich articles with metadata from Wikidata and DBpedia.
    
    Args:
        articles: List of article names to enrich
        resume: Skip already enriched articles
        dry_run: Don't save any data, just show what would be done
    
    Returns:
        Dictionary with statistics
    """
    stats = {
        'total': len(articles),
        'enriched': 0,
        'skipped': 0,
        'failed': 0,
        'no_data': 0
    }
    
    # Get already enriched if resuming
    already_enriched = get_already_enriched() if resume else set()
    
    if HAVE_RICH:
        console = Console()
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Enriching articles...", total=len(articles))
            
            for article in articles:
                # Skip if already enriched
                if article in already_enriched:
                    stats['skipped'] += 1
                    progress.update(task, advance=1)
                    continue
                
                # Update progress description
                progress.update(task, description=f"[cyan]Enriching: {article[:40]}...")
                
                if not dry_run:
                    try:
                        # Rate limiting
                        time.sleep(DELAY_BETWEEN_REQUESTS)
                        
                        # Get enriched metadata
                        metadata = get_enriched_metadata(article)
                        
                        if metadata and metadata.get('wikidata_id'):
                            save_enriched_metadata(article, metadata)
                            stats['enriched'] += 1
                        else:
                            stats['no_data'] += 1
                            logger.warning(f"No metadata found for: {article}")
                    
                    except Exception as e:
                        stats['failed'] += 1
                        logger.error(f"Failed to enrich {article}: {e}")
                else:
                    # Dry run - just simulate
                    stats['enriched'] += 1
                
                progress.update(task, advance=1)
    else:
        # Fallback without rich
        for i, article in enumerate(articles, 1):
            if article in already_enriched:
                stats['skipped'] += 1
                continue
            
            print(f"[{i}/{len(articles)}] Enriching: {article}", end='\r')
            
            if not dry_run:
                try:
                    time.sleep(DELAY_BETWEEN_REQUESTS)
                    metadata = get_enriched_metadata(article)
                    
                    if metadata and metadata.get('wikidata_id'):
                        save_enriched_metadata(article, metadata)
                        stats['enriched'] += 1
                    else:
                        stats['no_data'] += 1
                        logger.warning(f"No metadata found for: {article}")
                
                except Exception as e:
                    stats['failed'] += 1
                    logger.error(f"Failed to enrich {article}: {e}")
            else:
                stats['enriched'] += 1
        
        print()  # New line after progress
    
    return stats


def print_summary(stats: Dict[str, int], dry_run: bool = False) -> None:
    """Print enrichment summary.
    
    Args:
        stats: Statistics dictionary
        dry_run: Whether this was a dry run
    """
    if HAVE_RICH:
        console = Console()
        table = Table(title="Enrichment Summary" + (" (DRY RUN)" if dry_run else ""))
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="magenta", justify="right")
        
        table.add_row("Total articles", str(stats['total']))
        table.add_row("Newly enriched", str(stats['enriched']), style="green")
        table.add_row("Already enriched (skipped)", str(stats['skipped']))
        table.add_row("No metadata found", str(stats['no_data']), style="yellow")
        table.add_row("Failed", str(stats['failed']), style="red" if stats['failed'] > 0 else None)
        
        console.print(table)
    else:
        print("\n" + "="*50)
        print("Enrichment Summary" + (" (DRY RUN)" if dry_run else ""))
        print("="*50)
        print(f"Total articles:              {stats['total']}")
        print(f"Newly enriched:              {stats['enriched']}")
        print(f"Already enriched (skipped):  {stats['skipped']}")
        print(f"No metadata found:           {stats['no_data']}")
        print(f"Failed:                      {stats['failed']}")
    
    if not dry_run:
        print(f"\nEnriched metadata saved to: {ENRICHED_DIR}")


def main():
    parser = argparse.ArgumentParser(
        description='Enrich Wikipedia articles with Wikidata and DBpedia metadata'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        help='Enrich only top N articles by total views'
    )
    parser.add_argument(
        '--start-date',
        help='Filter by date range start (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        help='Filter by date range end (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Skip already enriched articles'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be enriched without saving data'
    )
    
    args = parser.parse_args()
    
    # Get articles to enrich
    print("Loading articles from database...")
    articles = get_articles_from_db(
        top_n=args.top_n,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    if not articles:
        print("No articles found to enrich")
        return 1
    
    print(f"Found {len(articles)} articles to process")
    
    if args.resume:
        already_enriched = get_already_enriched()
        print(f"Already enriched: {len(already_enriched)} articles")
    
    if args.dry_run:
        print("\n⚠️  DRY RUN MODE - No data will be saved ⚠️\n")
    
    # Enrich articles
    stats = enrich_articles(articles, resume=args.resume, dry_run=args.dry_run)
    
    # Print summary
    print_summary(stats, dry_run=args.dry_run)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
