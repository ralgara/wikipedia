#!/usr/bin/env python3
"""Generate evaluation cases from the SQLite pageviews database.

Produces 50+ YAML cases across 5 question types: comparison, trend, spike,
ranking, and edge_case. Each case includes a mock data snapshot extracted
from real data and ground-truth answers computed deterministically.

Usage:
    ./agents/evals/generate-cases.py                     # Generate all cases
    ./agents/evals/generate-cases.py --dry-run            # Preview without saving
    ./agents/evals/generate-cases.py --type comparison    # Single type
    ./agents/evals/generate-cases.py --seed 42            # Reproducible selection
    ./agents/evals/generate-cases.py --db path/to/db      # Custom DB path
"""

import argparse
import json
import math
import os
import random
import sqlite3
import sys
from pathlib import Path

# Allow imports from project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import yaml
except ImportError:
    print("pyyaml required: uv pip install pyyaml", file=sys.stderr)
    sys.exit(1)

DEFAULT_DB = PROJECT_ROOT / "data" / "pageviews.db"
OUTPUT_DIR = SCRIPT_DIR / "cases" / "_generated"

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


def get_db(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        print(f"Database not found: {db_path}", file=sys.stderr)
        print("Run ./scripts/convert-to-sqlite.py first.", file=sys.stderr)
        sys.exit(1)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def top_articles(conn: sqlite3.Connection, limit: int = 100) -> list[str]:
    """Articles with most total views (content only)."""
    rows = conn.execute(
        "SELECT article, SUM(views) as total FROM pageviews "
        "WHERE hide=0 GROUP BY article ORDER BY total DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [r["article"] for r in rows]


def monthly_views(
    conn: sqlite3.Connection, article: str, year: str
) -> dict[str, int]:
    """Monthly aggregated views for an article in a given year."""
    rows = conn.execute(
        "SELECT substr(date,1,7) as month, SUM(views) as total "
        "FROM pageviews WHERE hide=0 AND article=? AND date LIKE ? "
        "GROUP BY month ORDER BY month",
        (article, f"{year}%"),
    ).fetchall()
    return {r["month"]: r["total"] for r in rows}


def yearly_views(
    conn: sqlite3.Connection, article: str
) -> dict[str, int]:
    """Yearly aggregated views for an article."""
    rows = conn.execute(
        "SELECT substr(date,1,4) as year, SUM(views) as total "
        "FROM pageviews WHERE hide=0 AND article=? "
        "GROUP BY year ORDER BY year",
        (article,),
    ).fetchall()
    return {r["year"]: r["total"] for r in rows}


def daily_views(
    conn: sqlite3.Connection, article: str
) -> list[tuple[str, int]]:
    """All daily views for an article, sorted by date."""
    rows = conn.execute(
        "SELECT date, views FROM pageviews WHERE hide=0 AND article=? "
        "ORDER BY date",
        (article,),
    ).fetchall()
    return [(r["date"], r["views"]) for r in rows]


def top_articles_in_range(
    conn: sqlite3.Connection, start: str, end: str, limit: int = 10
) -> list[dict]:
    """Top articles by total views in a date range."""
    rows = conn.execute(
        "SELECT article, SUM(views) as total, COUNT(*) as days "
        "FROM pageviews WHERE hide=0 AND date >= ? AND date <= ? "
        "GROUP BY article ORDER BY total DESC LIMIT ?",
        (start, end, limit),
    ).fetchall()
    return [{"article": r["article"], "total_views": r["total"], "days": r["days"]} for r in rows]


def available_years(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT DISTINCT substr(date,1,4) as year FROM pageviews ORDER BY year"
    ).fetchall()
    return [r["year"] for r in rows]


# ---------------------------------------------------------------------------
# Spike detection
# ---------------------------------------------------------------------------


def find_spikes(
    conn: sqlite3.Connection, article: str, threshold: float = 3.0
) -> list[dict]:
    """Find dates where views exceed mean + threshold * stddev."""
    data = daily_views(conn, article)
    if len(data) < 30:
        return []
    views = [v for _, v in data]
    mean = sum(views) / len(views)
    variance = sum((v - mean) ** 2 for v in views) / len(views)
    stddev = math.sqrt(variance) if variance > 0 else 1
    cutoff = mean + threshold * stddev
    spikes = []
    for date, v in data:
        if v > cutoff:
            spikes.append({
                "date": date,
                "views": v,
                "mean": round(mean),
                "stddev": round(stddev),
                "z_score": round((v - mean) / stddev, 1),
            })
    return spikes


def trend_direction(yearly: dict[str, int], recent_years: int = 3) -> str:
    """Classify trend as increasing, decreasing, or stable over recent years."""
    years = sorted(yearly.keys())
    if len(years) < 2:
        return "insufficient_data"
    recent = years[-recent_years:] if len(years) >= recent_years else years
    vals = [yearly[y] for y in recent]
    if len(vals) < 2:
        return "insufficient_data"
    first_half = sum(vals[: len(vals) // 2]) / max(len(vals) // 2, 1)
    second_half = sum(vals[len(vals) // 2 :]) / max(len(vals) - len(vals) // 2, 1)
    ratio = second_half / first_half if first_half > 0 else 1
    if ratio > 1.2:
        return "increasing"
    elif ratio < 0.8:
        return "decreasing"
    return "stable"


# ---------------------------------------------------------------------------
# Case generators
# ---------------------------------------------------------------------------

COMPARISON_TEMPLATES = [
    "Compare pageviews for {a} vs {b} in {year}",
    "How do {a} and {b} compare in Wikipedia popularity during {year}?",
    "Which had more Wikipedia pageviews in {year}: {a} or {b}?",
]

TREND_TEMPLATES = [
    "How have {a} pageviews trended since {start_year}?",
    "What is the pageview trend for {a} from {start_year} to {end_year}?",
    "Show the Wikipedia traffic trend for {a} over the past {n_years} years",
]

SPIKE_TEMPLATES = [
    "Why did {a} spike on Wikipedia around {spike_month}?",
    "What caused the surge in {a} pageviews in {spike_month}?",
    "{a} had an unusual traffic spike in {spike_month}. What happened?",
]

RANKING_TEMPLATES = [
    "What were the top {n} Wikipedia articles in {month}?",
    "Which articles got the most pageviews on Wikipedia in {month}?",
    "Rank the most viewed Wikipedia articles for {month}",
]


def generate_comparison_cases(
    conn: sqlite3.Connection, rng: random.Random, count: int = 15
) -> list[dict]:
    articles = top_articles(conn, 60)
    years = available_years(conn)
    # pick years with full data
    full_years = [y for y in years if y not in (years[0], years[-1])]
    cases = []
    pairs_used = set()
    for _ in range(count * 3):  # oversample, deduplicate
        if len(cases) >= count:
            break
        a, b = rng.sample(articles, 2)
        pair_key = tuple(sorted([a, b]))
        if pair_key in pairs_used:
            continue
        year = rng.choice(full_years) if full_years else years[-2]
        ma = monthly_views(conn, a, year)
        mb = monthly_views(conn, b, year)
        if not ma or not mb:
            continue
        pairs_used.add(pair_key)
        total_a = sum(ma.values())
        total_b = sum(mb.values())
        higher = a if total_a >= total_b else b
        ratio = max(total_a, total_b) / max(min(total_a, total_b), 1)

        template = rng.choice(COMPARISON_TEMPLATES)
        question = template.format(a=a.replace("_", " "), b=b.replace("_", " "), year=year)

        cases.append({
            "id": f"comparison_{a.lower()}_{b.lower()}_{year}",
            "type": "comparison",
            "description": f"Compare {a} vs {b} in {year}",
            "question": question,
            "mock_data_inline": {
                "pageviews": {a: ma, b: mb},
                "ontology": None,
            },
            "ground_truth": {
                "total_views": {a: total_a, b: total_b},
                "ratio": round(ratio, 2),
                "higher_article": higher,
            },
            "expected": {
                "planner": {
                    "required_articles": sorted([a, b]),
                    "valid_json": True,
                },
                "retrieval": {
                    "valid_json": True,
                    "must_identify_higher": higher,
                    "ratio_tolerance": 0.2,
                },
                "synthesis": {
                    "must_contain_table": True,
                    "must_name_articles": sorted([a, b]),
                    "min_word_count": 100,
                },
            },
        })
    return cases


def generate_trend_cases(
    conn: sqlite3.Connection, rng: random.Random, count: int = 10
) -> list[dict]:
    articles = top_articles(conn, 40)
    cases = []
    used = set()
    for _ in range(count * 3):
        if len(cases) >= count:
            break
        a = rng.choice(articles)
        if a in used:
            continue
        yv = yearly_views(conn, a)
        if len(yv) < 3:
            continue
        used.add(a)
        years_list = sorted(yv.keys())
        start_year = rng.choice(years_list[: len(years_list) // 2])
        end_year = years_list[-1]
        n_years = int(end_year) - int(start_year)
        direction = trend_direction(yv)

        template = rng.choice(TREND_TEMPLATES)
        question = template.format(
            a=a.replace("_", " "),
            start_year=start_year,
            end_year=end_year,
            n_years=n_years,
        )

        # slice yearly views to the requested range
        sliced = {y: v for y, v in yv.items() if y >= start_year}

        cases.append({
            "id": f"trend_{a.lower()}_{start_year}",
            "type": "trend",
            "description": f"Trend for {a} since {start_year}",
            "question": question,
            "mock_data_inline": {
                "pageviews": {a: sliced},
                "ontology": None,
            },
            "ground_truth": {
                "yearly_views": sliced,
                "trend_direction": direction,
                "start_year": start_year,
                "end_year": end_year,
            },
            "expected": {
                "planner": {
                    "required_articles": [a],
                    "valid_json": True,
                },
                "retrieval": {
                    "valid_json": True,
                    "must_identify_trend": direction,
                },
                "synthesis": {
                    "must_contain_table": True,
                    "must_name_articles": [a],
                    "min_word_count": 100,
                },
            },
        })
    return cases


def generate_spike_cases(
    conn: sqlite3.Connection, rng: random.Random, count: int = 10
) -> list[dict]:
    articles = top_articles(conn, 80)
    rng.shuffle(articles)
    cases = []
    for a in articles:
        if len(cases) >= count:
            break
        spikes = find_spikes(conn, a, threshold=4.0)
        if not spikes:
            continue
        # pick the biggest spike
        spike = max(spikes, key=lambda s: s["z_score"])
        spike_month = spike["date"][:7]  # YYYY-MM

        template = rng.choice(SPIKE_TEMPLATES)
        question = template.format(
            a=a.replace("_", " "), spike_month=spike_month
        )

        # get monthly data around the spike (3 months before/after)
        year = spike["date"][:4]
        mv = monthly_views(conn, a, year)

        cases.append({
            "id": f"spike_{a.lower()}_{spike['date']}",
            "type": "spike",
            "description": f"Spike in {a} on {spike['date']}",
            "question": question,
            "mock_data_inline": {
                "pageviews": {a: mv},
                "ontology": None,
            },
            "ground_truth": {
                "spike_date": spike["date"],
                "spike_views": spike["views"],
                "mean_views": spike["mean"],
                "z_score": spike["z_score"],
                "spike_month": spike_month,
            },
            "expected": {
                "planner": {
                    "required_articles": [a],
                    "valid_json": True,
                },
                "retrieval": {
                    "valid_json": True,
                    "must_identify_spike_month": spike_month,
                },
                "synthesis": {
                    "must_name_articles": [a],
                    "min_word_count": 80,
                },
            },
        })
    return cases


def generate_ranking_cases(
    conn: sqlite3.Connection, rng: random.Random, count: int = 10
) -> list[dict]:
    years = available_years(conn)
    full_years = [y for y in years if y not in (years[0], years[-1])]
    cases = []
    used_months = set()
    for _ in range(count * 3):
        if len(cases) >= count:
            break
        year = rng.choice(full_years) if full_years else years[-2]
        month_num = rng.randint(1, 12)
        month = f"{year}-{month_num:02d}"
        if month in used_months:
            continue
        used_months.add(month)

        start = f"{month}-01"
        end = f"{month}-31"  # sqlite string comparison handles this fine
        n = rng.choice([5, 10])
        top = top_articles_in_range(conn, start, end, n)
        if len(top) < n:
            continue

        template = rng.choice(RANKING_TEMPLATES)
        question = template.format(n=n, month=month)

        # build mock data: monthly views for top articles
        mock_pv = {}
        for item in top:
            mv = monthly_views(conn, item["article"], year)
            mock_pv[item["article"]] = mv

        cases.append({
            "id": f"ranking_{month}_top{n}",
            "type": "ranking",
            "description": f"Top {n} articles in {month}",
            "question": question,
            "mock_data_inline": {
                "pageviews": mock_pv,
                "ontology": None,
            },
            "ground_truth": {
                "top_articles": [t["article"] for t in top],
                "top_views": {t["article"]: t["total_views"] for t in top},
                "month": month,
                "n": n,
            },
            "expected": {
                "planner": {
                    "valid_json": True,
                },
                "retrieval": {
                    "valid_json": True,
                    "must_include_top_1": top[0]["article"],
                },
                "synthesis": {
                    "must_contain_table": True,
                    "min_word_count": 80,
                },
            },
        })
    return cases


def generate_edge_cases(rng: random.Random) -> list[dict]:
    """Hand-designed edge cases that test graceful degradation."""
    return [
        {
            "id": "edge_no_articles",
            "type": "edge_case",
            "description": "Question with no specific articles",
            "question": "Which is more popular on Wikipedia?",
            "mock_data_inline": {"pageviews": {}, "ontology": None},
            "ground_truth": {"expected_behavior": "ask_for_clarification_or_refuse"},
            "expected": {
                "planner": {"valid_json": True},
                "synthesis": {"min_word_count": 20},
            },
        },
        {
            "id": "edge_unknown_article",
            "type": "edge_case",
            "description": "Question about nonexistent article",
            "question": "How many pageviews did Xyzzyplugh_The_Unknowable get last year?",
            "mock_data_inline": {"pageviews": {}, "ontology": None},
            "ground_truth": {"expected_behavior": "acknowledge_missing_data"},
            "expected": {
                "planner": {"valid_json": True},
                "synthesis": {"min_word_count": 20},
            },
        },
        {
            "id": "edge_single_day",
            "type": "edge_case",
            "description": "Question about a single specific date",
            "question": "What was the most viewed Wikipedia article on January 1, 2025?",
            "mock_data_inline": {"pageviews": {}, "ontology": None},
            "ground_truth": {"expected_behavior": "handle_single_date"},
            "expected": {
                "planner": {"valid_json": True},
                "synthesis": {"min_word_count": 20},
            },
        },
        {
            "id": "edge_future_date",
            "type": "edge_case",
            "description": "Question about future dates",
            "question": "What will be the most popular Wikipedia article in December 2030?",
            "mock_data_inline": {"pageviews": {}, "ontology": None},
            "ground_truth": {"expected_behavior": "refuse_or_speculate_clearly"},
            "expected": {
                "planner": {"valid_json": True},
                "synthesis": {"min_word_count": 20},
            },
        },
        {
            "id": "edge_three_articles",
            "type": "edge_case",
            "description": "Comparison with three articles instead of two",
            "question": "Compare Wikipedia pageviews for Python, JavaScript, and Rust in 2024",
            "mock_data_inline": {"pageviews": {}, "ontology": None},
            "ground_truth": {"expected_behavior": "handle_multi_article"},
            "expected": {
                "planner": {
                    "valid_json": True,
                    "min_tasks": 2,
                },
                "synthesis": {"min_word_count": 80},
            },
        },
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def write_case(case: dict, output_dir: Path, dry_run: bool = False) -> None:
    filepath = output_dir / f"{case['id']}.yaml"
    if dry_run:
        print(f"  [dry-run] Would write: {filepath.name}")
        return
    with open(filepath, "w") as f:
        yaml.dump(case, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def main():
    parser = argparse.ArgumentParser(description="Generate eval cases from SQLite DB")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Path to pageviews.db")
    parser.add_argument("--type", choices=["comparison", "trend", "spike", "ranking", "edge_case"],
                        help="Generate only this type")
    parser.add_argument("--count", type=int, help="Override count per type")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for reproducibility")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing files")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    generators = {
        "comparison": ("comparison", 15),
        "trend": ("trend", 10),
        "spike": ("spike", 10),
        "ranking": ("ranking", 10),
        "edge_case": ("edge_case", 5),
    }

    if args.type:
        generators = {args.type: generators[args.type]}

    all_cases = []

    # Edge cases don't need DB
    if "edge_case" in generators:
        edge_count = args.count if args.count else generators["edge_case"][1]
        edge = generate_edge_cases(rng)[:edge_count]
        all_cases.extend(edge)
        print(f"  edge_case: {len(edge)} cases")

    # DB-backed generators
    db_types = {k: v for k, v in generators.items() if k != "edge_case"}
    if db_types:
        conn = get_db(args.db)
        for gen_type, (_, default_count) in db_types.items():
            count = args.count if args.count else default_count
            if gen_type == "comparison":
                cases = generate_comparison_cases(conn, rng, count)
            elif gen_type == "trend":
                cases = generate_trend_cases(conn, rng, count)
            elif gen_type == "spike":
                cases = generate_spike_cases(conn, rng, count)
            elif gen_type == "ranking":
                cases = generate_ranking_cases(conn, rng, count)
            else:
                cases = []
            all_cases.extend(cases)
            print(f"  {gen_type}: {len(cases)} cases")
        conn.close()

    print(f"\nTotal: {len(all_cases)} cases")

    if not args.dry_run:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for case in all_cases:
        write_case(case, OUTPUT_DIR, dry_run=args.dry_run)

    if not args.dry_run:
        print(f"Written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
