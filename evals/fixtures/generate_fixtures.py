#!/usr/bin/env python3
"""Generate sample pageview data fixtures for evaluation testing.

Creates realistic Wikipedia pageview data with known properties
(spikes, consistent articles, weekend patterns) so evaluation
results can be validated.
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent

# Seed for reproducibility
random.seed(42)

# Articles with known behaviors
CONSISTENT_ARTICLES = [
    "YouTube", "Facebook", "Instagram", "Twitter", "Google",
    "ChatGPT", "TikTok", "Wikipedia", "Amazon_(company)", "Netflix",
]

SPIKE_ARTICLES = {
    "Super_Bowl_LVIII": {"spike_day": 10, "base": 5000, "spike": 500000},
    "Solar_eclipse": {"spike_day": 20, "base": 3000, "spike": 800000},
    "Academy_Awards": {"spike_day": 25, "base": 8000, "spike": 600000},
}

NON_CONTENT = [
    "Main_Page", "Special:Search", "Wikipedia:About",
    "User:Example", "Template:Cite_web", "Category:Living_people",
    "Portal:Science", "Help:Contents",
]

FLAGGED = ["Pornography", "XXX_(film_series)"]

REGULAR_ARTICLES = [
    "Python_(programming_language)", "Taylor_Swift", "Deaths_in_2025",
    "United_States", "India", "Elon_Musk", "Donald_Trump",
    "OpenAI", "Bitcoin", "COVID-19_pandemic",
    "Cristiano_Ronaldo", "Lionel_Messi", "NASA",
    "Artificial_intelligence", "Climate_change",
    "Ukraine", "Gaza", "Olympic_Games",
    "The_Beatles", "Star_Wars", "Marvel_Cinematic_Universe",
]


def generate_day(date: datetime, day_offset: int) -> list[dict]:
    """Generate pageview data for a single day."""
    records = []
    dow = date.weekday()
    weekend_factor = 0.85 if dow >= 5 else 1.0

    rank = 1

    # Non-content pages (should be filtered)
    for article in NON_CONTENT:
        views = int(random.gauss(2_000_000, 200_000) * weekend_factor)
        records.append({
            "article": article,
            "views": max(views, 100_000),
            "rank": rank,
            "date": date.strftime("%Y-%m-%d"),
        })
        rank += 1

    # Flagged articles
    for article in FLAGGED:
        views = int(random.gauss(50_000, 10_000) * weekend_factor)
        records.append({
            "article": article,
            "views": max(views, 5000),
            "rank": rank,
            "date": date.strftime("%Y-%m-%d"),
        })
        rank += 1

    # Consistent articles (appear every day)
    for article in CONSISTENT_ARTICLES:
        views = int(random.gauss(200_000, 30_000) * weekend_factor)
        records.append({
            "article": article,
            "views": max(views, 50_000),
            "rank": rank,
            "date": date.strftime("%Y-%m-%d"),
        })
        rank += 1

    # Spike articles
    for article, config in SPIKE_ARTICLES.items():
        if day_offset == config["spike_day"]:
            views = config["spike"]
        elif abs(day_offset - config["spike_day"]) <= 2:
            views = int(config["spike"] * 0.3)
        else:
            views = int(random.gauss(config["base"], config["base"] * 0.2))
        records.append({
            "article": article,
            "views": max(views, 1000),
            "rank": rank,
            "date": date.strftime("%Y-%m-%d"),
        })
        rank += 1

    # Regular articles (appear some days)
    for article in REGULAR_ARTICLES:
        if random.random() < 0.7:  # 70% chance of appearing
            views = int(random.gauss(80_000, 25_000) * weekend_factor)
            records.append({
                "article": article,
                "views": max(views, 10_000),
                "rank": rank,
                "date": date.strftime("%Y-%m-%d"),
            })
            rank += 1

    return records


def generate_fixture(name: str, num_days: int, start_date: datetime) -> Path:
    """Generate a complete fixture dataset."""
    output_dir = FIXTURES_DIR / name
    output_dir.mkdir(exist_ok=True)

    manifest = {
        "name": name,
        "num_days": num_days,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": (start_date + timedelta(days=num_days - 1)).strftime("%Y-%m-%d"),
        "known_properties": {
            "non_content_articles": NON_CONTENT,
            "flagged_articles": FLAGGED,
            "consistent_articles": CONSISTENT_ARTICLES,
            "spike_articles": {
                k: {
                    "spike_date": (start_date + timedelta(days=v["spike_day"])).strftime("%Y-%m-%d"),
                    "expected_peak_views": v["spike"],
                }
                for k, v in SPIKE_ARTICLES.items()
            },
            "total_unique_content_articles": len(CONSISTENT_ARTICLES) + len(SPIKE_ARTICLES) + len(REGULAR_ARTICLES),
        },
    }

    for day in range(num_days):
        date = start_date + timedelta(days=day)
        records = generate_day(date, day)
        filename = f"pageviews_{date.strftime('%Y%m%d')}.json"
        with open(output_dir / filename, "w") as f:
            json.dump(records, f, indent=2)

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return output_dir


def main():
    # Small fixture (7 days) - quick tests
    p = generate_fixture("small_7d", 7, datetime(2025, 1, 1))
    print(f"Generated small fixture: {p}")

    # Medium fixture (30 days) - standard tests
    p = generate_fixture("medium_30d", 30, datetime(2025, 1, 1))
    print(f"Generated medium fixture: {p}")

    # Large fixture (90 days) - batch tests
    p = generate_fixture("large_90d", 90, datetime(2025, 1, 1))
    print(f"Generated large fixture: {p}")


if __name__ == "__main__":
    main()
