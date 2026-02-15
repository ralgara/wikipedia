"""Wikipedia API client - cloud neutral.

Fetches pageview data from the Wikimedia REST API.
No cloud provider dependencies.
"""

import json
import logging
from datetime import datetime, timedelta

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    import urllib.request
    _HAS_REQUESTS = False

logger = logging.getLogger(__name__)

WIKIMEDIA_API_BASE = "https://wikimedia.org/api/rest_v1/metrics/pageviews"


def download_pageviews(date: datetime, project: str = "en.wikipedia", access: str = "all-access", user_agent: str = "WikipediaPageviewsBot/1.0") -> list[dict]:
    """Download top pageviews for a single date.

    Args:
        date: Date to fetch data for
        project: Wikipedia project (e.g., en.wikipedia, de.wikipedia)
        access: Access type (all-access, desktop, mobile-app, mobile-web)
        user_agent: User-Agent header (required by Wikimedia API)

    Returns:
        List of article dicts with keys: article, views, rank, date

    Raises:
        Exception: If API request fails
    """
    url = f"{WIKIMEDIA_API_BASE}/top/{project}/{access}/{date.year}/{date.strftime('%m')}/{date.strftime('%d')}"
    headers = {"User-Agent": user_agent}

    logger.info(f"Fetching pageviews from {url}")

    if _HAS_REQUESTS:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code != 200:
            raise Exception(f"API request failed with status {response.status_code}: {response.text}")
        data = response.json()
    else:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            if response.status != 200:
                raise Exception(f"API request failed with status {response.status}")
            data = json.loads(response.read().decode("utf-8"))

    item = data["items"][0]
    date_str = f"{item['year']}-{item['month']}-{item['day']}"

    return [{**article, "date": date_str} for article in item["articles"]]


def download_pageviews_range(start_date: datetime, end_date: datetime, **kwargs) -> list[list[dict]]:
    """Download pageviews for a date range.

    Args:
        start_date: Start date (inclusive)
        end_date: End date (inclusive)
        **kwargs: Passed to download_pageviews (project, access)

    Returns:
        List of daily results, each a list of article dicts
    """
    results = []
    current = start_date

    while current <= end_date:
        try:
            data = download_pageviews(current, **kwargs)
            results.append(data)
        except Exception as e:
            logger.error(f"Failed to fetch {current}: {e}")
            results.append([])  # Empty list for failed dates
        current += timedelta(days=1)

    return results
