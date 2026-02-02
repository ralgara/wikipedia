"""Storage utilities - cloud neutral.

Key/path generation for partitioned storage.
Works with S3, GCS, Azure Blob, DO Spaces, or local filesystem.
"""

from datetime import datetime


def generate_storage_key(date: datetime, prefix: str = "wikipedia/pageviews") -> str:
    """Generate a partitioned storage key for the given date.

    Uses Hive-style partitioning (year=YYYY/month=MM/day=DD) which is
    compatible with Athena, BigQuery, Spark, etc.

    Args:
        date: Date for the data
        prefix: Key prefix (default: wikipedia/pageviews)

    Returns:
        Storage key like: wikipedia/pageviews/year=2025/month=01/day=15/pageviews_20250115.json
    """
    return (
        f"{prefix}/"
        f"year={date.year}/"
        f"month={date.strftime('%m')}/"
        f"day={date.strftime('%d')}/"
        f"pageviews_{date.strftime('%Y%m%d')}.json"
    )


def parse_storage_key(key: str) -> datetime | None:
    """Extract date from a storage key.

    Args:
        key: Storage key with Hive-style partitioning

    Returns:
        datetime if parseable, None otherwise
    """
    import re
    match = re.search(r"year=(\d{4})/month=(\d{2})/day=(\d{2})", key)
    if match:
        return datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    return None
