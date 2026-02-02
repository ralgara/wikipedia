# Shared Library

Cloud-neutral code shared across all providers.

## wikipedia/

Wikipedia API client and utilities.

| Module | Purpose |
|--------|---------|
| `client.py` | Fetch pageviews from Wikimedia API |
| `storage.py` | Generate partitioned storage keys |

### Usage

```python
from wikipedia import download_pageviews, generate_storage_key

# Fetch data (cloud-neutral)
data = download_pageviews(datetime(2025, 1, 15))

# Generate storage key (works with S3, GCS, Azure Blob, etc.)
key = generate_storage_key(datetime(2025, 1, 15))
# -> "wikipedia/pageviews/year=2025/month=01/day=15/pageviews_20250115.json"
```

## Design Principles

1. **No cloud SDK imports** - only standard library + urllib3
2. **Hive-style partitioning** - compatible with Athena, BigQuery, Spark
3. **Simple functions** - no complex class hierarchies
