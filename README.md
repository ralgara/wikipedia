# Wikipedia Pageviews Analytics

Collect and analyze trends in Wikipedia pageview statistics. Multi-cloud architecture with current implementation on AWS.

## Local Testing

Download Wikipedia pageviews data locally without any cloud setup:

```bash
# Activate virtual environment
source .venv/bin/activate

# Download yesterday's pageviews
./scripts/download-pageviews.py

# Download a specific date
./scripts/download-pageviews.py 2025-01-20

# Download a date range
./scripts/download-pageviews.py 2025-01-01 2025-01-07

# Preview top 5 articles without saving
./scripts/download-pageviews.py --preview 2025-01-20
```

Output is saved to `data/`:

```bash
# View downloaded files
ls data/
# pageviews_20250120.json

# Inspect top articles
cat data/pageviews_20250120.json | python3 -m json.tool | head -50
```

## Reports

Generate professional HTML reports with seaborn visualizations:

```bash
source .venv/bin/activate

# Generate report for last 30 days (default)
./scripts/generate-report.py

# Last 90 days
./scripts/generate-report.py --days 90

# All available data
./scripts/generate-report.py --all
```

Reports are saved to `reports/` and include:
- Overview metrics and daily traffic trends
- Top articles by total views
- Traffic patterns by day of week
- Spike detection and analysis
- Consistency analysis (articles always trending)

### Advanced Spike Analysis

Deep analysis of traffic spikes using signal processing techniques:

```bash
# Analyze last year (default)
./scripts/analyze-spikes.py

# Analyze all available data
./scripts/analyze-spikes.py --all
```

The spike analysis report includes:
- **Periodicity detection** (ACF) - Find yearly anniversaries, weekly patterns
- **Spike shape classification** - Breaking news vs sustained interest vs anticipation
- **Cross-correlation** - Discover articles that spike together
- **Wavelet analysis** - Multi-scale time-frequency patterns

### Deep Correlation Analysis

Advanced correlation detection with causal inference:

```bash
# Analyze last year (default)
./scripts/analyze-deep.py

# Analyze with enriched metadata (requires enrichment step)
./scripts/analyze-deep.py --use-enriched
```

### Semantic Enrichment

Enrich article metadata using Wikidata and DBpedia:

```bash
# Enrich top 1000 articles
./scripts/enrich-metadata.py --top-n 1000

# Resume enrichment
./scripts/enrich-metadata.py --resume
```

## Analysis Notebook

For interactive exploration, run the Jupyter notebook:

```bash
source .venv/bin/activate
jupyter notebook notebooks/analysis.ipynb
```

The notebook includes:
- Summary statistics and top articles
- Daily trends and spike detection
- Day-of-week patterns
- Consistency analysis (persistent articles vs one-hit wonders)

## Project Structure

```
wikipedia/
├── shared/                       # Cloud-neutral library
│   └── wikipedia/
│       ├── client.py             # Wikimedia API client
│       ├── storage.py            # Storage key generation
│       └── filters.py            # Content filtering logic
├── scripts/                      # Local testing & CLI tools
│   ├── download-pageviews.py     # Download pageviews locally
│   ├── convert-to-sqlite.py      # Convert JSON to SQLite
│   ├── generate-report.py        # Generate HTML reports
│   ├── analyze-spikes.py         # Advanced spike analysis
│   ├── analyze-deep.py           # Deep correlation analysis
│   ├── enrich-metadata.py        # Wikidata/DBpedia enrichment
│   ├── find-date-gaps.py         # Identify missing data
│   └── review-flagged.py         # Review flagged articles
├── providers/                    # Cloud provider implementations
│   ├── aws/                      # Amazon Web Services (implemented)
│   │   ├── iac/                  # Infrastructure as Code
│   │   │   ├── cloudformation/   # SAM templates
│   │   │   ├── terraform/        # Terraform configs
│   │   │   └── localstack/       # Local AWS simulation
│   │   └── app/                  # Lambda handlers
│   ├── gcp/                      # Google Cloud (planned)
│   ├── azure/                    # Microsoft Azure (planned)
│   └── digitalocean/             # DigitalOcean (planned)
├── notebooks/                    # Jupyter notebooks
│   └── analysis.ipynb            # Main analysis notebook
├── data/                         # Local test output (gitignored)
├── reports/                      # Generated HTML reports (gitignored)
└── wikipedia.ipynb               # (legacy notebook)
```

## Cloud Deployment (AWS)

Requires [AWS SAM CLI](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html).

```bash
cd providers/aws/iac/cloudformation

# Validate template
sam validate --lint

# Deploy
sam deploy --capabilities CAPABILITY_NAMED_IAM
```

See [providers/aws/README.md](providers/aws/README.md) for details.

## Data Formats

### JSON Files

Each daily download contains the top 1000 Wikipedia articles:

```json
[
  {"article": "Main_Page", "views": 4859522, "rank": 1, "date": "2025-01-28"},
  {"article": "DeepSeek", "views": 860907, "rank": 3, "date": "2025-01-28"},
  ...
]
```

Storage uses Hive-style partitioning for compatibility with Athena, BigQuery, etc:
```
wikipedia/pageviews/year=2025/month=01/day=28/pageviews_20250128.json
```

### SQLite Database

Convert JSON files to SQLite for faster queries:

```bash
# Create database from JSON files
./scripts/convert-to-sqlite.py

# Overwrite existing database
./scripts/convert-to-sqlite.py --force
```

The database is created at `data/pageviews.db` with optimized indexes and content filtering:

**Schema:**
```sql
CREATE TABLE pageviews (
    article TEXT NOT NULL,
    views INTEGER NOT NULL,
    rank INTEGER NOT NULL,
    date TEXT NOT NULL,
    hide INTEGER NOT NULL DEFAULT 0,
    hide_reason TEXT
);
```

**Indexes:**

| Index | Columns | Optimizes |
|-------|---------|-----------|
| `idx_article_date` | (article, date) | Article X over time queries |
| `idx_date` | (date) | Top articles on date Y |
| `idx_date_views` | (date, views DESC) | Ranking queries |
| `idx_article` | (article) | Article lookups |
| `idx_hide` | (hide) | Filter hidden records |

### Content Filtering

The database includes a `hide` column to filter non-content pages:

**Automatically hidden:**
- `Main_Page` - Wikipedia main page
- `Special:*` - Special pages (search, admin tools)
- `User:*`, `Wikipedia:*`, `Template:*`, `Category:*`, etc.
- Pages with `_talk:` (discussion pages)

**Flagged for review:**
- Articles with keywords that may indicate adult content
- Can be manually reviewed and unhidden if legitimate

**Review flagged articles:**
```bash
./scripts/review-flagged.py list              # Show flagged articles
./scripts/review-flagged.py unhide "Article"  # Mark as legitimate
./scripts/review-flagged.py hide "Article"    # Manually hide
```

**Example queries:**

```sql
-- Article views over time (content only)
SELECT date, views FROM pageviews
WHERE hide=0 AND article='Python_(programming_language)'
ORDER BY date;

-- Top 10 articles on a specific date (content only)
SELECT article, views FROM pageviews
WHERE hide=0 AND date='2025-01-01'
ORDER BY views DESC LIMIT 10;

-- Compare two articles
SELECT date, article, views FROM pageviews
WHERE hide=0 AND article IN ('ChatGPT', 'Google')
ORDER BY date, article;

-- Total views per article (all time, content only)
SELECT article, SUM(views) as total
FROM pageviews
WHERE hide=0
GROUP BY article
ORDER BY total DESC LIMIT 20;
```

**Note:** Add `WHERE hide=0` to filter out Main_Page, Special:* pages, and flagged content.

## Requirements

- Python 3.12+
- Virtual environment: `python3 -m venv .venv && source .venv/bin/activate`
- For cloud deployment: AWS SAM CLI, AWS credentials
