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
│       └── storage.py            # Storage key generation
├── scripts/                      # Local testing & CLI tools
│   ├── download-pageviews.py     # Download pageviews locally
│   └── generate-report.py        # Generate HTML reports
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

## Data Format

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

## Requirements

- Python 3.12+
- Virtual environment: `python3 -m venv .venv && source .venv/bin/activate`
- For cloud deployment: AWS SAM CLI, AWS credentials
