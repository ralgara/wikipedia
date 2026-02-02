# Wikipedia Pageviews Analytics

Collect and analyze trends in Wikipedia pageview statistics. Multi-cloud architecture with current implementation on AWS.

## Local Testing

Download Wikipedia pageviews data locally without any cloud setup:

```bash
# Activate virtual environment
source .venv/bin/activate

# Download pageviews for a specific date
python3 providers/aws/app/lambda/wikipedia-downloader-lambda.py
```

Output is saved to `data/`:

```bash
# View the downloaded file
ls data/
# pageviews_20250128.json

# Inspect top articles
cat data/pageviews_20250128.json | python3 -m json.tool | head -50
```

To change the date, edit the `__main__` block in the Lambda file or run:

```bash
python3 -c "
from datetime import datetime
import sys, os
sys.path.insert(0, 'shared')
from wikipedia import download_pageviews
import json

data = download_pageviews(datetime(2025, 1, 15))
print(json.dumps(data[:5], indent=2))
"
```

## Project Structure

```
wikipedia/
├── shared/                       # Cloud-neutral library
│   └── wikipedia/
│       ├── client.py             # Wikimedia API client
│       └── storage.py            # Storage key generation
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
├── data/                         # Local test output (gitignored)
└── wikipedia.ipynb               # Analysis notebook
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
