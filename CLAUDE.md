# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Wikipedia pageviews analytics system designed for multi-cloud deployment. Collects daily Wikipedia top article statistics via the Wikimedia API, stores them in cloud object storage, and provides Jupyter notebooks for analysis. Currently implemented on AWS, with planned support for GCP, Azure, and DigitalOcean.

## Repository Structure

```
wikipedia/
├── providers/                    # Cloud provider implementations
│   ├── aws/                      # Amazon Web Services (implemented)
│   │   ├── iac/                  # Infrastructure as Code
│   │   │   ├── cloudformation/   # SAM/CloudFormation templates
│   │   │   ├── terraform/        # Terraform configs
│   │   │   ├── stack-init/       # Shell-based initialization
│   │   │   └── localstack/       # Local development
│   │   ├── app/                  # Application code (Lambda)
│   │   └── services/             # Service abstractions
│   ├── gcp/                      # Google Cloud (stub)
│   ├── azure/                    # Microsoft Azure (stub)
│   └── digitalocean/             # DigitalOcean (stub)
├── wikipedia.ipynb               # Main analysis notebook
├── find-date-gaps.py             # Utility: find missing dates in data
└── wikipedia-pageviews.csv       # Local data export
```

## Common Commands

### AWS Deployment (requires AWS SAM CLI)
```bash
cd providers/aws/iac/cloudformation

# Validate template
sam validate --lint

# Deploy stack
sam deploy --capabilities CAPABILITY_NAMED_IAM

# Or use deployment script
./deploy-inline-lambda.sh
```

### Lambda Invocation
```bash
# Invoke Lambda for a date range
python3 providers/aws/iac/cloudformation/call-pageviews-lambda-range.py \
  --function-name wikipedia-pageviews-collector-dev \
  --start-date 2025-01-01 \
  --end-date 2025-01-31

# Simple single-date invocation
cd providers/aws/app/lambda && ./lambda-invoke.sh
```

### Local Development
```bash
# Run Lambda locally
python3 providers/aws/app/lambda/wikipedia-downloader-lambda.py

# Find gaps in collected data
aws s3 ls s3://BUCKET/data/ --recursive | ./find-date-gaps.py
```

## Architecture (AWS)

```
Wikimedia Pageviews API
         │
         ▼
┌─────────────────────────────────┐
│ Lambda: wikipedia-downloader    │  ← EventBridge (daily) or manual
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ S3: wikipedia-pageviews-{env}   │  Partitioned by year/month/day
└─────────────────────────────────┘
         │
    ┌────┴────┐
    ▼         ▼
 Athena    Jupyter Notebooks
```

## Key Files

| File | Purpose |
|------|---------|
| `providers/aws/app/lambda/wikipedia-downloader-lambda.py` | Lambda handler (fetch API, store to S3) |
| `providers/aws/iac/cloudformation/wikipedia-stats-template-inline.yaml` | SAM template for all AWS resources |
| `providers/aws/iac/cloudformation/call-pageviews-lambda-range.py` | Batch invoker for backfill |
| `wikipedia.ipynb` | Main analysis notebook |

## Data Model

JSON stored per day:
```json
[{"article": "Article_Name", "views": 123456, "rank": 1, "date": "2025-01-28"}, ...]
```

**Content Filtering**: `is_content()` excludes Main_Page, Special:*, User:*, Wikipedia:*, Template:*, Category:*, and *_talk: pages.

## Provider Service Mapping

| Capability | AWS | GCP | Azure | DigitalOcean |
|------------|-----|-----|-------|--------------|
| Storage | S3 | Cloud Storage | Blob Storage | Spaces |
| Compute | Lambda | Cloud Functions | Functions | App Platform |
| Scheduling | EventBridge | Cloud Scheduler | Logic Apps | Scheduled Jobs |
| Query | Athena | BigQuery | Data Explorer | PostgreSQL |

## Python Environment

Python 3.12. Virtual environment at `.venv/`. Key deps: pandas, matplotlib, seaborn, plotly, boto3, statsmodels.
