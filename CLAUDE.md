# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Wikipedia pageviews analytics system designed for multi-cloud deployment. Collects daily Wikipedia top article statistics via the Wikimedia API, stores them in cloud object storage, and provides Jupyter notebooks for analysis. Currently implemented on AWS, with planned support for GCP, Azure, and DigitalOcean.

## Repository Structure

```
wikipedia/
├── shared/                       # Shared cloud-neutral library
│   └── wikipedia/                # Wikipedia API client & utilities
│       ├── client.py             # Fetch from Wikimedia API
│       └── storage.py            # Storage key generation
├── scripts/                      # Analysis & data processing tools
│   ├── download-pageviews.py     # Download pageviews locally
│   ├── convert-to-sqlite.py      # Convert JSON files to SQLite DB
│   ├── generate-report.py        # Generate HTML reports with visualizations
│   ├── analyze-spikes.py         # Advanced spike detection (ACF, wavelets)
│   ├── analyze-deep.py           # Deep correlation & causal inference
│   └── wavelet-tutorial.py       # Interactive wavelet transform examples
├── evals/                        # Evaluation framework
│   ├── framework.py              # Core: dimensions, rubrics, scoring
│   ├── heuristic.py              # Heuristic scorer (no LLM needed)
│   ├── judge.py                  # LLM judge with model swapping
│   ├── run_eval.py               # CLI eval runner
│   ├── test_eval.py              # Framework tests
│   ├── fixtures/                 # Test fixtures (gitignored outputs)
│   │   └── generate_fixtures.py  # Fixture data generator
│   └── results/                  # Eval results (gitignored)
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
├── notebooks/                    # Jupyter notebooks for analysis
│   └── analysis.ipynb            # Main interactive analysis notebook
├── data/                         # Local data storage (gitignored)
│   ├── pageviews_YYYYMMDD.json   # Daily JSON files
│   └── pageviews.db              # SQLite database (generated)
├── reports/                      # Generated HTML reports (gitignored)
├── wikipedia.ipynb               # Analysis notebook
├── wikipedia_analysis.ipynb      # Additional analysis notebook
└── find-date-gaps.py             # Utility: find missing dates in data
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

### Local Testing & Analysis

```bash
# Download pageviews locally (no cloud setup needed)
./scripts/download-pageviews.py 2025-01-20

# Download a date range
./scripts/download-pageviews.py 2025-01-01 2025-01-31

# Preview without saving
./scripts/download-pageviews.py --preview 2025-01-20

# Convert JSON files to SQLite for faster queries
./scripts/convert-to-sqlite.py
./scripts/convert-to-sqlite.py --force  # Overwrite existing DB

# Generate HTML reports with visualizations
./scripts/generate-report.py             # Last 30 days
./scripts/generate-report.py --days 90   # Last 90 days
./scripts/generate-report.py --all       # All available data

# Advanced spike analysis (ACF, wavelets, cross-correlation)
./scripts/analyze-spikes.py              # Last year
./scripts/analyze-spikes.py --all        # All data

# Deep correlation analysis with causal inference
./scripts/analyze-deep.py                # Last 365 days
./scripts/analyze-deep.py --days 730     # Last 2 years
./scripts/analyze-deep.py --all          # All data

# Wavelet transform tutorial
./scripts/wavelet-tutorial.py

# Find gaps in collected data
aws s3 ls s3://BUCKET/data/ --recursive | ./find-date-gaps.py
```

### Evaluation Framework

```bash
# Run heuristic eval (no API key needed)
python -m evals.run_eval --judge heuristic

# Run heuristic eval on all fixture sizes (7d, 30d, 90d)
python -m evals.run_eval --judge heuristic --batch all

# Investigate a specific dimension (e.g. synthesis)
python -m evals.run_eval --judge heuristic --dimension synthesis

# Run LLM judge eval (requires ANTHROPIC_API_KEY)
python -m evals.run_eval --judge llm --model haiku

# Compare two judge models (model swapping)
python -m evals.run_eval --judge llm --model haiku --compare sonnet

# Run both heuristic and LLM judge
python -m evals.run_eval --judge both --model haiku

# Save results to evals/results/
python -m evals.run_eval --judge heuristic --batch all --save

# Run framework tests
python -m evals.test_eval
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
    ┌────┴─────────┐
    ▼              ▼
 Athena    Local JSON Files (data/)
                   │
                   ▼
           ┌───────────────┐
           │ SQLite DB     │  convert-to-sqlite.py
           └───────────────┘
                   │
         ┌─────────┼─────────┐
         ▼         ▼         ▼
    Notebooks  Reports  Analysis Scripts
```

## Key Files

| File                                                                    | Purpose                                               |
| ----------------------------------------------------------------------- | ----------------------------------------------------- |
| `scripts/download-pageviews.py`                                         | Download pageviews to local data/ directory           |
| `scripts/convert-to-sqlite.py`                                          | Convert JSON files to SQLite database                 |
| `scripts/generate-report.py`                                            | Generate HTML reports with seaborn visualizations     |
| `scripts/analyze-spikes.py`                                             | Advanced spike detection with signal processing       |
| `scripts/analyze-deep.py`                                               | Deep correlation analysis & causal inference          |
| `scripts/wavelet-tutorial.py`                                           | Interactive wavelet transform examples                |
| `scripts/review-flagged.py`                                             | Manual review tool for flagged articles               |
| `shared/wikipedia/client.py`                                            | Cloud-neutral Wikipedia API client                    |
| `shared/wikipedia/storage.py`                                           | Cloud-neutral storage key generation                  |
| `shared/wikipedia/filters.py`                                           | Content filtering logic (is_content, get_hide_reason) |
| `providers/aws/app/lambda/wikipedia-downloader-lambda.py`               | AWS Lambda handler                                    |
| `providers/aws/iac/cloudformation/wikipedia-stats-template-inline.yaml` | SAM template for AWS resources                        |
| `notebooks/analysis.ipynb`                                              | Main interactive analysis notebook                    |
| `evals/framework.py`                                                    | Eval dimensions, rubrics, scoring data structures     |
| `evals/heuristic.py`                                                    | Heuristic scorer (structural report analysis)         |
| `evals/judge.py`                                                        | LLM judge with model swapping (haiku/sonnet/opus)     |
| `evals/run_eval.py`                                                     | CLI eval runner (heuristic, LLM, batch, compare)      |
| `evals/test_eval.py`                                                    | Framework tests (7 tests, includes mocked LLM judge)  |
| `evals/fixtures/generate_fixtures.py`                                   | Generate test data with known spikes & properties     |
| `find-date-gaps.py`                                                     | Find missing dates in collected data                  |

## Data Model

### JSON Format

JSON files stored per day in `data/pageviews_YYYYMMDD.json`:

```json
[{"article": "Article_Name", "views": 123456, "rank": 1, "date": "2025-01-28"}, ...]
```

### SQLite Database

For optimized queries, JSON files can be converted to SQLite at `data/pageviews.db`:

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

- `idx_article_date` (article, date) - Article trends over time
- `idx_date` (date) - Top articles on specific dates
- `idx_date_views` (date, views DESC) - Ranking queries
- `idx_article` (article) - Article lookups
- `idx_hide` (hide) - Filter hidden records

**Content Filtering**: The database includes a `hide` column to filter non-content and flagged articles. See Content Filtering section below for details.

## Generated Outputs

### Reports Directory

HTML reports are generated in `reports/` (gitignored):

- `latest.html` - Most recent standard report
- `spike_analysis.html` - Advanced spike detection report
- `deep_analysis.html` - Correlation and causal inference report
- `wavelet_tutorial.html` - Interactive wavelet examples

Reports include embedded visualizations and are self-contained (can be shared or archived).

## Content Filtering

The database includes a `hide` column to filter out non-content pages and articles flagged for review:

### Filtering Logic

**Non-content pages (automatically hidden):**

- `Main_Page` - Wikipedia main page
- `Special:*` - Special namespace (search, admin pages, etc.)
- `User:*`, `Wikipedia:*`, `Template:*`, `Category:*`, `Portal:*`, etc.
- Pages with `_talk:` in the name (discussion pages)

**Flagged for review:**

- Articles with keywords that may indicate adult content
- Conservative flagging - false positives can be manually unhidden
- Keywords include: pornography, xxx, sexual_intercourse, erotic, hentai, etc.

### Using Filters in Queries

**Standard queries should include `WHERE hide=0`:**

```sql
-- Top articles on a specific date
SELECT article, views
FROM pageviews
WHERE hide=0 AND date='2025-01-01'
ORDER BY views DESC
LIMIT 10;

-- Article trends over time
SELECT date, views
FROM pageviews
WHERE hide=0 AND article='Python_(programming_language)'
ORDER BY date;
```

### Manual Review Tools

**Review flagged articles:**

```bash
./scripts/review-flagged.py list              # Show all flagged articles
./scripts/review-flagged.py unhide "Article"  # Mark article as legitimate
./scripts/review-flagged.py hide "Article"    # Manually hide an article
./scripts/review-flagged.py hide "Article" "reason"  # Hide with custom reason
```

**Hide reasons:**

- `main_page` - Wikipedia main page
- `special_page` - Special: namespace (search, admin pages)
- `talk_page` - Talk/discussion pages
- `non_content_page` - Other non-content namespaces
- `flagged_for_review` - Needs manual review (potential adult content)
- `manual_block` - Manually hidden by admin

### Shared Filtering Module

All filtering logic is centralized in `shared/wikipedia/filters.py`:

- `is_content(article)` - Returns False for non-content pages
- `should_flag_for_review(article)` - Returns True for articles needing review
- `get_hide_reason(article)` - Returns hide reason string or None

Import in Python code:

```python
from shared.wikipedia.filters import is_content, get_hide_reason
```

## Evaluation Framework

The eval framework assesses analytics pipeline output quality across five weighted dimensions using either heuristic (structural) or LLM-based (Claude) judging.

### Scoring Dimensions

| Dimension       | Weight | Description                                          |
| --------------- | ------ | ---------------------------------------------------- |
| Accuracy        | 0.25   | Statistical correctness of computed metrics          |
| Completeness    | 0.20   | Coverage of all expected report sections and data    |
| Synthesis       | 0.25   | How well the report connects data into insights      |
| Filtering       | 0.15   | Proper exclusion of non-content pages                |
| Visualization   | 0.15   | Quality and appropriateness of charts                |

### Judge Types

- **Heuristic** (`--judge heuristic`): Structural analysis of report HTML. No API key needed. Checks for presence of sections, cross-references, causal language, chart rendering, and filtering correctness.
- **LLM** (`--judge llm`): Uses Claude as an evaluator. Requires `ANTHROPIC_API_KEY`. Supports model swapping via `--model` (haiku, sonnet, opus).

### Test Fixtures

Generated by `evals/fixtures/generate_fixtures.py` with known properties:

- **small_7d**: 7 days, quick validation
- **medium_30d**: 30 days, standard evaluation (default)
- **large_90d**: 90 days, stress testing

Fixtures include planted spikes (Super Bowl, Solar eclipse, Academy Awards), consistent articles (YouTube, Facebook, etc.), non-content pages, and flagged articles for filtering validation.

### Current Baseline Scores (Heuristic)

| Dimension     | Score | Notes                                    |
| ------------- | ----- | ---------------------------------------- |
| Accuracy      | 0.900 | Partial on article count verification    |
| Completeness  | 1.000 | All sections present                     |
| Synthesis     | 1.000 | Data-driven narrative with causal language |
| Filtering     | 0.800 | Non-content pages filtered               |
| Visualization | 1.000 | Charts render, alt text, responsive      |
| **Overall**   | **0.945** |                                      |

### Synthesis Score Analysis

The synthesis dimension scores 1.00 after adding data-driven narrative paragraphs. Criteria breakdown:

- `narrative_present`: 1.00 - 5+ explanatory paragraphs with causal language
- `contextual_language`: 1.00 - Uses pattern/trend/spike terminology
- `cross_references`: 1.00 - Weekend traffic patterns referenced
- `causal_explanations`: 1.00 - Uses "driven by", "this suggests", "likely", "because", "due to"
- `summary_present`: 1.00 - Overview section exists

The `generate_narrative()` function in `generate-report.py` produces data-driven paragraphs covering peak traffic context, day-of-week patterns, and spike explanations. All text is deterministic (no LLM calls) and derived from already-computed data.

**Future improvement paths**:
- Consider optional LLM-generated narrative summaries (`--enrich` flag)
- Add more cross-dimensional synthesis between spike and consistency data

### Model Swapping

The LLM judge supports three Claude models via `--model`:

| Key    | Model ID                     | Use Case              |
| ------ | ---------------------------- | --------------------- |
| haiku  | claude-3-5-haiku-20241022    | Fast, cost-effective  |
| sonnet | claude-sonnet-4-20250514     | Balanced              |
| opus   | claude-opus-4-0-20250514     | Most capable          |

Compare models: `python -m evals.run_eval --judge llm --model haiku --compare sonnet`

## Provider Service Mapping

| Capability | AWS         | GCP             | Azure         | DigitalOcean   |
| ---------- | ----------- | --------------- | ------------- | -------------- |
| Storage    | S3          | Cloud Storage   | Blob Storage  | Spaces         |
| Compute    | Lambda      | Cloud Functions | Functions     | App Platform   |
| Scheduling | EventBridge | Cloud Scheduler | Logic Apps    | Scheduled Jobs |
| Query      | Athena      | BigQuery        | Data Explorer | PostgreSQL     |

## Python Environment

Python 3.12. Virtual environment at `.venv/`.

**Key dependencies:**

- **Data analysis**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Signal processing**: scipy, PyWavelets (pywt)
- **Statistics**: statsmodels
- **Cloud services**: boto3 (AWS)
- **LLM evaluation**: anthropic (optional, for LLM judge)
- **Notebooks**: jupyter

## Analysis Capabilities

The project includes advanced statistical and signal processing analysis:

**Report Generation:**

- HTML reports with seaborn visualizations
- Top articles, traffic patterns, spike detection
- Day-of-week trends and consistency analysis

**Spike Detection & Analysis:**

- Autocorrelation Function (ACF) for periodicity detection
- Wavelet transforms for multi-scale time-frequency analysis
- Spike shape classification (breaking news, sustained interest, anticipation)
- Cross-correlation to find articles that spike together

**Deep Analysis:**

- Correlation analysis between article trends
- Causal inference techniques
- Time-series anomaly detection
- Multi-scale pattern recognition

**Performance Optimization:**

- SQLite database for fast queries on large datasets
- Optimized indexes for common query patterns
- Efficient date range filtering

## Recommended Workflow

1. **Data Collection:**

   ```bash
   # Download data for date range
   ./scripts/download-pageviews.py 2025-01-01 2025-01-31
   ```

2. **Database Conversion** (optional but recommended for large datasets):

   ```bash
   # Convert JSON to SQLite for faster analysis
   ./scripts/convert-to-sqlite.py
   ```

3. **Generate Reports:**

   ```bash
   # Standard report
   ./scripts/generate-report.py --days 30

   # Advanced spike analysis
   ./scripts/analyze-spikes.py

   # Deep correlation analysis
   ./scripts/analyze-deep.py
   ```

4. **Evaluate Pipeline Quality** (optional):

   ```bash
   # Run heuristic evaluation across all fixture sizes
   python -m evals.run_eval --judge heuristic --batch all --save

   # Run tests
   python -m evals.test_eval
   ```

5. **Interactive Analysis:**
   ```bash
   # Open Jupyter for custom analysis
   jupyter notebook notebooks/analysis.ipynb
   ```

## Tips & Best Practices

- **Use SQLite for large datasets**: For datasets > 100 days, convert to SQLite for significantly faster queries
- **Run analysis scripts in order**: Generate standard reports first to understand the data, then run spike/deep analysis
- **Date ranges**: Start with smaller date ranges (30-90 days) to validate analysis before running on full dataset
- **Report outputs**: All reports are self-contained HTML files with embedded visualizations
- **Script output**: Analysis scripts save results to `reports/` directory with timestamped filenames
- **Missing data**: Use `find-date-gaps.py` to identify gaps in your dataset before analysis

## Development Guidelines

> [!IMPORTANT]
> **Backward Compatibility Rule**: Never break existing scripts except in extreme cases that require explicit approval. All changes must be incremental and maintain backward compatibility.

**Key Principles**:

- New features should be additive, not destructive
- Existing scripts must continue to work without modification
- New dependencies should be optional when possible
- Use feature flags or separate scripts for experimental features
- Document breaking changes clearly if absolutely necessary

**Examples**:

- ✅ **Good**: Add optional `--enrich` flag to existing script
- ✅ **Good**: Create new `enrich-metadata.py` script for new functionality
- ❌ **Bad**: Modify existing script to require new dependencies
- ❌ **Bad**: Change output format of existing reports without flag

**Pipeline Approach**:

- Prefer separate scripts that can be chained together
- Each script should have a single responsibility
- Data flows through files (JSON, SQLite) between pipeline stages
- Allow users to run individual pipeline stages independently
