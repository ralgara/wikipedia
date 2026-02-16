# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Wikipedia pageviews analytics system designed for multi-cloud deployment. Collects daily Wikipedia top article statistics via the Wikimedia API, stores them in cloud object storage, and provides Jupyter notebooks for analysis. Currently implemented on AWS, with planned support for GCP, Azure, and DigitalOcean.

The project also serves as a vehicle for learning **multi-agent orchestration**. The `agents/` directory contains a prototype system that decomposes natural language questions about Wikipedia article performance into sub-tasks executed by specialized agents (planner, retrieval, graph/ontology, synthesis).

## Repository Structure

```
wikipedia/
├── agents/                       # Multi-agent orchestration (prototype)
│   ├── orchestrator.sh           # Shell-based orchestrator (Tier 1)
│   ├── mock_data/                # Mock data for agent development
│   │   ├── pageviews.json        # Mock pageview database
│   │   └── ontology.json         # Mock knowledge graph
│   ├── CONTEXT.md                # Agent architecture & design rationale
│   └── evals/                    # Evaluation framework
│       ├── run-evals.py          # CLI: run evals, compare runs
│       ├── generate-cases.py     # Auto-generate cases from SQLite DB
│       ├── cases/                # Eval case definitions (YAML)
│       ├── judges/               # Deterministic + LLM-as-judge
│       ├── lib/                  # Runner, scoring, schemas
│       └── results/              # Run outputs (gitignored)
├── shared/                       # Shared cloud-neutral library
│   └── wikipedia/                # Wikipedia API client & utilities
│       ├── client.py             # Fetch from Wikimedia API
│       ├── storage.py            # Storage key generation
│       ├── filters.py            # Content filtering (is_content, get_hide_reason)
│       ├── ontology.py           # Wikidata/DBpedia client (enrichment)
│       └── graph_analysis.py     # NetworkX graph analysis (enrichment)
├── scripts/                      # Analysis & data processing tools
│   ├── download-pageviews.py     # Download pageviews locally
│   ├── convert-to-sqlite.py      # Convert JSON files to SQLite DB
│   ├── generate-report.py        # Generate HTML reports with visualizations
│   ├── analyze-spikes.py         # Advanced spike detection (ACF, wavelets)
│   ├── analyze-deep.py           # Deep correlation & causal inference
│   ├── enrich-metadata.py        # Wikidata/DBpedia metadata enrichment
│   ├── review-flagged.py         # Manual review tool for flagged articles
│   └── wavelet-tutorial.py       # Interactive wavelet transform examples
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
│   ├── pageviews.db              # SQLite database (generated)
│   └── enriched_metadata/        # Enriched metadata JSON (generated)
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

# Enrich articles with Wikidata/DBpedia metadata (optional)
./scripts/enrich-metadata.py                   # Enrich top articles
./scripts/enrich-metadata.py --top 50          # Top 50 articles
./scripts/enrich-metadata.py --dry-run         # Preview without saving

# Deep analysis with enrichment data
./scripts/analyze-deep.py --use-enriched       # Graph + community analysis

# Wavelet transform tutorial
./scripts/wavelet-tutorial.py

# Find gaps in collected data
aws s3 ls s3://BUCKET/data/ --recursive | ./find-date-gaps.py
```

### Multi-Agent Orchestration

```bash
# Run the agent orchestrator (requires Claude Code CLI)
./agents/orchestrator.sh "How much more hits did Greenland vs Sweden get in wikipedia last year?"

# The orchestrator runs 4 agents via claude -p:
#   1. Planner: decomposes question into sub-tasks
#   2. Retrieval: looks up pageview data (mock)
#   3. Graph: analyzes ontology relationships (mock)
#   4. Synthesis: merges results into markdown report
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
| `scripts/enrich-metadata.py`                                            | Wikidata/DBpedia metadata enrichment pipeline         |
| `scripts/wavelet-tutorial.py`                                           | Interactive wavelet transform examples                |
| `scripts/review-flagged.py`                                             | Manual review tool for flagged articles               |
| `shared/wikipedia/client.py`                                            | Cloud-neutral Wikipedia API client                    |
| `shared/wikipedia/storage.py`                                           | Cloud-neutral storage key generation                  |
| `shared/wikipedia/filters.py`                                           | Content filtering logic (is_content, get_hide_reason) |
| `shared/wikipedia/ontology.py`                                          | Wikidata + DBpedia clients with caching               |
| `shared/wikipedia/graph_analysis.py`                                    | NetworkX graph analysis, community detection          |
| `agents/orchestrator.sh`                                                | Multi-agent orchestrator (Tier 1 prototype)           |
| `agents/CONTEXT.md`                                                     | Agent architecture design rationale                   |
| `agents/evals/run-evals.py`                                             | Eval framework CLI entry point                        |
| `agents/evals/generate-cases.py`                                        | Auto-generate eval cases from SQLite DB               |
| `providers/aws/app/lambda/wikipedia-downloader-lambda.py`               | AWS Lambda handler                                    |
| `providers/aws/iac/cloudformation/wikipedia-stats-template-inline.yaml` | SAM template for AWS resources                        |
| `notebooks/analysis.ipynb`                                              | Main interactive analysis notebook                    |
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
- **Notebooks**: jupyter
- **Enrichment (optional)**: networkx, SPARQLWrapper, requests-cache, python-louvain, rich

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

**Enrichment Pipeline (optional):**

- Wikidata entity resolution and metadata extraction
- DBpedia SPARQL queries for categories and relationships
- Semantic similarity between articles (Jaccard)
- NetworkX graph construction with community detection (Louvain)
- Centrality metrics (PageRank, betweenness, closeness)
- GraphML export for external tools (Gephi)

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

3. **Enrichment** (optional, requires enrichment dependencies):

   ```bash
   # Enrich top articles with Wikidata/DBpedia metadata
   ./scripts/enrich-metadata.py --top 50
   ```

4. **Generate Reports:**

   ```bash
   # Standard report
   ./scripts/generate-report.py --days 30

   # Advanced spike analysis
   ./scripts/analyze-spikes.py

   # Deep correlation analysis
   ./scripts/analyze-deep.py

   # Deep analysis with graph/community detection
   ./scripts/analyze-deep.py --use-enriched
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

## Agent Orchestration

The `agents/` directory contains a multi-agent system prototype for answering quantitative, ontology-aware questions about Wikipedia article performance. This is primarily a **didactic** project for learning multi-agent patterns.

**Architecture:** Orchestrator-Worker with parallel fan-out.

```
User Question
     │
  Planner Agent (decompose + normalize)
     │
     ├── Retrieval Agent (pageviews DB)
     ├── Graph Agent (ontology/taxonomy)
     │
  Synthesis Agent (merge → markdown report)
     │
  Final Report
```

**Tiered development path:**

| Tier | Orchestrator | Data Sources | Agents |
|------|-------------|--------------|--------|
| 1 (current) | `orchestrator.sh` | Mock JSON files | `claude -p` with inline data |
| 2 | Bash script (same) | MCP servers wrapping real DB | `claude -p` + `--mcp-config` |
| 3 | Python/TS with Agent SDK | MCP servers | SDK agents with typed tools |
| 4 | Skills + A2A protocol | Distributed | Reusable packaged Skills |

**Key design decisions:**
- LLM planner for prototype; target deterministic planner for structured parts
- Concurrency via `claude -p` background processes (`&` + `wait`)
- Intermediate artifacts saved to temp dir for inspection
- See `agents/CONTEXT.md` for full design rationale

## Agent Evaluation Framework

The `agents/evals/` directory provides a regression gate for evolving the multi-agent system. It enables swapping subtask agents to lighter/fine-tuned models while ensuring analytical quality is maintained.

### Quick Start

```bash
# Generate 50+ eval cases from SQLite DB (one-time, or after DB changes)
./agents/evals/generate-cases.py --db data/pageviews.db

# Run deterministic-only eval (fast, free, no LLM calls)
./agents/evals/run-evals.py --judge deterministic --case greenland_vs_sweden

# Run full eval with LLM judge (uses claude -p)
./agents/evals/run-evals.py --case greenland_vs_sweden

# Run a batch by question type
./agents/evals/run-evals.py --type comparison --limit 5

# Test model swap (e.g. Haiku for retrieval agent)
./agents/evals/run-evals.py --model retrieval:haiku --case greenland_vs_sweden

# Compare two runs for regression detection
./agents/evals/run-evals.py --compare results/run_A.json results/run_B.json

# List all available cases
./agents/evals/run-evals.py --list
```

### Architecture

**Two judge types:**
- **Deterministic** (`judges/deterministic.py`): JSON validity, required fields, numeric accuracy, keyword presence, table detection. Fast, free, stable baseline.
- **LLM-as-judge** (`judges/llm_judge.py`): Uses `claude -p --model claude-opus-4-6` with per-agent rubrics. Always uses flagship model for consistent evaluation.

**Five question types** (auto-generated from real pageviews DB):
- `comparison` (~15): Article A vs Article B in year Y
- `trend` (~10): How has article X changed since year Y?
- `spike` (~10): What caused the traffic spike in article X?
- `ranking` (~10): Top N articles in month M
- `edge_case` (5): Ambiguous, missing data, future dates

**Case format** (YAML): Each case defines question, inline mock data snapshot, ground-truth answers, and per-agent expected outputs with deterministic check targets + LLM judge guidance notes.

### Key Files

| File | Purpose |
|------|---------|
| `agents/evals/run-evals.py` | CLI entry point |
| `agents/evals/generate-cases.py` | Auto-generate cases from SQLite DB |
| `agents/evals/cases/greenland_vs_sweden.yaml` | Canonical hand-crafted case |
| `agents/evals/judges/deterministic.py` | Rule-based checks |
| `agents/evals/judges/llm_judge.py` | LLM-as-judge via claude -p |
| `agents/evals/judges/rubrics/*.txt` | Per-agent rubric templates |
| `agents/evals/lib/runner.py` | Agent invocation (claude -p subprocess) |
| `agents/evals/lib/scoring.py` | Score aggregation, terminal tables, comparison |
| `agents/evals/lib/schemas.py` | EvalCase, AgentResult, JudgmentResult dataclasses |

### Dependencies

Optional: `pyyaml>=6.0` (install via `uv pip install pyyaml`). Defined in `pyproject.toml` under `[project.optional-dependencies] evals`.

### Current Status

Smoke tested on canonical case: planner 1.0, retrieval 1.0, graph 1.0, synthesis 0.67, overall 0.92 PASS.

**Next steps:**
- Run full LLM judge evaluation across broader case set
- Test model swapping (e.g. Haiku/Sonnet for subtask agents)
- Investigate synthesis score (likely missing specific values in report)
- Add more hand-crafted cases for edge scenarios
- Extension point for classical NLP / rule-based judges beyond current deterministic checks

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
