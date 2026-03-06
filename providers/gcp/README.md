# GCP Provider

Wikipedia pageviews pipeline on Google Cloud Platform.

## Structure

```
gcp/
├── Dockerfile               # Alpine + amd64; lean deps (no scipy/statsmodels)
├── requirements.txt         # pandas, numpy, matplotlib, seaborn, google-cloud-storage
├── app/
│   └── pipeline.py          # download → GCS upload → report → GCS upload
└── ops/
    └── run-local.sh         # docker run wrapper for local / cron execution
```

## GCP Resources

| Resource | Name | Status |
|----------|------|--------|
| Project | `wikipedia-cortex` | Active |
| Bucket | `gs://wikipedia-cortex-data` | Active (uniform bucket access) |
| Service account | `wikipedia-pipeline@wikipedia-cortex.iam.gserviceaccount.com` | Active |
| Cloud Run Job | `wikipedia-pipeline` | Torn down |
| Cloud Scheduler | `wikipedia-daily` | Torn down |

## One-Time Setup

### 1. Build the Docker image

Run from the repo root (requires Docker Desktop or Docker CLI):

```bash
docker build --platform linux/amd64 \
  -f providers/gcp/Dockerfile \
  -t wikipedia-pipeline .
```

### 2. Create a service account key

```bash
gcloud iam service-accounts keys create ~/secrets/wikipedia-pipeline-key.json \
  --iam-account wikipedia-pipeline@wikipedia-cortex.iam.gserviceaccount.com
```

Store the key somewhere safe outside the repo (e.g. `~/secrets/`). Never commit it.

## Running the Pipeline

```bash
SA_KEY_PATH=~/secrets/wikipedia-pipeline-key.json \
  ./providers/gcp/ops/run-local.sh
```

Optional overrides:

```bash
REPORT_DAYS=90 \
SA_KEY_PATH=~/secrets/wikipedia-pipeline-key.json \
  ./providers/gcp/ops/run-local.sh
```

## Cron Setup (PC / cortex-puck)

Add a daily cron job that runs at 06:00 local time (Wikimedia data lags 2-5 days,
so exact timing doesn't matter):

```bash
crontab -e
```

Add this line (adjust paths to match your machine):

```
0 6 * * * SA_KEY_PATH=$HOME/secrets/wikipedia-pipeline-key.json \
  /path/to/wikipedia/providers/gcp/ops/run-local.sh \
  >> $HOME/logs/wikipedia-pipeline.log 2>&1
```

Create the log directory first:

```bash
mkdir -p ~/logs
```

To verify the cron entry was saved:

```bash
crontab -l
```

## What the Pipeline Does

1. **Download** — fetches top-1000 English pageviews for the most recent available date
   (Wikimedia API lags 2–5 days; pipeline tries day-2 through day-5)
2. **Upload JSON** — stores to `gs://wikipedia-cortex-data/wikipedia/pageviews/year=YYYY/month=MM/day=DD/pageviews_YYYYMMDD.json` (public)
3. **Sync recent JSON** — pulls the last N days from GCS to generate the report
4. **Generate report** — runs `scripts/generate-report.py --days $REPORT_DAYS`
5. **Upload report** — stores to `gs://wikipedia-cortex-data/reports/latest.html` and `reports/daily/report_YYYYMMDD.html` (public)

## Advanced Analytics

`scipy`, `PyWavelets`, and `statsmodels` are **not** in the Docker image (Alpine size/build
constraints). Run advanced analysis scripts directly in the local venv:

```bash
source .venv/bin/activate
python scripts/analyze-deep.py
```
