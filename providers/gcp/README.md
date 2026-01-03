# GCP Provider

Wikipedia pageviews infrastructure on Google Cloud Platform.

## Structure

```
gcp/
├── iac/                    # Infrastructure as Code (Deployment Manager, Terraform)
├── app/                    # Application code (Cloud Functions)
└── services/               # Service abstractions
```

## Services (planned)

- **Storage**: Cloud Storage
- **Compute**: Cloud Functions
- **Scheduling**: Cloud Scheduler
- **Query**: BigQuery
- **Time Series**: Cloud Monitoring / BigQuery
