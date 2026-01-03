# DigitalOcean Provider

Wikipedia pageviews infrastructure on DigitalOcean.

## Structure

```
digitalocean/
├── iac/                    # Infrastructure as Code (App Spec, Terraform)
├── app/                    # Application code (App Platform / Functions)
└── services/               # Service abstractions
```

## Services (planned)

- **Storage**: Spaces (S3-compatible)
- **Compute**: App Platform / Functions
- **Scheduling**: App Platform scheduled jobs
- **Query**: Managed PostgreSQL / external
- **Time Series**: Managed PostgreSQL / external
