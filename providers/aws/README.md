# AWS Provider

Wikipedia pageviews infrastructure on Amazon Web Services.

## Structure

```
aws/
├── iac/                    # Infrastructure as Code
│   ├── cloudformation/     # SAM/CloudFormation templates
│   ├── terraform/          # Terraform configs
│   ├── stack-init/         # Shell-based stack initialization
│   └── localstack/         # Local development with LocalStack
├── app/                    # Application code
│   └── lambda/             # Lambda function handlers
└── services/               # Service abstractions (S3, EventBridge, etc.)
```

## Services Used

- **Storage**: S3
- **Compute**: Lambda
- **Scheduling**: EventBridge
- **Query**: Athena + Glue
- **Time Series**: Timestream
