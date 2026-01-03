# Azure Provider

Wikipedia pageviews infrastructure on Microsoft Azure.

## Structure

```
azure/
├── iac/                    # Infrastructure as Code (ARM templates, Bicep, Terraform)
├── app/                    # Application code (Azure Functions)
└── services/               # Service abstractions
```

## Services (planned)

- **Storage**: Blob Storage
- **Compute**: Azure Functions
- **Scheduling**: Logic Apps / Timer Triggers
- **Query**: Data Explorer / Synapse
- **Time Series**: Data Explorer
