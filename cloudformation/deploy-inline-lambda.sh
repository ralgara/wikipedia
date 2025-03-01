#!/bin/bash
set -e

# Configuration
STACK_NAME="wikipedia-stats-stack"
ENVIRONMENT="dev"
REGION="us-east-1"

echo "==============================================="
echo "  Wikipedia Pageviews Analytics Deployment"
echo "==============================================="
echo "Stack Name: $STACK_NAME"
echo "Environment: $ENVIRONMENT"
echo "Region: $REGION"
echo

# Deploy CloudFormation stack
echo "Deploying CloudFormation stack..."
aws cloudformation deploy \
  --template-file wikipedia-stats-template-inline.yaml \
  --stack-name $STACK_NAME \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM CAPABILITY_AUTO_EXPAND \
  --parameter-overrides \
    Environment=$ENVIRONMENT \
    RetentionDays=30 \
  --region $REGION

# Check if deployment was successful
if [ $? -eq 0 ]; then
  echo
  echo "✅ Deployment completed successfully!"
  echo
  echo "Stack outputs:"
  aws cloudformation describe-stacks \
    --stack-name $STACK_NAME \
    --query "Stacks[0].Outputs" \
    --output table \
    --region $REGION
  
  # Trigger the Lambda function manually to collect today's data
  echo
  echo "Triggering initial data collection..."
  aws lambda invoke \
    --function-name $(aws cloudformation describe-stacks --stack-name $STACK_NAME --query "Stacks[0].Outputs[?OutputKey=='CollectorFunctionName'].OutputValue" --output text --region $REGION) \
    --payload '{}' \
    --region $REGION \
    response.json
  
  echo "Check S3 bucket for collected data:"
  S3_BUCKET=$(aws cloudformation describe-stacks --stack-name $STACK_NAME --query "Stacks[0].Outputs[?OutputKey=='RawDataBucketName'].OutputValue" --output text --region $REGION)
  echo "aws s3 ls s3://$S3_BUCKET/data/ --recursive"
else
  echo
  echo "❌ Deployment failed. Please check CloudFormation events for details."
  exit 1
fi