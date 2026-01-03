#!/bin/bash

# Define your Lambda function name
LAMBDA_FUNCTION_NAME="wikipedia-pageviews-collector-dev"

# Define the AWS region
AWS_REGION="us-east-1"

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "Error: AWS CLI is not installed. Please install it first."
    exit 1
fi

# Create a temporary file to store the dates
DATES_FILE='./missing-dates'

echo "Starting to process dates..."

# Loop through each date and invoke the Lambda function
while IFS= read -r date; do
    echo "Processing date: $date"
    
    # Create the payload JSON with the date
    PAYLOAD="{\"date\": \"$date\"}"
    
    # Invoke the Lambda function with the date payload
    aws lambda invoke \
        --function-name "$LAMBDA_FUNCTION_NAME" \
        --region "$AWS_REGION" \
        --payload "$PAYLOAD" \
        --cli-binary-format raw-in-base64-out \
        /tmp/lambda-response-$date.json
    
    # Check the exit status
    if [ $? -eq 0 ]; then
        echo "Successfully processed $date"
    else
        echo "Failed to process $date"
    fi
    
    # Optional: Add a small delay between invocations
    sleep 4
done < "$DATES_FILE"

# Clean up the temporary file
rm "$DATES_FILE"

echo "All dates have been processed."