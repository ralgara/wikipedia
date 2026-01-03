#!/usr/bin/env bash -x

source init-globals.sh

SCRIPT_NAME="wikipedia-downloader-lambda"

echo "Creating lambda for ${SCRIPT_NAME}"

# Create deployment package
zip -j function.zip ../${SCRIPT_NAME}.py

# Create Lambda function
ROLE_ARN=$(aws iam get-role --role-name $ROLE_NAME --query 'Role.Arn' --output text)

aws lambda delete-function \
    --function-name $FUNCTION_NAME

aws lambda create-function \
    --function-name $FUNCTION_NAME \
    --runtime python3.9 \
    --handler $SCRIPT_NAME.lambda_handler \
    --role $ROLE_ARN \
    --zip-file fileb://function.zip \
    --timeout 30 \
    --memory-size 256 \
    --environment "Variables={BUCKET_NAME=$BUCKET_NAME}" \
    --region $REGION

# Clean up deployment package
rm function.zip

echo "Setup complete. Check response.json for test results."