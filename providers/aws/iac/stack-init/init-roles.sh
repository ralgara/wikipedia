#!/usr/bin/env bash -x

source init-globals.sh

# Create IAM role
aws iam create-role \
    --role-name $ROLE_NAME \
    --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {
                "Service": "lambda.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }]
    }'

echo "Creating bucket access policy ${ROLE_NAME}, ${BUCKET_NAME}"
# Create S3 bucket access policy
aws iam put-role-policy \
    --role-name $ROLE_NAME \
    --policy-name wikipedia-s3-access \
    --policy-document '{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Action": [
                "s3:PutObject"
            ],
            "Resource": "arn:aws:s3:::'"$BUCKET_NAME"'/*"
        }]
    }'

# Attach basic Lambda execution role
aws iam attach-role-policy \
    --role-name $ROLE_NAME \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

# Create S3 bucket access policy
aws iam put-role-policy \
    --role-name $ROLE_NAME \
    --policy-name wikipedia-s3-access \
    --policy-document '{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Action": [
                "s3:PutObject"
            ],
            "Resource": "arn:aws:s3:::'"$BUCKET_NAME"'/*"
        }]
    }'


# Wait for role creation to propagate
echo "Waiting for role to be available..."
sleep 3