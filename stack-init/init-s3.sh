#!/usr/bin/env bash -x

source init-globals.sh

# Create bucket
aws s3api create-bucket \
  --bucket wikipedia-raw-data \
  --region us-east-1

# Enable versioning
aws s3api put-bucket-versioning \
  --bucket wikipedia-raw-data \
  --versioning-configuration Status=Enabled

# Block public access
aws s3api put-public-access-block \
  --bucket wikipedia-raw-data \
  --public-access-block-configuration "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
