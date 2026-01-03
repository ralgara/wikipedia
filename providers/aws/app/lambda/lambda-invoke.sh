#!/usr/bin/env bash

FUNCTION_NAME=wikipedia-downloader

# Test the function
aws lambda invoke \
    --function-name $FUNCTION_NAME \
    --payload '{"date": "2025-02-14"}' \
    response.json