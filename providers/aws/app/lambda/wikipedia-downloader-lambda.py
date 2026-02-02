"""AWS Lambda handler for Wikipedia pageviews collection.

Uses shared library for cloud-neutral operations.
For local testing, use scripts/download-pageviews.py instead.
"""

import json
import logging
import os
from datetime import datetime, timedelta

# Import from shared library
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../shared'))
from wikipedia import download_pageviews, generate_storage_key

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Lazy-load boto3
_s3 = None
def get_s3_client():
    global _s3
    if _s3 is None:
        import boto3
        _s3 = boto3.client('s3')
    return _s3


def process_single_date(date: datetime) -> dict:
    """Process a single date - fetch from API and store to S3."""
    bucket = os.environ['BUCKET_NAME']
    key = generate_storage_key(date)

    # Cloud-neutral: fetch from Wikipedia API
    data = download_pageviews(date)

    # AWS-specific: store to S3
    logger.info(f"Storing data in s3://{bucket}/{key}")

    get_s3_client().put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(data),
        ContentType='application/json',
        Metadata={
            'source': 'wikipedia-pageviews',
            'date': date.strftime('%Y-%m-%d')
        }
    )

    return {
        'date': date.strftime('%Y-%m-%d'),
        'bucket': bucket,
        'key': key,
        'status': 'success'
    }


def lambda_handler(event, context):
    """AWS Lambda entry point."""

    # Single date
    if 'date' in event:
        date = datetime.strptime(event['date'], '%Y-%m-%d')
        result = process_single_date(date)
        return {
            'statusCode': 200 if result['status'] == 'success' else 500,
            'body': json.dumps(result)
        }

    # Date range
    elif 'start_date' in event and 'end_date' in event:
        start_date = datetime.strptime(event['start_date'], '%Y-%m-%d')
        end_date = datetime.strptime(event['end_date'], '%Y-%m-%d')

        results = []
        current = start_date
        while current <= end_date:
            result = process_single_date(current)
            results.append(result)
            current += timedelta(days=1)

        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Batch processing complete',
                'results': results
            })
        }

    # Default: yesterday
    else:
        date = datetime.utcnow() - timedelta(days=1)
        result = process_single_date(date)
        return {
            'statusCode': 200 if result['status'] == 'success' else 500,
            'body': json.dumps(result)
        }
