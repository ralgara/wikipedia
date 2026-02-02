"""AWS Lambda handler for Wikipedia pageviews collection.

Uses shared library for cloud-neutral operations.
"""

import json
import logging
import os
from datetime import datetime, timedelta

import boto3

# Import from shared library
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../shared'))
from wikipedia import download_pageviews, generate_storage_key

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client('s3')


def process_single_date(date: datetime, use_bucket: bool = True) -> dict:
    """Process a single date - fetch from API and store."""

    # Cloud-neutral: fetch from Wikipedia API
    data = download_pageviews(date)

    if use_bucket:
        # AWS-specific: store to S3
        bucket = os.environ['BUCKET_NAME']
        key = generate_storage_key(date)

        logger.info(f"Storing data in s3://{bucket}/{key}")

        s3.put_object(
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
    else:
        # Local file (for testing)
        file_name = f"pageviews_{date.strftime('%Y%m%d')}.json"
        with open(file_name, 'w') as f:
            json.dump(data, f)
        return {
            'date': date.strftime('%Y-%m-%d'),
            'file_name': file_name,
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


if __name__ == "__main__":
    # Local testing without S3
    result = process_single_date(datetime(2025, 1, 28), use_bucket=False)
    print(json.dumps(result, indent=2))
