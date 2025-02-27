import json
import os
import urllib3
from datetime import datetime, timedelta
import logging

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize clients
http = urllib3.PoolManager()

def get_s3_key(date):
    """Generate S3 key with partitioning structure"""
    return f"wikipedia/pageviews/year={date.year}/month={date.strftime('%m')}/day={date.strftime('%d')}/pageviews_{date.strftime('%Y%m%d')}.json"

def download_pageviews(date):
    """Download pageviews data from Wikipedia API"""
    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/top/en.wikipedia/all-access/{date.year}/{date.strftime('%m')}/{date.strftime('%d')}"
    
    try:
        logger.info(f"Downloading data from {url}")
        response = http.request('GET', url)
        
        if response.status != 200:
            raise Exception(f"API request failed with status {response.status}")

        data = response.read().decode('utf-8')
        s = json.loads(data)
        s = s['items'][0]
        date = f"{s['year']}-{s['month']}-{s['day']}"
        return [ {**z, 'date':date} for z in s['articles'] ]
    except Exception as e:
        logger.error(f"Error downloading data: {str(e)}")
        raise

def process_single_date(date):
    """Process a single date"""
    bucket = os.environ['BUCKET_NAME']
    
    try:
        # Download data
        data = download_pageviews(date)
        
        # Store in S3
        s3_key = get_s3_key(date)
        logger.info(f"Storing data in s3://{bucket}/{s3_key}")
        
        s3.put_object(
            Bucket=bucket,
            Key=s3_key,
            Body=data,
            ContentType='application/json',
            Metadata={
                'source': 'wikipedia-pageviews',
                'date': date.strftime('%Y-%m-%d')
            }
        )
        
        return {
            'date': date.strftime('%Y-%m-%d'),
            'bucket': bucket,
            'key': s3_key,
            'status': 'success'
        }
        
    except Exception as e:
        logger.error(f"Error processing date {date}: {str(e)}")
        return {
            'date': date.strftime('%Y-%m-%d'),
            'status': 'error',
            'error': str(e)
        }

def lambda_handler(event, context):
    # Handle single date
    if 'date' in event:
        date = datetime.strptime(event['date'], '%Y-%m-%d')
        result = process_single_date(date)
        return {
            'statusCode': 200 if result['status'] == 'success' else 500,
            'body': json.dumps(result)
        }
    
    # Handle date range
    elif 'start_date' in event and 'end_date' in event:
        start_date = datetime.strptime(event['start_date'], '%Y-%m-%d')
        end_date = datetime.strptime(event['end_date'], '%Y-%m-%d')
        
        results = []
        current_date = start_date
        
        while current_date <= end_date:
            result = process_single_date(current_date)
            results.append(result)
            current_date += timedelta(days=1)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Batch processing complete',
                'results': results
            })
        }
    
    else:
        # Default to yesterday if no date specified
        date = datetime.utcnow() - timedelta(days=1)
        result = process_single_date(date)
        return {
            'statusCode': 200 if result['status'] == 'success' else 500,
            'body': json.dumps(result)
        }