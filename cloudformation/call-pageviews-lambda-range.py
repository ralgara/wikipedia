#!/usr/bin/env python3

import boto3
import argparse
import json
import random
from datetime import datetime, timedelta
import time

def parse_date(date_str):
    """Parse date string in YYYY-MM-DD format."""
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        raise ValueError(f"Invalid date format: {date_str}. Please use YYYY-MM-DD.")

def generate_date_range(start_date, end_date):
    """Generate a list of dates between start_date and end_date, inclusive."""
    date_list = []
    current_date = start_date
    
    while current_date <= end_date:
        date_list.append(current_date)
        current_date += timedelta(days=1)
    
    return date_list

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Invoke a Lambda function for a range of dates.')
    parser.add_argument('--function-name', required=True, help='Lambda function name')
    parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--region', default='us-east-1', help='AWS region (default: us-east-1)')
    parser.add_argument('--async', action='store_true', help='Use async invocation')
    args = parser.parse_args()
    
    # Parse dates
    start_date = parse_date(args.start_date)
    end_date = parse_date(args.end_date)
    
    if start_date > end_date:
        raise ValueError("Start date must be before end date.")
    
    # Generate date range
    date_range = generate_date_range(start_date, end_date)
    
    # Initialize Lambda client
    lambda_client = boto3.client('lambda', region_name=args.region)
    
    # Set invocation type based on args
    invocation_type = 'RequestResponse'
    
    # Invoke Lambda function for each date
    for date in date_range:
        print(f"Invoking Lambda for date: {date.strftime('%Y-%m-%d')}")
        payload = {
            'date': date.strftime('%Y-%m-%d')
        }
        
        response = lambda_client.invoke(
            FunctionName=args.function_name,
            InvocationType=invocation_type,
            Payload=json.dumps(payload)
        )
        
        if invocation_type == 'RequestResponse':
            # For synchronous calls, print status code
            status_code = response['StatusCode']
            print(f"  Status code: {status_code}")
            
            payload = response['Payload'].read().decode('utf-8')
            print(f"  Response: {payload}")
        delay = random.randint(10,1000)
        print(f"Delay: {delay}")
        time.sleep(delay)
    
    print(f"Invoked Lambda function '{args.function_name}' for {len(date_range)} dates.")

if __name__ == "__main__":
    main()