#!/usr/bin/env python3

import sys
import re
from datetime import datetime, timedelta

def increment_date(date_str):
    """Increment a date string by one day"""
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    next_date = date_obj + timedelta(days=1)
    return next_date.strftime("%Y-%m-%d")

def main():
    """
    Process a list of filenames from stdin, extract dates, and find gaps in the sequence.
    Assumes the input is already sorted by date.
    """
    print("Processing input filenames (assuming already sorted)...")
    
    # Initialize variables
    prev_date = None
    gap_count = 0
    total_dates = 0
    
    # Regular expression to extract date from filename
    date_pattern = re.compile(r'pageviews-(\d{4}-\d{2}-\d{2})')
    
    # Process each line of input
    for line in sys.stdin:
        line = line.strip()
        match = date_pattern.search(line)
        
        if match:
            current_date = match.group(1)
            total_dates += 1
            
            # Skip the first iteration (no previous date to compare)
            if prev_date:
                # Calculate expected date (previous date + 1 day)
                expected_date = increment_date(prev_date)
                
                # If current date is not the expected date, we found a gap
                if expected_date != current_date:
                    print(f"Gap found: {prev_date} is not followed by {expected_date}")
                    
                    # Show the range of missing dates
                    missing_date = expected_date
                    while missing_date != current_date:
                        print(f"  Missing date: {missing_date}")
                        missing_date = increment_date(missing_date)
                        gap_count += 1
            
            # Update previous date for next iteration
            prev_date = current_date
    
    print(f"Found {total_dates} unique dates in the input.")
    
    # Print summary
    if gap_count == 0:
        print("Analysis complete. No gaps found in the date sequence.")
    else:
        print(f"Analysis complete. Found {gap_count} missing date(s) in the sequence.")

if __name__ == "__main__":
    main()
