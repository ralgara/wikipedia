# Set variables
export BUCKET_NAME="wikipedia-raw-data"
export FUNCTION_NAME="wikipedia-downloader"
export ROLE_NAME="wikipedia-downloader-role"
export REGION="us-east-1"  # Change as needed
export ATHENA_DATABASE="wikipedia"
export S3_QUERY_RESULTS="s3://wikipedia-raw-data/athena-results/"  # This bucket/prefix must exist
