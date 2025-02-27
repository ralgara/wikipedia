#!/usr/bin/env bash -x

source init-globals.sh

# Create DDL file
cat > create_table.sql << 'EOF'
--CREATE DATABASE IF NOT EXISTS wikipedia;

CREATE EXTERNAL TABLE wikipedia.pageviews (
  items ARRAY<STRUCT<
    project: STRING,
    access: STRING,
    year: STRING,
    month: STRING,
    day: STRING,
    articles: ARRAY<STRUCT<
      article: STRING,
      views: BIGINT,
      rank: INT
    >>
  >>
)
PARTITIONED BY (
  year STRING,
  month STRING,
  day STRING
)
ROW FORMAT SERDE 'org.openx.data.jsonserde.JsonSerDe'
LOCATION 's3://wikipedia-raw-data/'
TBLPROPERTIES ('has_encrypted_data'='false');
EOF

# Execute the query
QUERY_ID=$(aws athena start-query-execution \
    --query-string file://create_table.sql \
    --query-execution-context Database=${ATHENA_DATABASE} \
    --result-configuration OutputLocation=${S3_QUERY_RESULTS} \
    --region ${REGION} \
    --query 'QueryExecutionId' \
    --output text)

echo "Started query execution with ID: ${QUERY_ID}"

# Wait for query to complete
while true; do
    STATUS=$(aws athena get-query-execution \
        --query-execution-id $QUERY_ID \
        --region ${REGION} \
        --query 'QueryExecution.Status.State' \
        --output text)
    
    echo "Query status: ${STATUS}"
    
    if [ "$STATUS" = "SUCCEEDED" ]; then
        break
    elif [ "$STATUS" = "FAILED" ] || [ "$STATUS" = "CANCELLED" ]; then
        echo "Query failed or was cancelled"
        aws athena get-query-execution --query-execution-id $QUERY_ID --region ${REGION}
        exit 1
    fi
    
    sleep 2
done

echo "Table creation completed successfully"

# Load partitions
REPAIR_QUERY_ID=$(aws athena start-query-execution \
    --query-string "MSCK REPAIR TABLE wikipedia.pageviews;" \
    --query-execution-context Database=${ATHENA_DATABASE} \
    --result-configuration OutputLocation=${S3_QUERY_RESULTS} \
    --region ${REGION} \
    --query 'QueryExecutionId' \
    --output text)

echo "Started partition repair with ID: ${REPAIR_QUERY_ID}"