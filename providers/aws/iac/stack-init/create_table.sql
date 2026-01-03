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
LOCATION 's3://wikipedia-raw-data/wikipedia/pageviews'
TBLPROPERTIES ('has_encrypted_data'='false');
