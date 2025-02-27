#!/usr/bin/env bash -x

source init-globals.sh

./init-roles.sh
./init-s3.sh
./init-lambda.sh
./init-athena.sh
