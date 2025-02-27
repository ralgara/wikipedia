#!/usr/bin/env bash

aws cloudformation create-stack \
  --stack-name http-wrapper-solution \
  --template-body file://claudeformation-template.yaml \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameters ParameterKey=Environment,ParameterValue=dev