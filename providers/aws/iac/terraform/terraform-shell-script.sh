#!/bin/bash
set -e

# Create project directory
mkdir -p terraform-lambda-layer
cd terraform-lambda-layer

# Create Lambda code directory
mkdir -p lambda-code

# Create Lambda code file
cat > lambda-code/index.py << 'EOF'
import json
import requests  # This will be available from the layer

def handler(event, context):
    # Example API call using requests from the layer
    response = requests.get('https://jsonplaceholder.typicode.com/todos/1')
    data = response.json()
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'message': 'Success! The requests package works from the layer.',
            'data': data
        })
    }
EOF

# Create layer content directory structure (Python packages must be in python/ subdirectory)
mkdir -p layer-content/python

# Install requests package into the layer directory
pip install requests -t layer-content/python/

# Create terraform configuration file
cat > main.tf << 'EOF'
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
  }
}

provider "aws" {
  region = "us-east-1"  # Modify as needed
}

# Archive file for the Lambda Layer
data "archive_file" "lambda_layer" {
  type        = "zip"
  source_dir  = "${path.module}/layer-content"
  output_path = "${path.module}/layer-package.zip"
}

# Lambda Layer resource
resource "aws_lambda_layer_version" "common_dependencies" {
  layer_name          = "common-dependencies"
  description         = "Common Python dependencies like requests"
  filename            = data.archive_file.lambda_layer.output_path
  source_code_hash    = data.archive_file.lambda_layer.output_base64sha256
  compatible_runtimes = ["python3.9"]
}

# Archive file for the Lambda function
data "archive_file" "lambda_function" {
  type        = "zip"
  source_file = "${path.module}/lambda-code/index.py"
  output_path = "${path.module}/function-package.zip"
}

# Lambda function resource
resource "aws_lambda_function" "function" {
  function_name    = "lambda-with-layer-example"
  filename         = data.archive_file.lambda_function.output_path
  source_code_hash = data.archive_file.lambda_function.output_base64sha256
  role             = aws_iam_role.lambda_exec.arn
  handler          = "index.handler"
  runtime          = "python3.9"
  timeout          = 30
  memory_size      = 128
  
  # Attach the layer to the function
  layers = [aws_lambda_layer_version.common_dependencies.arn]
}

# IAM Role for Lambda execution
resource "aws_iam_role" "lambda_exec" {
  name = "lambda_layer_example_role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })
}

# Basic execution policy for Lambda
resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda_exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# Outputs
output "function_name" {
  value = aws_lambda_function.function.function_name
}

output "layer_arn" {
  value = aws_lambda_layer_version.common_dependencies.arn
}
EOF

echo "Initializing Terraform..."
terraform init

echo "Running Terraform plan..."
terraform plan

echo "Ready to deploy! Run 'terraform apply' to deploy the Lambda function with layer."
echo "Directory setup completed at: $(pwd)"
