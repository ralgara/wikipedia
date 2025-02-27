#!/usr/bin/env bash

# Set variables
AWS_REGION="us-east-1"  # Change this to your desired region
LAMBDA_LAYER_NAME="python-requests-layer"
PROJECT_DIR="python-deps-project"

# Create project directory
mkdir -p "${PROJECT_DIR}"
cd "${PROJECT_DIR}"

# Create http_wrapper.py
cat << 'EOF' > http_wrapper.py
import http.client
import json
import urllib.parse
from typing import Optional, Dict, Any, Union

class HttpWrapper:
    @staticmethod
    def get_json(url: str) -> Optional[Dict[str, Any]]:
        """
        Fetch JSON from a URL using http.client instead of requests
        
        Args:
            url: The URL to fetch from
            
        Returns:
            Parsed JSON data as dictionary or None if request fails
        """
        parsed_url = urllib.parse.urlparse(url)
        
        # Get hostname and path
        hostname = parsed_url.netloc
        path = parsed_url.path
        if parsed_url.query:
            path += '?' + parsed_url.query
        
        # Create connection
        conn = http.client.HTTPSConnection(hostname) if parsed_url.scheme == 'https' else http.client.HTTPConnection(hostname)
        
        try:
            # Send GET request
            conn.request('GET', path)
            
            # Get response
            response = conn.getresponse()
            
            # Check if request was successful
            if response.status == 200:
                # Read and parse JSON
                data = response.read().decode('utf-8')
                return json.loads(data)
            else:
                print(f"Error: Received status code {response.status}")
                return None
        except Exception as e:
            print(f"Error: {e}")
            return None
        finally:
            conn.close()

    @staticmethod
    def post_json(url: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Post JSON to a URL using http.client instead of requests
        
        Args:
            url: The URL to post to
            data: Dictionary to be sent as JSON
            
        Returns:
            Parsed JSON response as dictionary or None if request fails
        """
        parsed_url = urllib.parse.urlparse(url)
        
        # Get hostname and path
        hostname = parsed_url.netloc
        path = parsed_url.path
        if parsed_url.query:
            path += '?' + parsed_url.query
        
        # Convert data to JSON
        json_data = json.dumps(data)
        
        # Create connection
        conn = http.client.HTTPSConnection(hostname) if parsed_url.scheme == 'https' else http.client.HTTPConnection(hostname)
        
        try:
            # Set headers
            headers = {
                'Content-Type': 'application/json',
                'Content-Length': str(len(json_data))
            }
            
            # Send POST request
            conn.request('POST', path, json_data, headers)
            
            # Get response
            response = conn.getresponse()
            
            # Check if request was successful
            if response.status in [200, 201]:
                # Read and parse JSON
                data = response.read().decode('utf-8')
                return json.loads(data)
            else:
                print(f"Error: Received status code {response.status}")
                return None
        except Exception as e:
            print(f"Error: {e}")
            return None
        finally:
            conn.close()
EOF

# Create example usage script
cat << 'EOF' > example.py
from http_wrapper import HttpWrapper

# Example GET request
result = HttpWrapper.get_json('https://jsonplaceholder.typicode.com/todos/1')
print("GET result:", result)

# Example POST request
post_data = {
    'title': 'foo',
    'body': 'bar',
    'userId': 1
}
post_result = HttpWrapper.post_json('https://jsonplaceholder.typicode.com/posts', post_data)
print("POST result:", post_result)
EOF

# Create Lambda layer directory structure
mkdir -p lambda-layer/python
cd lambda-layer

# Create requirements.txt for Lambda layer
cat << 'EOF' > requirements.txt
requests==2.31.0
EOF

# Install dependencies for Lambda layer
pip install -r requirements.txt -t python/

# Create Lambda layer
echo "Creating Lambda layer..."
zip -r layer.zip python/
aws lambda publish-layer-version \
    --layer-name "${LAMBDA_LAYER_NAME}" \
    --description "Layer containing requests package" \
    --zip-file fileb://layer.zip \
    --compatible-runtimes python3.8 python3.9 \
    --region "${AWS_REGION}"

echo "Setup complete!"
echo "Lambda layer: ${LAMBDA_LAYER_NAME}"

