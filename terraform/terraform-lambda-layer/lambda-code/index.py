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
