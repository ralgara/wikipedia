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
