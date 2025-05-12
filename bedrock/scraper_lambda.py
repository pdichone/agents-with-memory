import urllib.request
import os
import shutil
import json
import gzip
import io
from bs4 import BeautifulSoup
import sys  # Import sys to get the size of the response

MAX_RESPONSE_SIZE = 22000  # 22KB limit

def get_page_content(url):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            # Check if the content is compressed with GZIP
            if response.info().get('Content-Encoding') == 'gzip':
                print(f"Content from {url} is GZIP encoded, decompressing...")
                buf = io.BytesIO(response.read())
                with gzip.GzipFile(fileobj=buf) as f:
                    content = f.read().decode('utf-8')
            else:
                content = response.read().decode('utf-8')
            
            if response.geturl() != url:  # Check if there were any redirects
                print(f"Redirect detected for {url}")
                return None

            return content
    except Exception as e:
        print(f"Error while fetching content from {url}: {e}")
        return None

def empty_tmp_folder():
    try:
        for filename in os.listdir('/tmp'):
            file_path = os.path.join('/tmp', filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        print("Temporary folder emptied.")
        return "Temporary folder emptied."
    except Exception as e:
        print(f"Error while emptying /tmp folder: {e}")
        return None

def save_to_tmp(filename, content):
    try:
        if content is not None:
            print(content)
            with open(f'/tmp/{filename}', 'w') as file:
                file.write(content)
            print(f"Saved {filename} to /tmp")
            return f"Saved {filename} to /tmp"
        else:
            raise Exception("No content to save.")
    except Exception as e:
        print(f"Error while saving {filename} to /tmp: {e}")
        return None

def check_tmp_for_data(query):
    try:
        data = []
        for filename in os.listdir('/tmp'):
            if query in filename:
                with open(f'/tmp/{filename}', 'r') as file:
                    data.append(file.read())
        print(f"Found {len(data)} file(s) in /tmp for query {query}")
        return data if data else None
    except Exception as e:
        print(f"Error while checking /tmp for query {query}: {e}")
        return None

def handle_search(event):
    # Extract inputURL from the requestBody content
    request_body = event.get('requestBody', {})
    input_url = ''
    
    # Check if the inputURL exists within the properties
    if 'content' in request_body:
        properties = request_body['content'].get('application/json', {}).get('properties', [])
        input_url = next((prop['value'] for prop in properties if prop['name'] == 'inputURL'), '')

    # Handle missing URL
    if not input_url:
        return {"error": "No URL provided"}

    # Ensure URL starts with http or https
    if not input_url.startswith(('http://', 'https://')):
        input_url = 'http://' + input_url

    # Check for existing data in /tmp
    tmp_data = check_tmp_for_data(input_url)
    if tmp_data:
        return {"results": tmp_data}

    # Clear /tmp folder
    empty_tmp_result = empty_tmp_folder()
    if empty_tmp_result is None:
        return {"error": "Failed to empty /tmp folder"}

    # Get the page content
    content = get_page_content(input_url)
    if content is None:
        return {"error": "Failed to retrieve content"}

    # Parse and clean the HTML content
    cleaned_content = parse_html_content(content)

    # Save the content to /tmp
    filename = input_url.split('//')[-1].replace('/', '_') + '.txt'
    save_result = save_to_tmp(filename, cleaned_content)

    if save_result is None:
        return {"error": "Failed to save to /tmp"}

    # Check the size of the response and truncate if necessary
    response_data = {'url': input_url, 'content': cleaned_content}
    response_size = sys.getsizeof(json.dumps(response_data))

    if response_size > MAX_RESPONSE_SIZE:
        print(f"Response size {response_size} exceeds limit. Truncating content...")
        truncated_content = cleaned_content[:(MAX_RESPONSE_SIZE - response_size)]
        response_data['content'] = truncated_content

    return {"results": response_data}

def parse_html_content(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    cleaned_text = '\n'.join(chunk for chunk in chunks if chunk)

    max_size = 25000
    if len(cleaned_text) > max_size:
        cleaned_text = cleaned_text[:max_size]

    return cleaned_text

def lambda_handler(event, context):
    response_code = 200
    action_group = event['actionGroup']
    api_path = event['apiPath']

    print("THE EVENT: ", event)

    if api_path == '/search':
        result = handle_search(event)
    else:
        response_code = 404
        result = f"Unrecognized api path: {action_group}::{api_path}"

    response_body = {
        'application/json': {
            'body': result
        }
    }

    action_response = {
        'actionGroup': event['actionGroup'],
        'apiPath': event['apiPath'],
        'httpMethod': event['httpMethod'],
        'httpStatusCode': response_code,
        'responseBody': response_body
    }

    api_response = {'messageVersion': '1.0', 'response': action_response}
    print("action_response: ", action_response)
    print("response_body: ", response_body)
    return api_response