import streamlit as st
from boto3.session import Session
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.credentials import Credentials
import json
import os
import uuid
import base64
import io
import sys
from requests import request
from dotenv import load_dotenv
from PIL import Image, ImageOps, ImageDraw
import pandas as pd

# Load environment variables from .env file if it exists
load_dotenv()

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "session_id" not in st.session_state:
    st.session_state.session_id = f"session-{uuid.uuid4()}"

# Page configuration
st.set_page_config(page_title="Web Crawler Agent", page_icon="🕸️", layout="wide")

# Get agent credentials from environment variables
agent_id = os.environ.get("BEDROCK_AGENT_ID", "")
agent_alias_id = os.environ.get("BEDROCK_AGENT_ALIAS_ID", "")
region = os.environ.get("AWS_REGION", "us-west-2")

# Sidebar for agent configuration
with st.sidebar:
    st.title("Agent Configuration")

    # Agent settings
    agent_id = st.text_input("Agent ID", value=agent_id)
    agent_alias_id = st.text_input("Agent Alias ID", value=agent_alias_id)
    region = st.text_input("AWS Region", value=region)

    # Set environment variable for region
    os.environ["AWS_REGION"] = region

    # Debug information
    with st.expander("Debug Info", expanded=False):
        st.write("Session ID:", st.session_state.session_id)

        # Display AWS credentials status
        creds_available = (
            "Yes"
            if os.environ.get("AWS_ACCESS_KEY_ID")
            and os.environ.get("AWS_SECRET_ACCESS_KEY")
            else "No"
        )
        st.write("AWS Credentials Available:", creds_available)

    # Trace data display
    st.subheader("Trace Data")
    if "trace_data" in st.session_state:
        st.text_area("", value=st.session_state.trace_data, height=300)
    else:
        st.text_area("", value="Trace data will appear here", height=300)

# Main content area
st.title("Web Crawler Agent")


# SigV4 authentication for AWS API requests
def sigv4_request(
    url,
    method="GET",
    body=None,
    params=None,
    headers=None,
    service="bedrock",
    region=os.environ.get("AWS_REGION", "us-west-2"),
    credentials=Session().get_credentials().get_frozen_credentials(),
):
    """
    Sends an HTTP request signed with SigV4 authentication
    """
    # Sign request
    req = AWSRequest(method=method, url=url, data=body, params=params, headers=headers)
    SigV4Auth(credentials, service, region).add_auth(req)
    req = req.prepare()

    # Send request
    return request(method=req.method, url=req.url, headers=req.headers, data=req.body)


# Function to ask question to agent and handle response
def ask_agent(question, session_id, end_session=False):
    # Form the URL using agent ID, alias ID, and session ID
    url = f"https://bedrock-agent-runtime.{region}.amazonaws.com/agents/{agent_id}/agentAliases/{agent_alias_id}/sessions/{session_id}/text"

    # Prepare request body
    request_body = {
        "inputText": question,
        "enableTrace": True,
        "endSession": end_session,
    }

    # Send request with SigV4 authentication
    response = sigv4_request(
        url=url,
        method="POST",
        headers={
            "content-type": "application/json",
            "accept": "application/json",
        },
        body=json.dumps(request_body),
    )

    # Process and decode the response
    return decode_response(response)


# Function to decode the streaming response from Bedrock Agent
def decode_response(response):
    # Capture output for debugging
    captured_output = io.StringIO()
    sys.stdout = captured_output

    # Decode the response
    string = ""
    for line in response.iter_content():
        try:
            string += line.decode(encoding="utf-8")
        except:
            continue

    print("Decoded response:", string)

    # Process the response to extract the actual content
    split_response = string.split(":message-type")
    print(f"Split Response: {split_response}")
    print(f"Length of split: {len(split_response)}")

    # Look for bytes content in all parts
    for idx in range(len(split_response)):
        if "bytes" in split_response[idx]:
            print(f"Bytes found index {idx}")
            try:
                encoded_response = split_response[idx].split('"')[3]
                decoded = base64.b64decode(encoded_response)
                final_response = decoded.decode("utf-8")
                print(final_response)
            except Exception as e:
                print(f"Error decoding bytes at index {idx}: {str(e)}")
        else:
            print(f"No bytes at index {idx}")
            print(split_response[idx])

    # Process the last part of the response
    last_response = split_response[-1]
    print(f"Last Response: {last_response}")

    if "bytes" in last_response:
        print("Bytes in last response")
        try:
            encoded_last_response = last_response.split('"')[3]
            decoded = base64.b64decode(encoded_last_response)
            final_response = decoded.decode("utf-8")
        except Exception as e:
            print(f"Error decoding last response: {str(e)}")
            final_response = "Error decoding response"
    else:
        print("No bytes in last response")
        try:
            part1 = string[string.find("finalResponse") + len('finalResponse":') :]
            part2 = part1[: part1.find('"}') + 2]
            final_response = json.loads(part2)["text"]
        except Exception as e:
            print(f"Error extracting text from finalResponse: {str(e)}")
            final_response = "Error processing response"

    # Clean up the response
    final_response = final_response.replace('"', "")
    final_response = final_response.replace("{input:{value:", "")
    final_response = final_response.replace(",source:null}}", "")

    # Restore original stdout
    sys.stdout = sys.__stdout__

    # Get the string from captured output
    debug_output = captured_output.getvalue()

    # Return both the debug output and the final response
    return debug_output, final_response


# Function to process user input and get agent response
def process_input(question, end_session=False):
    if not agent_id or not agent_alias_id:
        return "Please configure Agent ID and Agent Alias ID in the sidebar."

    try:
        # Call the agent with the question
        debug_output, agent_response = ask_agent(
            question=question,
            session_id=st.session_state.session_id,
            end_session=end_session,
        )

        # Store trace data
        st.session_state.trace_data = debug_output

        # Return the agent's response
        return agent_response
    except Exception as e:
        # If there's an error, display it
        error_message = f"Error: {str(e)}"
        st.session_state.trace_data = error_message
        return error_message


# Display a text input field for user questions
user_input = st.text_input(
    "Ask about a website:", placeholder="Crawl this URL: https://example.com"
)

# Create columns for buttons
col1, col2, col3 = st.columns([1, 1, 3])

with col1:
    # Submit button
    if st.button("Submit", type="primary"):
        if user_input:
            # Show spinner while processing
            with st.spinner("Crawling web content..."):
                # Get response from agent
                response = process_input(user_input)

                # Add to conversation history
                st.session_state.history.append(
                    {"question": user_input, "answer": response}
                )

with col2:
    # End session button
    if st.button("End Session"):
        # End the session and reset
        process_input("End session", end_session=True)
        st.session_state.history.append(
            {
                "question": "Session Ended",
                "answer": "Thank you for using the Web Crawler Agent!",
            }
        )
        # Generate a new session ID
        st.session_state.session_id = f"session-{uuid.uuid4()}"
        st.rerun()

with col3:
    # Test URL input for quick testing
    test_url = st.text_input("Test URL:", placeholder="https://example.com")
    if st.button("Crawl URL"):
        if test_url:
            crawl_command = f"Crawl this website: {test_url}"

            # Get response from agent
            with st.spinner("Crawling web content..."):
                response = process_input(crawl_command)

                # Add to conversation history
                st.session_state.history.append(
                    {"question": crawl_command, "answer": response}
                )

# Display conversation history
st.write("## Conversation History")

# Display the conversation history
for idx, chat in enumerate(reversed(st.session_state.history)):
    # Create a unique key for each chat message
    q_key = f"q_{idx}"
    a_key = f"a_{idx}"

    # Display the user's question
    with st.container():
        st.markdown(f"**You:** {chat['question']}")

    # Display the agent's answer
    with st.container():
        st.markdown(f"**Agent:** {chat['answer']}")

    # Add a divider between conversations
    st.divider()

# Example prompts section
st.write("## Example Prompts")

example_prompts = [
    "Crawl this website: https://example.com and summarize the content",
    "Search for information about AWS Bedrock Agents",
    "Crawl this URL and tell me about the main features: https://aws.amazon.com/bedrock/",
    "What is the current weather in New York City?",
    "Find recent news about artificial intelligence",
]

# Display example prompts as a table
example_df = pd.DataFrame({"Example Prompts": example_prompts})
st.table(example_df)
