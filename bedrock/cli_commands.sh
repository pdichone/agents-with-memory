
aws bedrock-runtime invoke-model \
  --region us-west-2 \
  --model-id us.anthropic.claude-3-7-sonnet-20250219-v1:0 \
  --body '{
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 512,
    "temperature": 0.5,
    "top_p": 0.9,
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Describe the purpose of a \"hello world\" program in one line."
          }
        ]
      }
    ]
  }' \
  --cli-binary-format raw-in-base64-out \
  --content-type application/json \
  --accept application/json \
  invoke-model-output-text.json



# Submit a text prompt to a model and generate a text response with Converse
aws bedrock-runtime converse \
--model-id us.anthropic.claude-3-7-sonnet-20250219-v1:0 \
--messages '[{"role": "user", "content": [{"text": "Describe the purpose of a \"hello world\" program in one line."}]}]' \
--inference-config '{"maxTokens": 512, "temperature": 0.5, "topP": 0.9}'
