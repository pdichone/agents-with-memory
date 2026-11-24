# Model: Claude Sonnet 5.5 via the US cross-Region inference profile (works from us-west-2).
# Claude Sonnet 5.5 rejects temperature / top_p, so they are not sent.
# It thinks by default, so give it a little more room than 512 tokens.
aws bedrock-runtime invoke-model \
  --region us-west-2 \
  --model-id us.anthropic.claude-sonnet-5-5 \
  --body '{
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 1024,
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
--region us-west-2 \
--model-id us.anthropic.claude-sonnet-5-5 \
--messages '[{"role": "user", "content": [{"text": "Describe the purpose of a \"hello world\" program in one line."}]}]' \
--inference-config '{"maxTokens": 1024}'
