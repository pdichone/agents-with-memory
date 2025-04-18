from mem0 import AsyncMemoryClient
import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import asyncio


# Load environment variables
load_dotenv()

# Set up OpenAI client
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)


client = AsyncMemoryClient(api_key=os.getenv("MEM0_API_KEY"))


# ==== Add memory ===
# messages = [
#     {
#         "role": "user",
#         "content": "Hi, I'm Alex. I'm a vegetarian and I'm allergic to nuts.",
#     },
#     {
#         "role": "assistant",
#         "content": "Hello Alex! I've noted that you're a vegetarian and have a nut allergy. I'll keep this in mind for any food-related recommendations or discussions.",
#     },
# ]


async def main():

    messages = [
        {
            "role": "user",
            "content": "Hi, I'm Alex. I'm a vegetarian and I'm allergic to nuts.",
        },
        {
            "role": "assistant",
            "content": "Hello Alex! I've noted that you're a vegetarian and have a nut allergy. I'll keep this in mind for any food-related recommendations or discussions.",
        },
    ]

    # The default output_format is v1.0
    # response = await client.add(
    #     messages,
    #     user_id="alex",
    #     output_format="v1.1",
    #     metadata={"food": "vegan"},
    #     version="v2",
    # )

    # query = "Can Alex eat nuts?"
    # mem = await client.search(query, user_id="alex")
    # print("Memory Search Results:")
    # print(json.dumps(mem, indent=2))

    query = "What do you know about alex?"

    mem = await client.search(query, user_id="alex", output_format="v1.1", version="v2")
    print("Memory Search Results:")
    print(json.dumps(mem, indent=2))


asyncio.run(main())

# client.add(messages, user_id="alex")


# # == Retrieve memory ===
# query = "What can I cook for dinner tonight?"
# mem = client.search(query, user_id="alex")
# print("Memory Search Results:")
# print(json.dumps(mem, indent=2))
