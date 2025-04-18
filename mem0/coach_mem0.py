"""
Interactive Health Coach with Memory using Mem0

This implementation allows users to have a real-time conversation with the health coach via console.
"""

import os
import json
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from mem0 import AsyncMemoryClient

# Load environment variables
load_dotenv()

# Set up OpenAI client
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# Set up Mem0 client
memory_client = AsyncMemoryClient(api_key=os.getenv("MEM0_API_KEY"))


# System prompt for the health coach
SYSTEM_PROMPT = """
You are a personal health coach with access to memory. Your goal is to help users achieve their fitness goals
by providing personalized advice and tracking their progress over time.

You have information about the user's past interactions that you can use to provide personalized advice.

Always be supportive, encouraging, and provide actionable advice.
"""


async def store_memory(user_id, messages):
    """
    Store conversation in Mem0
    """
    try:
        response = await memory_client.add(
            messages, user_id=user_id, output_format="v1.1", version="v2"
        )
        return True
    except Exception as e:
        print(f"⚠️ Memory storage error: {str(e)}")
        return False


async def search_memory(user_id, query):
    """
    Search for relevant memories in Mem0
    """
    try:
        memories = await memory_client.search(
            query, user_id=user_id, output_format="v1.1", version="v2"
        )
        return memories
    except Exception as e:
        print(f"⚠️ Memory search error: {str(e)}")
        return []


async def run_interactive_health_coach():
    """
    Interactive console interface for the health coach with memory capabilities
    """
    print("\n" + "=" * 80)
    print("🏃‍♂️ WELCOME TO YOUR PERSONAL HEALTH COACH 🏋️‍♀️")
    print("=" * 80)
    print(
        "I'm your AI health coach with memory. I can provide personalized fitness advice"
    )
    print("and will remember details about your fitness journey between conversations.")
    print("\nType 'exit', 'quit', or 'bye' to end our conversation.\n")

    # User ID - in a real app, this would be linked to user accounts
    user_id = "user123"

    # Initialize conversation history for the current session
    conversation_history = []

    # Start the conversation loop
    while True:
        # Get user input
        user_message = input("\nYou: ")

        # Check if user wants to exit
        if user_message.lower() in ["exit", "quit", "bye"]:
            print(
                "\nThank you for chatting with your health coach! Stay healthy and keep moving!"
            )
            break

        # Create message object for the current user message
        current_message = {"role": "user", "content": user_message}

        # Search memory for relevant context
        print("\n🔍 Searching memory for relevant context...")
        memory_results = await search_memory(user_id, user_message)

        if memory_results and len(memory_results) > 0:
            print(f"📚 Found {len(memory_results)} relevant memories:")
            for idx, item in enumerate(
                memory_results[:2]
            ):  # Show top 3 memories for brevity
                if "message" in item and "content" in item["message"]:
                    print(f"  • Memory {idx+1}: {item['message']['content'][:100]}...")
        else:
            print("📭 No relevant memories found.")

        # Format memories for inclusion in the prompt
        formatted_memories = ""
        if memory_results and len(memory_results) > 0:
            for idx, item in enumerate(memory_results):
                if "message" in item and "content" in item["message"]:
                    role = item["message"].get("role", "unknown")
                    content = item["message"].get("content", "")
                    formatted_memories += f"{role}: {content}\n\n"

        # Generate response with context
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=f"""
User message: {user_message}

Relevant past information from memory:
{formatted_memories}

Respond to the user with personalized health coaching advice based on their message and any relevant information from memory.
Don't mention the memory system directly to the user - they should just experience personalized advice.
"""
            ),
        ]

        # Get response from model
        print("\n⏳ Thinking...")
        response = llm.invoke(messages)
        coach_response = response.content

        # Clean up response to remove any artifacts
        coach_response = clean_response(coach_response)

        # Display response to user
        print(f"\nCoach: {coach_response}")

        # Add current exchange to conversation history for the current session
        conversation_history.append(current_message)
        conversation_history.append({"role": "assistant", "content": coach_response})

        # Store the current exchange in Mem0
        print("\n💾 Storing conversation in memory...")
        storage_success = await store_memory(
            user_id, [current_message, {"role": "assistant", "content": coach_response}]
        )

        if storage_success:
            print("✅ Conversation stored successfully!")
        else:
            print("⚠️ Failed to store conversation.")


def clean_response(response):
    """Clean up the response to remove any system artifacts"""
    # Remove common formatting markers
    for marker in ["RESPONSE:", "USER RESPONSE:", "COACH:"]:
        if response.startswith(marker):
            response = response[len(marker) :].strip()

    return response.strip()


if __name__ == "__main__":
    asyncio.run(run_interactive_health_coach())
