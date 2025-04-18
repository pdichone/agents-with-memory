"""
Interactive Health Coach with Memory using LangMem

This implementation allows users to have a real-time conversation with the health coach via console.
"""

import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langmem import create_manage_memory_tool, create_search_memory_tool
from langgraph.store.memory import InMemoryStore

# Load environment variables
load_dotenv()

# Set up OpenAI client
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.2)
# Set up memory store
store = InMemoryStore()

# Create memory tools
manage_memory = create_manage_memory_tool(
    namespace=("health_coach", "user123", "memories"), store=store
)

search_memory = create_search_memory_tool(
    namespace=("health_coach", "user123", "memories"), store=store
)


# System prompt for the health coach
SYSTEM_PROMPT = """
You are a personal health coach with access to memory tools. Your goal is to help users achieve their fitness goals
by providing personalized advice and tracking their progress over time.

You have two special abilities:
1. You can search past memories about the user to provide personalized advice
2. You can store new information about the user for future reference

When appropriate, you will:
- Search memory for user context (diet, fitness level, goals, etc.)
- Store important new information (changes in weight, new goals, etc.)
- Reference past context to provide continuity

Always be supportive, encouraging, and provide actionable advice.
"""


def run_interactive_health_coach():
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

    # Initialize conversation history
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

        # Search memory for relevant context
        memory_results = "[]"
        try:
            # Create a search query based on user input and conversation history
            search_query = user_message
            if len(search_query) > 100:
                search_query = search_query[:100]  # Limit query length

            print("\n🔍 Searching memory for relevant context...")
            memory_results = search_memory.invoke({"query": search_query})

            # Parse and pretty-print memory results for debugging
            if memory_results and memory_results != "[]":
                try:
                    memory_items = json.loads(memory_results)
                    if memory_items:
                        print(f"📚 Found {len(memory_items)} relevant memories:")
                        for idx, item in enumerate(memory_items):
                            if "value" in item and "content" in item["value"]:
                                print(
                                    f"  • Memory {idx+1}: {item['value']['content'][:100]}..."
                                )
                    else:
                        print("📭 No relevant memories found.")
                except:
                    print(f"⚠️ Memory format: {memory_results[:100]}...")
            else:
                print("📭 No previous memories found.")
        except Exception as e:
            print(f"⚠️ Memory search error: {str(e)}")

        # Add to conversation history
        conversation_history.append(user_message)

        # Create context for the model
        context = (
            "\n".join(conversation_history[-3:])
            if len(conversation_history) > 1
            else user_message
        )

        # Generate response with context
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=f"""
User message: {user_message}

Previous conversation context: {context}

Memory search results: {memory_results}

Respond to the user with personalized health coaching advice based on their message and any relevant information from memory.

Also, identify if there's new important information about the user that should be stored in memory.
Don't mention the memory system directly to the user - they should just experience personalized advice.
"""
            ),
        ]

        # Get response from model
        print("\n⏳ Thinking...")
        response = llm.invoke(messages)

        # Process response to extract potential memory updates
        coach_response = response.content
        memory_to_store = None

        # Check if the response contains a clear indication of memory content
        if (
            "MEMORY:" in coach_response
            or "STORE IN MEMORY:" in coach_response
            or "REMEMBER:" in coach_response
        ):
            try:
                # Try to extract memory section - assumes model might format its response with a designated memory section
                for marker in ["MEMORY:", "STORE IN MEMORY:", "REMEMBER:"]:
                    if marker in coach_response:
                        parts = coach_response.split(marker, 1)
                        if len(parts) > 1:
                            # Extract the memory part and clean up the response
                            memory_section = parts[1].split("\n\n", 1)[0].strip()
                            coach_response = coach_response.replace(
                                marker + memory_section, ""
                            ).strip()
                            memory_to_store = memory_section
                            break
            except:
                # If extraction fails, create a general memory
                memory_to_store = f"User message: {user_message}"
        else:
            # Create an automatic memory from this interaction
            memory_to_store = extract_key_information(user_message)

        # Store memory if we have something to store
        if memory_to_store:
            try:
                print("\n💾 Storing new information in memory...")
                manage_memory.invoke({"content": memory_to_store})
                print(
                    f"✅ Stored: {memory_to_store[:50]}..."
                    if len(memory_to_store) > 50
                    else f"✅ Stored: {memory_to_store}"
                )
            except Exception as e:
                print(f"⚠️ Memory storage error: {str(e)}")

        # Clean up response to remove any artifacts and display to user
        coach_response = clean_response(coach_response)
        print(f"\nCoach: {coach_response}")


def extract_key_information(user_message):
    """
    Automatically extract key health/fitness information from user messages
    when no explicit memory instruction is found
    """
    # Simple extraction for demonstration - in a real system, use NLP for better extraction
    memory = f"User shared: {user_message}"
    return memory


def clean_response(response):
    """Clean up the response to remove any system artifacts"""
    # Remove common formatting markers
    for marker in ["RESPONSE:", "USER RESPONSE:", "COACH:"]:
        if response.startswith(marker):
            response = response[len(marker) :].strip()

    # Remove any remaining memory markers and their content
    for marker in ["MEMORY:", "STORE IN MEMORY:", "REMEMBER:"]:
        if marker in response:
            parts = response.split(marker, 1)
            if len(parts) > 1:
                second_part = parts[1].split("\n\n", 1)
                if len(second_part) > 1:
                    response = parts[0] + second_part[1]
                else:
                    response = parts[0]

    return response.strip()


if __name__ == "__main__":
    run_interactive_health_coach()
