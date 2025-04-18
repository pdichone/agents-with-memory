import os
from dotenv import load_dotenv
from openai import OpenAI
from mem0 import Memory
import supabase
from datetime import datetime
import uuid
import requests

# CREATE TABLE chat_memories (
#     id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
#     user_id TEXT NOT NULL,
#     user_message TEXT NOT NULL,
#     assistant_response TEXT NOT NULL,
#     timestamp TIMESTAMPTZ NOT NULL,
#     session_id TEXT NOT NULL
# );
# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase_url = os.environ.get("SUPABASE_URL", "")
supabase_key = os.environ.get("SUPABASE_KEY", "")
supabase_client = supabase.create_client(supabase_url, supabase_key)

# Use a stable model that's guaranteed to exist
model = os.getenv("MODEL_CHOICE", "gpt-3.5-turbo")  # Fallback to a known model
print(f"Using model===> {model}")

# Initialize OpenAI client
openai_client = OpenAI()

# Configuration for Memory
config = {
    "llm": {"provider": "openai", "config": {"model": model}},
    "vector_store": {
        "provider": "supabase",
        "config": {
            "connection_string": os.environ.get("DATABASE_URL"),
            "collection_name": "chat_memories",
        },
    },
}

# Initialize Memory
try:
    memory = Memory.from_config(config)
    print("Memory initialized successfully")
except Exception as e:
    print(f"Error initializing memory: {str(e)}")


def chat_with_memories(message: str, user_id: str = "default_user") -> str:
    # Generate a session ID if not provided
    session_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().isoformat()

    # Retrieve relevant memories
    try:
        relevant_memories = memory.search(query=message, user_id=user_id, limit=3)
        formatted_memories = ""
        if (
            relevant_memories
            and "results" in relevant_memories
            and len(relevant_memories["results"]) > 0
        ):
            formatted_memories = "\n".join(
                f"- {entry['memory']}" for entry in relevant_memories["results"]
            )
            print(
                f"\nRetrieved {len(relevant_memories['results'])} memories for user {user_id}"
            )
    except Exception as e:
        print(f"Error retrieving memories: {str(e)}")
        formatted_memories = ""

    # Generate Assistant response
    system_prompt = f"""You are a helpful AI assistant. Answer the question based on the query and memories.
    
User's Relevant Memories:
{formatted_memories}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message},
    ]

    try:
        response = openai_client.chat.completions.create(model=model, messages=messages)
        assistant_response = response.choices[0].message.content
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        assistant_response = (
            "I'm sorry, I encountered an error while processing your request."
        )

    # Create memory entry for Supabase
    chat_memory = {
        "user_id": user_id,
        "user_message": message,
        "assistant_response": assistant_response,
        "timestamp": timestamp,
        "session_id": session_id,
    }

    # Save directly to Supabase
    try:
        table = supabase_client.table("chat_memories")
        result = table.insert([chat_memory]).execute()
        print(f"Conversation saved to Supabase for user: {user_id}")
    except Exception as e:
        print(f"Error saving to Supabase: {str(e)}")
        try:
            url = f"{supabase_url}/rest/v1/chat_memories"
            headers = {
                "apikey": supabase_key,
                "Authorization": f"Bearer {supabase_key}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal",
            }
            response = requests.post(url, headers=headers, json=chat_memory)
            print(f"Direct REST API response: {response.status_code}")
            if response.status_code >= 400:
                print(f"Response content: {response.text}")
        except Exception as alt_e:
            print(f"Alternative approach also failed: {str(alt_e)}")

    # Add the conversation to memory (for vector search)
    try:
        memory.add(
            [
                {"role": "user", "content": message},
                {"role": "assistant", "content": assistant_response},
            ],
            user_id=user_id,
        )
        print(f"Conversation added to vector store for user: {user_id}")
    except Exception as e:
        print(f"Error saving to vector store: {str(e)}")

    return assistant_response


def get_user_chat_history(user_id: str, limit: int = 5):
    """Retrieve the most recent chat history for a user"""
    try:
        result = (
            supabase_client.table("chat_memories")
            .select("*")
            .eq("user_id", user_id)
            .order("timestamp", desc=True)
            .limit(limit)
            .execute()
        )
        if result and result.data:
            return result.data
        return []
    except Exception as e:
        print(f"Error retrieving chat history: {str(e)}")
        return []


def main():
    print("Chat with Memory (type 'exit' to quit, or use commands below)")
    print("Commands:")
    print("  user:<name> - Switch to a different user")
    print("  history - Show your recent chat history")
    print("Using model:", model)

    current_user = "default_user"

    while True:
        user_input = input(f"You ({current_user}): ").strip()

        # Handle commands
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        elif user_input.lower().startswith("user:"):
            current_user = user_input[5:].strip()
            print(f"Switched to user: {current_user}")
            continue

        elif user_input.lower() == "history":
            history = get_user_chat_history(current_user)
            if history:
                print("\n--- Recent Chat History ---")
                for i, entry in enumerate(history):
                    print(f"{i+1}. You: {entry['user_message']}")
                    print(f"   AI: {entry['assistant_response']}")
                    print(f"   Time: {entry['timestamp']}")
                    print()
            else:
                print("No chat history found.")
            continue

        # Process normal chat input
        response = chat_with_memories(
            user_input, user_id=current_user
        )  # Fixed typo here
        print(f"AI: {response}")
        print("=====================================")


if __name__ == "__main__":
    main()
