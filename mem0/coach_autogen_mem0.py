"""
Interactive Health Coach with Memory using AutoGen and Mem0

This implementation combines Mem0's Memory class with AutoGen's ConversableAgent.
"""

import os
import json
import re
import uuid
from dotenv import load_dotenv
from mem0 import Memory
import autogen
from autogen import ConversableAgent

# Load environment variables
load_dotenv()

# Suppress Pydantic deprecation warnings from AutoGen
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, module="pydantic")
warnings.filterwarnings("ignore", message=".*copy.*", module="autogen")

# Configure Mem0 memory system
config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4o-mini"  # You can change this to gpt-4o-mini if preferred
        },
    }
}

# Initialize memory client
memory = Memory.from_config(config) # saving memory to local file

# User profile cache to ensure consistent information
user_profile = {}

# System prompt for the health coach agent
SYSTEM_PROMPT = """
You are a personal health coach with access to memory. Your goal is to help users achieve their fitness goals
by providing personalized advice and tracking their progress over time.

You have information about the user's past interactions that you can use to provide personalized advice.

Always be supportive, encouraging, and provide actionable advice.

Important instructions:
1. Keep your responses concise and focused on health coaching
2. Provide specific, actionable advice when possible
3. Be empathetic and understanding of the user's health journey
4. Never mention the memory system directly to the user
"""


def extract_user_info(message, user_id):
    """Extract user information from message and update profile cache"""
    # Initialize user profile if needed
    if user_id not in user_profile:
        user_profile[user_id] = {
            "name": None,
            "age": None,
            "weight": None,
            "height": None,
            "goals": None,
            "dietary_preferences": None,
        }

    profile = user_profile[user_id]
    updated = False

    # Extract name
    name_patterns = [
        r"(?:my name is|i'm called|i am called|call me) ([A-Za-z]+(?:\s[A-Za-z]+)*)",
        r"([A-Za-z]+(?:\s[A-Za-z]+)*) is my name",
        r"name(?:'s|\s+is|\s*:\s*)([A-Za-z]+(?:\s[A-Za-z]+)*)",
    ]

    for pattern in name_patterns:
        matches = re.search(pattern, message, re.IGNORECASE)
        if matches and matches.group(1).strip().lower() not in ["is", "my", "the", "a"]:
            # Get the name and make sure it's properly capitalized
            raw_name = matches.group(1).strip()
            # Capitalize each word in the name
            profile["name"] = " ".join(word.capitalize() for word in raw_name.split())
            print(f"✅ Extracted name: '{profile['name']}'")
            updated = True
            break

    # Extract age
    age_patterns = [
        r"(?:i'm|i am) (\d+)(?: years old)?",
        r"(\d+) years old",
        r"age(?:'s|\s+is|\s*:\s*)(\d+)",
    ]

    for pattern in age_patterns:
        matches = re.search(pattern, message, re.IGNORECASE)
        if matches:
            profile["age"] = matches.group(1)
            print(f"✅ Extracted age: {profile['age']}")
            updated = True
            break

    # Extract weight
    weight_patterns = [
        r"(?:i weigh|my weight is) (\d+\.?\d*) ?(?:kg|kilos|pounds|lbs)",
        r"weight(?:'s|\s+is|\s*:\s*)(\d+\.?\d*) ?(?:kg|kilos|pounds|lbs)",
    ]

    for pattern in weight_patterns:
        matches = re.search(pattern, message, re.IGNORECASE)
        if matches:
            profile["weight"] = matches.group(1)
            unit = "kg" if "kg" in message or "kilo" in message else "lbs"
            profile["weight_unit"] = unit
            print(f"✅ Extracted weight: {profile['weight']} {unit}")
            updated = True
            break

    # Store profile information as a memory if updated
    if updated:
        store_profile_memory(user_id)

    return updated


def store_profile_memory(user_id):
    """Store profile information as a dedicated memory"""
    profile = user_profile.get(user_id, {})
    if not profile:
        return

    # Create a clean profile summary
    clean_profile = {k: v for k, v in profile.items() if v is not None}
    if not clean_profile:
        return

    profile_text = "USER PROFILE:\n"
    for key, value in clean_profile.items():
        profile_text += f"{key.replace('_', ' ').capitalize()}: {value}\n"

    # Create a special system message for profile storage
    profile_message = [{"role": "system", "content": profile_text}]

    # Store in memory with metadata
    memory.add(
        profile_message,
        user_id=user_id,
        metadata={"type": "user_profile", "priority": "high"},
    )
    print(f"✅ Stored user profile: {profile_text}")


def get_profile_summary(user_id):
    """Get a formatted summary of the user profile"""
    profile = user_profile.get(user_id, {})

    # Create a clean profile with only non-None values
    clean_profile = {k: v for k, v in profile.items() if v is not None}

    if not clean_profile:
        return "No profile information available."

    summary = "USER PROFILE:\n"
    for key, value in clean_profile.items():
        summary += f"- {key.replace('_', ' ').capitalize()}: {value}\n"

    return summary


class HealthCoachApp:
    def __init__(self, user_id=None):
        """Initialize the health coach with a unique user ID"""
        # Generate a unique user ID if none is provided
        self.user_id = user_id if user_id else str(uuid.uuid4())

        # Initialize the conversable agent
        self.agent = ConversableAgent(
            name="health_coach",
            system_message=SYSTEM_PROMPT,
            llm_config={
                "config_list": [
                    {"model": "gpt-4o-mini", "api_key": os.getenv("OPENAI_API_KEY")}
                ]
            },
            code_execution_config=False,
            function_map=None,
            human_input_mode="NEVER",
        )

        # Initialize conversation history
        self.conversation = []

    def process_message(self, user_message):
        """Process a user message and get a response from the health coach"""
        # First try to extract any user information
        extract_user_info(user_message, self.user_id)

        # Add user message to conversation history
        self.conversation.append({"role": "user", "content": user_message})

        # Get context-aware response using improved memory retrieval
        coach_response = self.get_context_aware_response(user_message)

        # Add response to conversation history
        self.conversation.append({"role": "assistant", "content": coach_response})

        # Store the conversation in memory
        print("\n💾 Storing conversation in memory...")
        memory.add(
            [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": coach_response},
            ],
            user_id=self.user_id,
        )

        return coach_response

    def get_context_aware_response(self, question):
        """Get a context-aware response using memory retrieval"""
        print("\n🔍 Searching memory for relevant context...")

        # Retrieve relevant memories with direct access to profile for specific queries
        profile = user_profile.get(self.user_id, {})
        direct_answer = ""

        # Handle direct profile questions with local cache
        if any(
            term in question.lower() for term in ["my name", "who am i"]
        ) and profile.get("name"):
            direct_answer = f"Your name is {profile['name']}. "
        elif any(
            term in question.lower() for term in ["my age", "how old"]
        ) and profile.get("age"):
            direct_answer = f"You are {profile['age']} years old. "
        elif any(
            term in question.lower() for term in ["my weight", "how much do i weigh"]
        ) and profile.get("weight"):
            unit = profile.get("weight_unit", "units")
            direct_answer = f"Your weight is {profile['weight']} {unit}. "

        # Get relevant memories from Mem0
        memory_results = memory.search(question, user_id=self.user_id, limit=5)

        # Format memories for inclusion in the prompt
        if memory_results and memory_results.get("results"):
            memories = memory_results["results"]
            print(f"📚 Found {len(memories)} relevant memories")
            context = "\n".join([m["memory"] for m in memories])
        else:
            print("📭 No relevant memories found")
            context = "No relevant memories found."

        # Get user profile summary
        profile_summary = get_profile_summary(self.user_id)

        # Create prompt with context for the agent using the improved format
        prompt = f"""Answer the user question considering the following information:

User Profile:
{profile_summary}

Previous interactions:
{context}

Recent conversation:
{json.dumps(self.conversation[-2:] if len(self.conversation) > 2 else self.conversation, indent=2)}

Question: {question}

When responding:
1. Use the user's name if available
2. Reference their profile information when relevant
3. Provide personalized health coaching advice
"""

        print("\n⏳ Thinking...")
        reply = self.agent.generate_reply(
            messages=[{"content": prompt, "role": "user"}]
        )

        # If we have a direct answer from the profile, ensure the information is included correctly
        if direct_answer and direct_answer.lower() not in reply.lower():
            reply = direct_answer + reply

        return reply


def main():
    """Interactive console interface for the health coach"""
    print("\n" + "=" * 80)
    print("🏃‍♂️ WELCOME TO YOUR PERSONAL HEALTH COACH 🏋️‍♀️")
    print("=" * 80)
    print(
        "I'm your AI health coach with memory. I can provide personalized fitness advice"
    )
    print("and will remember details about your fitness journey between conversations.")
    print("\nType 'exit', 'quit', or 'bye' to end our conversation.\n")

    # Create health coach with a unique user ID
    coach = HealthCoachApp()
    print(f"Session ID: {coach.user_id}")

    while True:
        # Get user input
        user_message = input("\nYou: ").strip()

        # Check if user wants to exit
        if user_message.lower() in ["exit", "quit", "bye"]:
            print(
                "\nThank you for chatting with your health coach! Stay healthy and keep moving!"
            )
            break

        # Process message and get response
        coach_response = coach.process_message(user_message)

        # Display response
        print(f"\nCoach: {coach_response}")

        # Display current profile for reference
        profile = user_profile.get(coach.user_id, {})
        clean_profile = {k: v for k, v in profile.items() if v is not None}
        if clean_profile:
            print("\n[DEBUG] Current profile information:")
            for k, v in clean_profile.items():
                print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()


# """
# Interactive Health Coach with Memory using Mem0 (simplified version)

# This implementation uses Mem0's Memory class for simple memory management.
# """

# import os
# import json
# import re
# import uuid
# from dotenv import load_dotenv
# from openai import OpenAI
# from mem0 import Memory

# # Load environment variables
# load_dotenv()

# # Configure the memory system
# config = {
#     "llm": {
#         "provider": "openai",
#         "config": {
#             "model": "gpt-4-turbo"  # You can change this to gpt-4o-mini if preferred
#         },
#     }
# }

# # Initialize clients
# openai_client = OpenAI()
# memory = Memory.from_config(config)

# # User profile cache to ensure consistent information
# user_profile = {}


# def extract_user_info(message, user_id):
#     """Extract user information from message and update profile cache"""
#     # Initialize user profile if needed
#     if user_id not in user_profile:
#         user_profile[user_id] = {
#             "name": None,
#             "age": None,
#             "weight": None,
#             "height": None,
#             "goals": None,
#             "dietary_preferences": None,
#         }

#     profile = user_profile[user_id]
#     updated = False

#     # Extract name
#     name_patterns = [
#         r"(?:my name is|i'm called|i am called|call me) ([A-Za-z]+(?:\s[A-Za-z]+)*)",
#         r"([A-Za-z]+(?:\s[A-Za-z]+)*) is my name",
#         r"name(?:'s|\s+is|\s*:\s*)([A-Za-z]+(?:\s[A-Za-z]+)*)",
#     ]

#     for pattern in name_patterns:
#         matches = re.search(pattern, message, re.IGNORECASE)
#         if matches and matches.group(1).strip().lower() not in ["is", "my", "the", "a"]:
#             # Get the name and make sure it's properly capitalized
#             raw_name = matches.group(1).strip()
#             # Capitalize each word in the name
#             profile["name"] = " ".join(word.capitalize() for word in raw_name.split())
#             print(f"✅ Extracted name: '{profile['name']}'")
#             updated = True
#             break

#     # Extract age
#     age_patterns = [
#         r"(?:i'm|i am) (\d+)(?: years old)?",
#         r"(\d+) years old",
#         r"age(?:'s|\s+is|\s*:\s*)(\d+)",
#     ]

#     for pattern in age_patterns:
#         matches = re.search(pattern, message, re.IGNORECASE)
#         if matches:
#             profile["age"] = matches.group(1)
#             print(f"✅ Extracted age: {profile['age']}")
#             updated = True
#             break

#     # Extract weight
#     weight_patterns = [
#         r"(?:i weigh|my weight is) (\d+\.?\d*) ?(?:kg|kilos|pounds|lbs)",
#         r"weight(?:'s|\s+is|\s*:\s*)(\d+\.?\d*) ?(?:kg|kilos|pounds|lbs)",
#     ]

#     for pattern in weight_patterns:
#         matches = re.search(pattern, message, re.IGNORECASE)
#         if matches:
#             profile["weight"] = matches.group(1)
#             unit = "kg" if "kg" in message or "kilo" in message else "lbs"
#             profile["weight_unit"] = unit
#             print(f"✅ Extracted weight: {profile['weight']} {unit}")
#             updated = True
#             break

#     # Store profile information as a memory if updated
#     if updated:
#         store_profile_memory(user_id)

#     return updated


# def store_profile_memory(user_id):
#     """Store profile information as a dedicated memory"""
#     profile = user_profile.get(user_id, {})
#     if not profile:
#         return

#     # Create a clean profile summary
#     clean_profile = {k: v for k, v in profile.items() if v is not None}
#     if not clean_profile:
#         return

#     profile_text = "USER PROFILE:\n"
#     for key, value in clean_profile.items():
#         profile_text += f"{key.replace('_', ' ').capitalize()}: {value}\n"

#     # Create a special system message for profile storage
#     profile_message = [{"role": "system", "content": profile_text}]

#     # Store in memory with metadata
#     memory.add(
#         profile_message,
#         user_id=user_id,
#         metadata={"type": "user_profile", "priority": "high"},
#     )
#     print(f"✅ Stored user profile: {profile_text}")


# def get_profile_summary(user_id):
#     """Get a formatted summary of the user profile"""
#     profile = user_profile.get(user_id, {})

#     # Create a clean profile with only non-None values
#     clean_profile = {k: v for k, v in profile.items() if v is not None}

#     if not clean_profile:
#         return "No profile information available."

#     summary = "USER PROFILE:\n"
#     for key, value in clean_profile.items():
#         summary += f"- {key.replace('_', ' ').capitalize()}: {value}\n"

#     return summary


# def chat_with_health_coach(message, user_id="default_user"):
#     """Process a user message and get a response from the health coach"""
#     # First try to extract any user information
#     extract_user_info(message, user_id)

#     # Get user profile summary
#     profile_summary = get_profile_summary(user_id)

#     # Prepare memory search based on the message
#     print("\n🔍 Searching memory for relevant context...")

#     # Add specific search terms for profile queries
#     query = message
#     if any(term in message.lower() for term in ["my name", "who am i"]):
#         query = "USER PROFILE name"
#     elif any(term in message.lower() for term in ["my age", "how old"]):
#         query = "USER PROFILE age"
#     elif any(term in message.lower() for term in ["my weight", "how much do i weigh"]):
#         query = "USER PROFILE weight"

#     # Retrieve relevant memories
#     memory_results = memory.search(query=query, user_id=user_id, limit=5)

#     if memory_results and memory_results.get("results"):
#         memories = memory_results["results"]
#         print(f"📚 Found {len(memories)} relevant memories")
#         memories_str = "\n".join(f"- {entry['memory']}" for entry in memories)
#     else:
#         print("📭 No relevant memories found")
#         memories_str = "No relevant memories found."

#     # Check if this is a direct profile query
#     direct_answer = ""
#     profile = user_profile.get(user_id, {})

#     if "name" in message.lower() and profile.get("name"):
#         direct_answer = f"Your name is {profile['name']}. "
#     elif "age" in message.lower() and profile.get("age"):
#         direct_answer = f"You are {profile['age']} years old. "
#     elif "weight" in message.lower() and profile.get("weight"):
#         unit = profile.get("weight_unit", "units")
#         direct_answer = f"Your weight is {profile['weight']} {unit}. "

#     # Generate system prompt with memory context and profile information
#     system_prompt = f"""
# You are a supportive health coach who helps users achieve their fitness goals.
# Always be encouraging, positive, and provide personalized advice based on user information.

# {profile_summary}

# Relevant past information:
# {memories_str}

# If the user asks about information that's in their profile, be sure to answer accurately.
# If information isn't in the profile or memories, politely state you don't have that information yet.
# """

#     print("\n⏳ Thinking...")

#     # Generate response
#     messages = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": message},
#     ]

#     response = openai_client.chat.completions.create(
#         model=config["llm"]["config"]["model"], messages=messages
#     )

#     assistant_response = response.choices[0].message.content

#     # If we have a direct answer from the profile, ensure the information is included correctly
#     if direct_answer and direct_answer.lower() not in assistant_response.lower():
#         assistant_response = direct_answer + assistant_response

#     # Store conversation in memory
#     print("\n💾 Storing conversation in memory...")
#     conversation = [
#         {"role": "user", "content": message},
#         {"role": "assistant", "content": assistant_response},
#     ]

#     memory.add(conversation, user_id=user_id)

#     return assistant_response


# def main():
#     """Interactive console interface for the health coach"""
#     print("\n" + "=" * 80)
#     print("🏃‍♂️ WELCOME TO YOUR PERSONAL HEALTH COACH 🏋️‍♀️")
#     print("=" * 80)
#     print(
#         "I'm your AI health coach with memory. I can provide personalized fitness advice"
#     )
#     print("and will remember details about your fitness journey between conversations.")
#     print("\nType 'exit', 'quit', or 'bye' to end our conversation.\n")

#     # Generate a unique user ID for this session
#     user_id = str(uuid.uuid4())
#     print(f"Session ID: {user_id}")

#     while True:
#         # Get user input
#         user_message = input("\nYou: ").strip()

#         # Check if user wants to exit
#         if user_message.lower() in ["exit", "quit", "bye"]:
#             print(
#                 "\nThank you for chatting with your health coach! Stay healthy and keep moving!"
#             )
#             break

#         # Process message and get response
#         coach_response = chat_with_health_coach(user_message, user_id)

#         # Display response
#         print(f"\nCoach: {coach_response}")

#         # Display current profile for reference
#         profile = user_profile.get(user_id, {})
#         clean_profile = {k: v for k, v in profile.items() if v is not None}
#         if clean_profile:
#             print("\n[DEBUG] Current profile information:")
#             for k, v in clean_profile.items():
#                 print(f"  - {k}: {v}")


# if __name__ == "__main__":
#     main()
