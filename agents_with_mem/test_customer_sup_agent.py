"""
Memory Testing Script for Customer Support Agent

This script demonstrates the memory capabilities of the customer support agent
by simulating a sequence of interactions with the same customer.

Run this after setting up the customer_support_agent.py file.
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv
from typing import Literal
from typing_extensions import TypedDict, Annotated
from langchain_core.tools import tool
from langgraph.store.memory import InMemoryStore
from langgraph.graph import add_messages, StateGraph, START, END
from langgraph.types import Command

# Load environment variables
_ = load_dotenv()


# Import the core agent components
# In a real application, you'd import these from your main agent file
# For this demo, we'll recreate the minimal necessary components
class State(TypedDict):
    inquiry_input: dict
    messages: Annotated[list, add_messages]


# Set up enhanced memory store with vector embeddings
memory_store = InMemoryStore(index={"embed": "openai:text-embedding-3-small"})

# Create enhanced memory tools with debugging
MEMORY_DB = {}  # Simple dictionary to track memories for demonstration


@tool
def manage_memory(content: str = None, action: str = "create", id: str = None) -> str:
    """Create, update, or delete persistent MEMORIES for this customer."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n🧠 [MEMORY OPERATION - {action.upper()}]")
    print(f"📝 Content: {content}")

    if action == "create" and content:
        memory_id = f"mem_{abs(hash(content))}"[:12]
        MEMORY_DB[memory_id] = {
            "content": content,
            "created_at": timestamp,
            "updated_at": timestamp,
        }
        print(f"✅ Created memory: {memory_id}")
        return f"created memory {memory_id}"

    elif action == "update" and id and content:
        if id in MEMORY_DB:
            MEMORY_DB[id]["content"] = content
            MEMORY_DB[id]["updated_at"] = timestamp
            print(f"✅ Updated memory: {id}")
            return f"updated memory {id}"
        else:
            print(f"❌ Failed to update: Memory {id} not found")
            return f"error: memory {id} not found"

    elif action == "delete" and id:
        if id in MEMORY_DB:
            del MEMORY_DB[id]
            print(f"✅ Deleted memory: {id}")
            return f"deleted memory {id}"
        else:
            print(f"❌ Failed to delete: Memory {id} not found")
            return f"error: memory {id} not found"

    return "Memory operation failed. Check parameters."


@tool
def search_memory(
    query: str, limit: int = 10, offset: int = 0, filter: dict = None
) -> str:
    """Search memories for information relevant to the current context."""
    print(f"\n🔍 [MEMORY SEARCH]")
    print(f"🔎 Query: '{query}'")

    # Simplified memory search based on basic text matching
    results = []
    query_lower = query.lower()

    for mem_id, data in MEMORY_DB.items():
        content = data["content"].lower()
        # Simple relevance score based on word matching
        if any(term in content for term in query_lower.split()):
            score = sum(term in content for term in query_lower.split()) / len(
                query_lower.split()
            )
            results.append(
                {
                    "id": mem_id,
                    "value": {"content": data["content"]},
                    "created_at": data["created_at"],
                    "updated_at": data["updated_at"],
                    "score": score,
                }
            )

    # Sort by relevance score
    results = sorted(results, key=lambda x: x["score"], reverse=True)[:limit]

    print(f"📊 Found {len(results)} relevant memories")
    for r in results:
        print(f"  • [{r['id']}] - Score: {r['score']:.2f}")
        print(
            f"    '{r['value']['content'][:100]}{'...' if len(r['value']['content']) > 100 else ''}'"
        )

    return json.dumps(results)


# Simplified triage and response functions for the memory test
def handle_inquiry(inquiry, history=None):
    """
    Process a customer inquiry and demonstrate memory operations.
    This is a simplified version of what would normally be handled by the agent graph.
    """
    if history is None:
        history = []

    customer_email = inquiry["author"]
    subject = inquiry["subject"]
    message = inquiry["message_thread"]

    print("\n" + "=" * 80)
    print(f"📩 PROCESSING INQUIRY: {subject}")
    print(f"👤 From: {customer_email}")
    print("-" * 80)
    print(message)
    print("=" * 80)

    # Step 1: Search for customer context
    print("\n📚 STEP 1: Retrieving customer context from memory")
    search_result = search_memory.invoke({"query": customer_email})

    # Step 2: Process the inquiry with context
    print("\n📝 STEP 2: Processing the inquiry with available context")
    # In a real implementation, this would be where the agent generates a response

    # Step 3: Update memory with new information
    print("\n💾 STEP 3: Updating memory with new information")
    if "login issue" in message.lower() or "login issues" in message.lower():
        manage_memory.invoke(
            {
                "content": f"Customer {customer_email} reported login issues with the mobile app: '{message[:100]}...'"
            }
        )
    elif "payment" in message.lower() or "billing" in message.lower():
        manage_memory.invoke(
            {
                "content": f"Customer {customer_email} had billing/payment question: '{message[:100]}...'"
            }
        )
    elif "feature" in message.lower() or "suggestion" in message.lower():
        manage_memory.invoke(
            {
                "content": f"Customer {customer_email} suggested new feature: '{message[:100]}...'"
            }
        )
    else:
        manage_memory.invoke(
            {
                "content": f"Interaction with {customer_email} about {subject}: '{message[:100]}...'"
            }
        )

    # Record this interaction in history
    history.append(
        {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "inquiry": inquiry}
    )

    return history


def display_memory_contents():
    """Display all memories stored in the system."""
    print("\n" + "=" * 80)
    print("📚 CURRENT MEMORY CONTENTS")
    print("=" * 80)

    if not MEMORY_DB:
        print("No memories stored yet.")
        return

    for mem_id, data in MEMORY_DB.items():
        print(f"ID: {mem_id}")
        print(f"Content: {data['content']}")
        print(f"Created: {data['created_at']}")
        print(f"Updated: {data['updated_at']}")
        print("-" * 80)


def run_customer_journey_simulation():
    """
    Simulate a complete customer journey with multiple interactions
    to demonstrate how memory enhances the agent's capabilities.
    """
    print("\n" + "=" * 80)
    print("🔄 CUSTOMER JOURNEY SIMULATION WITH MEMORY")
    print("=" * 80)

    # Customer history will track all interactions
    customer_history = []

    # Day 1: Initial inquiry about login issues
    day1_inquiry = {
        "author": "john.smith@example.com",
        "to": "support@yourcompany.com",
        "subject": "Cannot login to mobile app",
        "message_thread": """Hello Support,

I've been trying to login to your mobile app since yesterday but keep getting an "Authentication Failed" error. 
I can login to the website just fine with the same credentials.

Thanks,
John Smith""",
    }

    print("\n\n📅 DAY 1: Initial Contact")
    customer_history = handle_inquiry(day1_inquiry, customer_history)

    # Show memory after first interaction
    display_memory_contents()

    # Day 3: Follow-up about the same issue
    day3_inquiry = {
        "author": "john.smith@example.com",
        "to": "support@yourcompany.com",
        "subject": "Re: Cannot login to mobile app",
        "message_thread": """Hi again,

I tried the app reinstall you suggested but I'm still having the login issue.
Could this be related to the fact that I recently changed my password?

John""",
    }

    print("\n\n📅 DAY 3: Follow-up on Same Issue")
    customer_history = handle_inquiry(day3_inquiry, customer_history)

    # Show memory after second interaction
    display_memory_contents()

    # Day 10: New issue about billing
    day10_inquiry = {
        "author": "john.smith@example.com",
        "to": "support@yourcompany.com",
        "subject": "Question about my recent bill",
        "message_thread": """Hello Support Team,

I just noticed I was charged twice for my subscription this month.
Can you please look into this and process a refund for the duplicate charge?

Best regards,
John Smith""",
    }

    print("\n\n📅 DAY 10: New Issue (Billing)")
    customer_history = handle_inquiry(day10_inquiry, customer_history)

    # Show memory after third interaction
    display_memory_contents()

    # Day 45: Feature suggestion
    day45_inquiry = {
        "author": "john.smith@example.com",
        "to": "support@yourcompany.com",
        "subject": "Feature suggestion",
        "message_thread": """Hi there,

Now that my login and billing issues are resolved, I've been using the app regularly. 
I'd love to see a dark mode option added in a future update.

Thanks for all your help,
John""",
    }

    print("\n\n📅 DAY 45: Feature Suggestion")
    customer_history = handle_inquiry(day45_inquiry, customer_history)

    # Show final memory state
    display_memory_contents()

    # Search for this customer's login issues
    print("\n\n🔍 SEARCHING FOR CUSTOMER'S LOGIN ISSUES")
    search_memory.invoke({"query": "john.smith@example.com login"})

    # Search for this customer's billing issues
    print("\n\n🔍 SEARCHING FOR CUSTOMER'S BILLING ISSUES")
    search_memory.invoke({"query": "john.smith@example.com billing"})

    # Search for feature suggestions
    print("\n\n🔍 SEARCHING FOR FEATURE SUGGESTIONS")
    search_memory.invoke({"query": "feature dark mode"})


def hot_path_vs_background_demo():
    """
    Demonstrate the difference between hot path and background memory updates.
    """
    print("\n" + "=" * 80)
    print("🔥 HOT PATH VS BACKGROUND MEMORY UPDATES")
    print("=" * 80)

    # Hot path example (synchronous)
    print("\n📌 HOT PATH MEMORY UPDATE (Synchronous)")
    print("1. Customer inquiry received")
    print("2. Search memory for context")
    print("3. Process inquiry")
    print("4. Update memory ← This happens BEFORE sending response")
    print("5. Send response to customer")
    print("⏱️ Response time: 2.5 seconds (slower but with latest information)")

    # Background example (asynchronous)
    print("\n📌 BACKGROUND MEMORY UPDATE (Asynchronous)")
    print("1. Customer inquiry received")
    print("2. Search memory for context")
    print("3. Process inquiry")
    print("4. Send response to customer")
    print("5. Update memory ← This happens AFTER sending response")
    print(
        "⏱️ Response time: 1.2 seconds (faster but might use slightly outdated information)"
    )

    # Tradeoffs
    print("\n📊 TRADEOFFS:")
    print("• Hot Path: Higher latency, most up-to-date information")
    print("• Background: Lower latency, potentially slightly outdated information")


if __name__ == "__main__":
    print("\n🚀 MEMORY SYSTEM DEMONSTRATION FOR CUSTOMER SUPPORT AGENT")

    # Run the customer journey simulation
    run_customer_journey_simulation()

    # Demonstrate hot path vs background memory updates
    hot_path_vs_background_demo()

    print("\n✅ DEMONSTRATION COMPLETE")
