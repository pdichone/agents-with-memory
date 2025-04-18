from langchain.chat_models import init_chat_model
from langgraph.func import entrypoint
from langgraph.store.memory import InMemoryStore
from langmem import ReflectionExecutor, create_memory_store_manager
from dotenv import load_dotenv
import asyncio
import time
import json
import uuid

load_dotenv()

# Initialize memory store with vector embedding capability
store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": "openai:text-embedding-3-small",
    }
)

# Initialize the LLM
llm = init_chat_model("openai:gpt-4o-mini", temperature=0.2)

# Create memory manager to extract memories from conversations
memory_manager = create_memory_store_manager(
    "openai:gpt-4o-mini",
    namespace=("food_preferences",),
)

# Global variable to control verbosity
verbose_mode = True


@entrypoint(store=store)
async def recipe_assistant(message: str):
    """
    A recipe assistant that remembers food preferences.
    """
    global verbose_mode

    if verbose_mode:
        print("\n" + "-" * 60)
        print("🔄 MEMORY SYSTEM INTERNALS:")
        print("-" * 60)

    # Step 1: Vector search for relevant memories
    if verbose_mode:
        print("\n🔍 SEARCHING VECTOR STORE")
        print(f"  Query: '{message}'")
        print(f"  Namespace: ('food_preferences',)")
        print(f"  Limit: 3")

    # First, check if there are any relevant food preferences in memory
    relevant_memories = store.search(("food_preferences",), query=message, limit=3)

    # Display embedding and similarity details for memory retrieval
    if verbose_mode:
        if relevant_memories:
            print("\n  Results:")
            for i, memory in enumerate(relevant_memories, 1):
                print(f"  {i}. Memory ID: {memory.key}")
                print(f"     Content: {memory.value['content']['content']}")
                if hasattr(memory, "score"):
                    print(f"     Relevance score: {memory.score}")
                print(f"     Created at: {memory.created_at}")
        else:
            print("\n  No relevant memories found")

    # Format memories as context for the LLM
    memory_context = ""
    if relevant_memories:
        memory_context = "Previous information about the user's food preferences:\n"
        for memory in relevant_memories:
            memory_context += f"- {memory.value['content']['content']}\n"

    # Step 2: Construct prompt with memory context
    if verbose_mode:
        print("\n📋 CONSTRUCTING AUGMENTED PROMPT")
        print(f"  Base message: '{message}'")
        if memory_context:
            print(f"  Prepending {len(relevant_memories)} memories as context")
        else:
            print("  No memory context to prepend")

    prompt = f"{memory_context}\nUser message: {message}\n\nPlease provide a helpful response about recipes or food, using any relevant previous preferences if appropriate."

    if verbose_mode:
        print("\n  Final prompt to LLM:")
        print("  " + "-" * 40)
        for line in prompt.split("\n"):
            print(f"  {line}")
        print("  " + "-" * 40)

    # Step 3: Generate response from the LLM
    if verbose_mode:
        print("\n🤖 GENERATING LLM RESPONSE")

    start_time = time.time()
    response = llm.invoke(prompt)
    end_time = time.time()

    if verbose_mode:
        print(f"  Response time: {end_time - start_time:.2f} seconds")
        print(f"  Response length: {len(response.content)} characters")

    # Step 4: Memory extraction process
    if verbose_mode:
        print("\n🧠 MEMORY EXTRACTION PROCESS")
        print("  Packaging conversation for memory extraction:")
        print("  - User message")
        print("  - Assistant response")

    # Create conversation object to process
    conversation_id = str(uuid.uuid4())
    if verbose_mode:
        print(f"  Conversation ID: {conversation_id}")

    to_process = {"messages": [{"role": "user", "content": message}] + [response]}

    if verbose_mode:
        print("\n  Sending to memory manager for extraction...")

    # Get the count of memories before extraction
    before_count = len(list(store.search(("food_preferences",))))

    # Extract memories from the conversation
    memory_extraction_start = time.time()
    extraction_result = await memory_manager.ainvoke(to_process)
    memory_extraction_end = time.time()

    if verbose_mode:
        print(
            f"  Extraction time: {memory_extraction_end - memory_extraction_start:.2f} seconds"
        )

    # Step 5: Check for new memories
    after_count = len(list(store.search(("food_preferences",))))
    new_count = after_count - before_count

    if verbose_mode:
        print(f"\n💾 MEMORY STORAGE RESULTS")
        print(f"  Memories before: {before_count}")
        print(f"  Memories after: {after_count}")
        print(f"  New memories added: {new_count}")

        if new_count > 0:
            print("\n  New memories:")
            # Get all memories and sort by creation time to find newest
            all_memories = list(store.search(("food_preferences",)))
            all_memories.sort(key=lambda x: x.created_at, reverse=True)

            for i, memory in enumerate(all_memories[:new_count], 1):
                print(f"  {i}. ID: {memory.key}")
                print(f"     Content: {memory.value['content']['content']}")
                print(f"     Created at: {memory.created_at}")

        print("\n" + "-" * 60 + "\n")

    return response.content


async def interactive_console():
    """
    Interactive console interface for the recipe assistant.
    """
    global verbose_mode

    print("\n" + "=" * 70)
    print("🍳 Welcome to the Interactive Recipe Assistant with Memory Visualization 🍳")
    print("This assistant remembers your food preferences over time.")
    print("Commands:")
    print("  'exit' or 'quit': End the conversation")
    print("  'memories': View all stored memories")
    print("  'verbose on/off': Toggle detailed memory system visibility")
    print("=" * 70 + "\n")

    while True:
        # Get user input
        user_input = input("YOU: ")

        # Handle special commands
        if user_input.lower() in ["exit", "quit"]:
            print("\nThank you for using the Recipe Assistant. Goodbye!")
            break
        elif user_input.lower() == "memories":
            print("\n🧠 ALL STORED MEMORIES:")
            memories = list(store.search(("food_preferences",)))
            if memories:
                # Sort by creation time, newest first
                memories.sort(key=lambda x: x.created_at, reverse=True)
                for i, memory in enumerate(memories, 1):
                    print(f"{i}. ID: {memory.key}")
                    print(f"   Content: {memory.value['content']['content']}")
                    print(f"   Created: {memory.created_at}")
                    print()
            else:
                print("No memories stored yet.")
            print()
            continue
        elif user_input.lower() == "verbose on":
            verbose_mode = True
            print("Verbose mode enabled - showing memory system details")
            continue
        elif user_input.lower() == "verbose off":
            verbose_mode = False
            print("Verbose mode disabled - hiding memory system details")
            continue

        # Process normal input through the assistant
        response = await recipe_assistant.ainvoke(user_input)
        print(f"\nASSISTANT: {response}\n")


if __name__ == "__main__":
    asyncio.run(interactive_console())
