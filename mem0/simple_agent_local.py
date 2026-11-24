from openai import OpenAI
from mem0 import Memory
from dotenv import load_dotenv

load_dotenv()

# This example runs Mem0 locally! all for free and no need for an API key.
openai_client = OpenAI()
# Pin the LLM mem0 uses internally (otherwise mem0 picks its own default model)
config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-6-luna",
            # gpt-6-luna: tell mem0 to send reasoning-model params (no temperature/max_tokens)
            # and turn reasoning off so its internal tool calls are accepted
            "is_reasoning_model": True,
            "reasoning_effort": "none",
        },
    }
}
memory = Memory.from_config(config)


def chat_with_memories(message: str, user_id: str = "default_user") -> str:
    # Retrieve relevant memories
    relevant_memories = memory.search(query=message, filters={"user_id": user_id}, limit=3)
    memories_str = "\n".join(
        f"- {entry['memory']}" for entry in relevant_memories["results"]
    )

    # Generate Assistant response
    system_prompt = f"You are a helpful AI. Answer the question based on query and memories.\nUser Memories:\n{memories_str}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message},
    ]
    response = openai_client.chat.completions.create(
        model="gpt-6-luna", messages=messages
    )
    assistant_response = response.choices[0].message.content

    # Create new memories from the conversation
    messages.append({"role": "assistant", "content": assistant_response})
    memory.add(messages, user_id=user_id)

    return assistant_response


def main():
    print("Chat with AI (type 'exit' to quit)")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        print(f"AI: {chat_with_memories(user_input)}")


if __name__ == "__main__":
    main()
