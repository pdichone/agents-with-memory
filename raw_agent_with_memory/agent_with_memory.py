import json
import os
import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()


class AgentMemory:
    """Simple memory system for an AI agent."""

    def __init__(self, storage_dir: str = "./agent_memory"):
        """Initialize the memory system with a storage directory."""
        # Create storage directories
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)

        # Memory components
        self.facts_file = os.path.join(storage_dir, "facts_semantic.json")
        self.conversations_file = os.path.join(
            storage_dir, "conversations_episodic.json"
        )
        self.procedures_file = os.path.join(storage_dir, "procedures.json")

        # Initialize memory stores
        self.facts = self._load_json(self.facts_file, default=[])
        self.procedures = self._load_json(self.procedures_file, default={})
        self.conversations = self._load_json(self.conversations_file, default=[])

        # Working memory (stays in RAM)
        self.working_memory = []
        self.working_memory_capacity = 10

    def _load_json(self, file_path: str, default: Any = None) -> Any:
        """Load data from a JSON file or return default if file doesn't exist."""
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return default
        return default

    def _save_json(self, data: Any, file_path: str) -> None:
        """Save data to a JSON file."""
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

    def add_fact(self, content: str, category: Optional[str] = None) -> None:
        """Add a fact to semantic memory."""
        fact = {
            "content": content,
            "category": category,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        self.facts.append(fact)
        self._save_json(self.facts, self.facts_file)

    def add_procedure(
        self, name: str, steps: List[str], description: Optional[str] = None
    ) -> None:
        """Add a procedure to procedural memory."""
        procedure = {
            "name": name,
            "steps": steps,
            "description": description,
            "timestamp": datetime.datetime.now().isoformat(),
            "usage_count": 0,
        }
        self.procedures[name] = procedure
        self._save_json(self.procedures, self.procedures_file)

    def add_conversation(
        self,
        user_message: str,
        agent_response: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a conversation turn to episodic memory."""
        conversation = {
            "user_message": user_message,
            "agent_response": agent_response,
            "timestamp": datetime.datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        self.conversations.append(conversation)
        self._save_json(self.conversations, self.conversations_file)

        # Also update working memory
        self.add_to_working_memory(f"User: {user_message}", importance=1.0)
        self.add_to_working_memory(f"Agent: {agent_response}", importance=0.9)

    def add_to_working_memory(self, content: str, importance: float = 1.0) -> None:
        """Add an item to working memory with importance score."""
        item = {
            "content": content,
            "importance": importance,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        self.working_memory.append(item)

        # If over capacity, remove least important items
        if len(self.working_memory) > self.working_memory_capacity:
            self.working_memory.sort(key=lambda x: (x["importance"], x["timestamp"]))
            self.working_memory = self.working_memory[1:]  # Remove least important

    def search_facts(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Simple keyword search for facts."""
        query_terms = query.lower().split()
        results = []

        for fact in self.facts:
            content = fact["content"].lower()
            # Score based on number of matching terms
            score = sum(1 for term in query_terms if term in content)
            if score > 0:
                results.append((fact, score))

        # Sort by score (descending) and return top matches
        results.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in results[:limit]]

    def search_conversations(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Simple keyword search for past conversations."""
        query_terms = query.lower().split()
        results = []

        for conv in self.conversations:
            text = f"{conv['user_message']} {conv['agent_response']}".lower()
            # Score based on number of matching terms
            score = sum(1 for term in query_terms if term in text)
            if score > 0:
                results.append((conv, score))

        # Sort by score (descending) and return top matches
        results.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in results[:limit]]

    def search_procedures(self, query: str, limit: int = 2) -> List[Dict[str, Any]]:
        """Search for procedures by keyword matching."""
        query = query.lower()
        results = []

        for name, procedure in self.procedures.items():
            text = f"{name} {procedure.get('description', '')}".lower()
            # Check if query terms appear in the text
            if query in text:
                results.append(procedure)

        # Sort by usage count and return top matches
        results.sort(key=lambda x: x.get("usage_count", 0), reverse=True)
        return results[:limit]

    def get_recent_conversations(self, count: int = 5) -> List[Dict[str, Any]]:
        """Get the most recent conversations."""
        return (
            self.conversations[-count:]
            if len(self.conversations) >= count
            else self.conversations
        )

    def generate_context_for_llm(self, current_message: str) -> str:
        """Generate a context string for the LLM using relevant memory."""
        # Get working memory
        working_items = sorted(
            self.working_memory,
            key=lambda x: (x["importance"], x["timestamp"]),
            reverse=True,
        )
        working_memory_text = "\n".join(
            [f"- {item['content']}" for item in working_items]
        )

        # Get recent conversations
        recent = self.get_recent_conversations(count=3)
        recent_text = "\n".join(
            [
                f"User: {conv['user_message']}\nAgent: {conv['agent_response']}"
                for conv in recent
            ]
        )

        # Get relevant facts and conversations
        relevant_facts = self.search_facts(current_message)
        facts_text = "\n".join([f"- {fact['content']}" for fact in relevant_facts])

        # Get relevant procedures
        relevant_procedures = self.search_procedures(current_message)
        procedures_text = ""
        for proc in relevant_procedures:
            steps = "\n".join(
                [f"  {i+1}. {step}" for i, step in enumerate(proc["steps"])]
            )
            procedures_text += f"Procedure: {proc['name']}\n{steps}\n\n"

        # Combine everything into a context string
        context = f"""
### Current Context (Working Memory):
{working_memory_text}

### Recent Conversation History:
{recent_text}

### Relevant Facts from Memory:
{facts_text}

### Relevant Procedures:
{procedures_text}
"""
        return context.strip()


class OpenAIAgent:
    """AI agent using OpenAI's GPT-4o-mini with memory capabilities."""

    def __init__(self, api_key: str = None, memory_dir: str = "./agent_memory"):
        """Initialize the agent with an OpenAI API key and memory system."""
        self.api_key = (
            api_key  # If None, the OpenAI client will use OPENAI_API_KEY env var
        )
        self.memory = AgentMemory(memory_dir)
        self.model = "gpt-4o-mini"  # You can change this to "gpt-4o" for the full model

        # Add some basic facts about the agent itself
        self.memory.add_fact("I am an AI assistant with memory capabilities.")
        self.memory.add_fact("I can remember user interactions and recall them later.")
        self.memory.add_fact("I can store and retrieve factual information.")
        self.memory.add_fact("I can remember and execute procedures and workflows.")

    def generate_system_prompt(self) -> str:
        """Generate the system prompt for the agent."""
        return """You are a helpful AI assistant with memory capabilities. You can remember past interactions, 
facts you've learned, and procedures you know. Use the provided context to give personalized, 
contextually relevant responses. If you don't have relevant memory information, you can draw on 
your general knowledge. Always be helpful, accurate, and conversational."""

    def query(self, user_message: str) -> str:
        # Check if the message is a command to learn a fact
        if user_message.lower().startswith("remember that"):
            fact = user_message[len("remember that ") :].strip()
            return self.learn_fact(fact)

        # Check if the message is a command to learn a procedure
        elif user_message.lower().startswith("remember the steps for"):
            # This example assumes the procedure's name is part of the message.
            # You might need a more robust parser to extract the steps.
            # Here we assume a simple split: first sentence is the title, subsequent sentences are steps.
            try:
                title_part, steps_part = user_message.split(":", 1)
                procedure_name = title_part[len("remember the steps for") :].strip()
                # Assume steps are separated by commas in this simple example
                steps = [step.strip() for step in steps_part.split(",")]
                return self.learn_procedure(procedure_name, steps)
            except ValueError:
                return "I couldn't parse the procedure. Please follow the format: 'Remember the steps for [Procedure Name]: step1, step2, step3'"

        # If no special command, proceed with a normal query
        # Get relevant context from memory
        memory_context = self.memory.generate_context_for_llm(user_message)

        # Create the messages for the API
        messages = [
            {"role": "system", "content": self.generate_system_prompt()},
            {"role": "system", "content": f"Context from memory:\n{memory_context}"},
            {"role": "user", "content": user_message},
        ]

        try:
            from openai import OpenAI

            client = OpenAI()
            completion = client.chat.completions.create(
                model=self.model, messages=messages, temperature=0.7, max_tokens=1000
            )
            response_text = completion.choices[0].message.content

            # Store the interaction in memory
            self.memory.add_conversation(user_message, response_text)

            return response_text
        except Exception as e:
            return f"Error: API request failed: {str(e)}"

    # def query(self, user_message: str) -> str:
    #     """Process a user message and generate a response using the OpenAI client library."""
    #     # Get relevant context from memory
    #     memory_context = self.memory.generate_context_for_llm(user_message)

    #     # Create the messages for the API
    #     messages = [
    #         {"role": "system", "content": self.generate_system_prompt()},
    #         {"role": "system", "content": f"Context from memory:\n{memory_context}"},
    #         {"role": "user", "content": user_message},
    #     ]

    #     try:
    #         # Use the OpenAI client library
    #         from openai import OpenAI

    #         client = OpenAI()
    #         # client = OpenAI(api_key=self.api_key)

    #         completion = client.chat.completions.create(
    #             model=self.model, messages=messages, temperature=0.7, max_tokens=1000
    #         )

    #         # Extract the response text
    #         response_text = completion.choices[0].message.content

    #         # Store the interaction in memory
    #         self.memory.add_conversation(user_message, response_text)

    #         return response_text

    #     except Exception as e:
    #         error_msg = f"Error: API request failed: {str(e)}"
    #         return error_msg

    def learn_fact(self, fact: str, category: Optional[str] = None) -> str:
        """Add a new fact to the agent's memory."""
        self.memory.add_fact(fact, category)
        confirmation = f"I've learned this fact: {fact}"
        print(confirmation)  # Print confirmation to console
        return confirmation

    def learn_procedure(
        self, name: str, steps: List[str], description: Optional[str] = None
    ) -> str:
        """Add a new procedure to the agent's memory."""
        self.memory.add_procedure(name, steps, description)
        confirmation = f"I've learned the procedure: {name}"
        print(confirmation)  # Print confirmation to console
        return confirmation


# Example usage
def main():
    # Initialize the agent (will use OPENAI_API_KEY environment variable by default)
    agent = OpenAIAgent()

    # Add some initial knowledge
    agent.learn_fact(
        "Python is a high-level programming language known for its readability.",
        "programming",
    )
    agent.learn_fact(
        "Memory systems in AI agents are crucial for continuity and personalization.",
        "AI",
    )

    agent.learn_procedure(
        "Create a simple AI agent with memory",
        [
            "Initialize memory storage (files, database, etc.)",
            "Create functions to add facts to semantic memory",
            "Create functions to store conversation history in episodic memory",
            "Create functions to retrieve relevant memory based on context",
            "Connect memory system to an LLM like GPT-4o-mini",
            "Implement a query function that includes memory context in prompts",
            "Add memory updating after each interaction",
        ],
        "Steps to create a basic AI agent with memory capabilities",
    )

    print("AI Assistant with Memory initialized! Type 'exit' to quit.")
    print("-" * 50)

    # Simple conversation loop
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("\nAssistant: Goodbye! It was nice talking with you.")
            break

        # Process the user's message
        response = agent.query(user_input)
        print(f"\nAssistant: {response}")


if __name__ == "__main__":
    main()
