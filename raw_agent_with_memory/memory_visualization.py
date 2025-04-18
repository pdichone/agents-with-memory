import json
import os
import datetime
from tabulate import tabulate


class MemoryVisualizer:
    """
    Utility class to visualize the different memory systems
    for educational purposes.
    """

    def __init__(self, memory_dir: str = "./agent_memory"):
        self.memory_dir = memory_dir
        self.semantic_file = os.path.join(memory_dir, "facts_semantic.json")
        self.episodic_file = os.path.join(memory_dir, "conversations_episodic.json")
        self.procedural_file = os.path.join(memory_dir, "procedures.json")

    def _load_json(self, file_path):
        """Load data from a JSON file if it exists."""
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return None
        return None

    def show_semantic_memory(self):
        """Display the content of semantic memory (facts)."""
        facts = self._load_json(self.semantic_file)

        if not facts:
            print("Semantic memory is empty or could not be loaded.")
            return

        # Prepare data for tabular display
        table_data = []
        for i, fact in enumerate(facts, 1):
            created_date = datetime.datetime.fromisoformat(
                fact.get("timestamp", "")
            ).strftime("%Y-%m-%d %H:%M")
            table_data.append(
                [i, fact.get("content", ""), fact.get("category", ""), created_date]
            )

        # Display as table
        print("\n=== SEMANTIC MEMORY (Facts) ===")
        print(
            tabulate(
                table_data,
                headers=["#", "Content", "Category", "Created"],
                tablefmt="pretty",
            )
        )

    def show_episodic_memory(self, limit=10):
        """Display the content of episodic memory (conversations)."""
        conversations = self._load_json(self.episodic_file)

        if not conversations:
            print("Episodic memory is empty or could not be loaded.")
            return

        # Sort by timestamp (newest first)
        sorted_conversations = sorted(
            conversations, key=lambda x: x.get("timestamp", ""), reverse=True
        )

        # Limit the number of conversations to display
        display_conversations = sorted_conversations[:limit]

        print("\n=== EPISODIC MEMORY (Conversations) ===")
        for i, conv in enumerate(display_conversations, 1):
            timestamp = datetime.datetime.fromisoformat(
                conv.get("timestamp", "")
            ).strftime("%Y-%m-%d %H:%M")
            print(f"\n--- Conversation {i} - {timestamp} ---")
            print(f"User: {conv.get('user_message', '')}")
            print(f"Agent: {conv.get('agent_response', '')}")

    def show_procedural_memory(self):
        """Display the content of procedural memory (procedures)."""
        procedures = self._load_json(self.procedural_file)

        if not procedures:
            print("Procedural memory is empty or could not be loaded.")
            return

        print("\n=== PROCEDURAL MEMORY (Procedures) ===")
        for i, (name, proc) in enumerate(procedures.items(), 1):
            print(f"\n--- Procedure {i}: {name} ---")
            if proc.get("description"):
                print(f"Description: {proc.get('description')}")

            print("Steps:")
            for j, step in enumerate(proc.get("steps", []), 1):
                print(f"  {j}. {step}")

            usage_count = proc.get("usage_count", 0)
            print(f"Usage count: {usage_count}")

            if proc.get("last_used"):
                last_used = datetime.datetime.fromisoformat(
                    proc.get("last_used")
                ).strftime("%Y-%m-%d %H:%M")
                print(f"Last used: {last_used}")

    def visualize_all(self):
        """Visualize all memory systems."""
        print("\n======= AI AGENT MEMORY VISUALIZATION =======")
        print(f"Memory directory: {self.memory_dir}")

        self.show_semantic_memory()
        self.show_procedural_memory()
        self.show_episodic_memory()

        print(
            "\nNote: Working memory is volatile and only exists in RAM during runtime."
        )
        print("==================================================\n")


# Usage example
if __name__ == "__main__":
    visualizer = MemoryVisualizer()
    visualizer.visualize_all()

    # Or you can view specific memory types:
    # visualizer.show_semantic_memory()  # Show facts
    # visualizer.show_episodic_memory(limit=3)  # Show last 3 conversations
    # visualizer.show_procedural_memory()  # Show procedures
