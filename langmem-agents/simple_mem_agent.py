# Import core components
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool


# Set up storage
store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": "openai:text-embedding-3-small",
    }
)

# Create an agent with memory capabilities
agent = create_react_agent(
    # "anthropic:claude-3-5-sonnet-latest",
    "openai:gpt-4o-mini",
    tools=[
        # Memory tools use LangGraph's BaseStore for persistence (4)
        create_manage_memory_tool(namespace=("memories",)),
        create_search_memory_tool(namespace=("memories",)),
    ],
    store=store,
)
# Store a new memory
agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "Remember that I prefer sushi over pasta."}
        ]
    }
)

# Retrieve the stored memory
response = agent.invoke(
    {"messages": [{"role": "user", "content": "What are my food preferences?"}]}
)

print(response["messages"][-1].content)
