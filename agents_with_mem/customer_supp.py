import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal
from typing_extensions import TypedDict, Annotated
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.store.memory import InMemoryStore
from langgraph.graph import add_messages, StateGraph, START, END
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent

# Load API tokens
_ = load_dotenv()


# Define user profile
profile = {
    "name": "Anna",
    "full_name": "Anna Chen",
    "user_profile_background": "Customer support specialist for a software company",
}

# Define prompt instructions
prompt_instructions = {
    "triage_rules": {
        "ignore": "Spam inquiries, marketing solicitations, non-customer messages",
        "notify": "Product feedback, feature requests, general inquiries from customers",
        "respond": "Technical support questions, urgent issues, account problems",
    },
    "agent_instructions": "Use these tools when appropriate to help Anna provide excellent customer support efficiently.",
}


# Define State
class State(TypedDict):
    inquiry_input: dict
    messages: Annotated[list, add_messages]


# Initialize the language model
llm = init_chat_model("openai:gpt-4o-mini")


# Define Router for message classification
class Router(BaseModel):
    """Analyze the customer inquiry and route it according to its content."""

    reasoning: str = Field(
        description="Step-by-step reasoning behind the classification."
    )
    classification: Literal["ignore", "notify", "respond"] = Field(
        description="The classification of a message: 'ignore' for irrelevant messages, "
        "'notify' for important information that doesn't need an immediate response, "
        "'respond' for inquiries that need a reply",
    )


# Set up structured output for the router
llm_router = llm.with_structured_output(Router)

# Define triage prompt templates
triage_system_prompt = """
You are an AI assistant for {full_name}, who is a {user_profile_background}.

Your job is to classify incoming customer inquiries into one of three categories:
1. IGNORE: {triage_no}
2. NOTIFY: {triage_notify}
3. RESPOND: {triage_email}

Think carefully about the content of each message to determine how it should be classified.
"""

triage_user_prompt = """
Please classify this customer inquiry:

From: {author}
To: {to}
Subject: {subject}
Message: {message_thread}

Which category should this be assigned to: IGNORE, NOTIFY, or RESPOND?
"""


# Define tools for handling customer support tasks
@tool
def send_response(to: str, subject: str, content: str) -> str:
    """Send a response to a customer inquiry."""
    # Placeholder response - in real app would send actual message
    return f"Response sent to {to} with subject '{subject}'"


@tool
def create_support_ticket(
    customer_name: str, issue_type: str, description: str, priority: str
) -> str:
    """Create a support ticket in the system."""
    # Placeholder response - in real app would create ticket in support system
    return f"Support ticket created for {customer_name} with priority {priority}"


# Set up memory store with embeddings
store = InMemoryStore(index={"embed": "openai:text-embedding-3-small"})


# Create simplified memory tools
@tool
def manage_memory(content: str = None, action: str = "create", id: str = None) -> str:
    """Create, update, or delete memories about customers and interactions."""
    # This is a simplified version - in a real app, would interact with a memory store
    if action == "create" and content:
        memory_id = "mem_" + str(hash(content))[:8]
        print(f"Created memory: {memory_id} with content: {content}")
        return f"created memory {memory_id}"
    return "Memory operation completed."


@tool
def search_memory(query: str) -> str:
    """Search for relevant customer information in memory."""
    # Simplified mock implementation
    print(f"Searching memory for: {query}")
    return "[]"  # Return empty results for now


# Define system prompt with memory capabilities
agent_system_prompt = """
< Role >
You are {full_name}'s customer support assistant. You are an expert at helping customers with their technical questions and account issues.
</ Role >

< Tools >
You have access to the following tools to help manage customer support:

1. send_response(to, subject, content) - Send responses to customers
2. create_support_ticket(customer_name, issue_type, description, priority) - Create support tickets
3. manage_memory - Store any relevant information about customers
4. search_memory - Search for any relevant information that may have been stored in memory
</ Tools >

< Instructions >
{instructions}
</ Instructions >
"""


# Create the prompt for the agent
def create_prompt(state):
    return [
        {
            "role": "system",
            "content": agent_system_prompt.format(
                instructions=prompt_instructions["agent_instructions"], **profile
            ),
        }
    ] + state["messages"]


# Set up tools for the agent
tools = [send_response, create_support_ticket, manage_memory, search_memory]
# Create the response agent
response_agent = create_react_agent(
    # "anthropic:claude-3-5-sonnet-latest",
    "openai:gpt-4o-mini",
    tools=tools,
    prompt=create_prompt,
    store=store,  # Pass the store to ensure memory functions work
)


# Define the triage router function - THIS IS THE KEY FIX
def triage_router(state: State) -> Command[Literal["response_agent", "__end__"]]:
    author = state["inquiry_input"]["author"]
    to = state["inquiry_input"]["to"]
    subject = state["inquiry_input"]["subject"]
    message_thread = state["inquiry_input"]["message_thread"]

    system_prompt = triage_system_prompt.format(
        full_name=profile["full_name"],
        name=profile["name"],
        user_profile_background=profile["user_profile_background"],
        triage_no=prompt_instructions["triage_rules"]["ignore"],
        triage_notify=prompt_instructions["triage_rules"]["notify"],
        triage_email=prompt_instructions["triage_rules"]["respond"],
        examples=None,
    )
    user_prompt = triage_user_prompt.format(
        author=author, to=to, subject=subject, message_thread=message_thread
    )
    result = llm_router.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    print(f"Classification: {result.classification}")

    if result.classification == "respond":
        print("🔴 Classification: RESPOND - This inquiry requires a response")
        # This is the correct way to return a command with state update
        return Command(
            goto="response_agent",
            update={
                "messages": [
                    {
                        "role": "user",
                        "content": f"Respond to the customer inquiry:\nFrom: {author}\nSubject: {subject}\nMessage: {message_thread}",
                    }
                ]
            },
        )
    else:
        print(
            f"⚪ Classification: {result.classification.upper()} - No response needed"
        )
        # For ignore/notify, just end without state update
        return Command(goto="__end__")


# Create the graph
support_agent = StateGraph(State)
support_agent.add_node("triage_router", triage_router)
support_agent.add_node("response_agent", response_agent)
support_agent.add_edge(START, "triage_router")
support_agent.add_edge("triage_router", "response_agent")
support_agent.add_edge("triage_router", END)
support_agent.add_edge("response_agent", END)


# Compile the graph
support_agent = support_agent.compile()

# Configuration for the agent
config = {"configurable": {"langgraph_user_id": "user123"}}


# Example usage
if __name__ == "__main__":
    # Example: Technical support question
    tech_inquiry = {
        "author": "david.wilson@example.com",
        "to": "support@ourcompany.com",
        "subject": "Login issues with mobile app",
        "message_thread": """Hello Support,

I've been trying to log into your mobile app for the past day but keep getting an "Authentication Failed" error. 
I've verified my password is correct through the website login. 
Is there a known issue with the mobile app login system?

Thanks,
David Wilson""",
    }

    print("\n=== HANDLING TECHNICAL SUPPORT INQUIRY ===")

    # Initialize state properly with both required fields
    initial_state = {"inquiry_input": tech_inquiry, "messages": []}

    try:
        # Invoke the agent with error handling
        response = support_agent.invoke(initial_state, config=config)

        print("\nAGENT RESPONSE:")
        for message in response.get("messages", []):
            if hasattr(message, "pretty_print"):
                message.pretty_print()
            else:
                print(f"Role: {message.get('role')}")
                content = message.get("content")
                print(
                    f"Content: {content[:200]}..."
                    if len(content) > 200
                    else f"Content: {content}"
                )
                print("-" * 50)
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback

        traceback.print_exc()

    # Example: Feature request (should be classified as notify)
    feature_request = {
        "author": "sarah.thompson@example.com",
        "to": "support@ourcompany.com",
        "subject": "Feature suggestion",
        "message_thread": """Hi there,

I love your product, but I think it would be even better if you could add dark mode support.
Many apps have this now and it would be great for using the app at night.

Best,
Sarah""",
    }

    print("\n=== HANDLING FEATURE REQUEST ===")
    try:
        support_agent.invoke(
            {"inquiry_input": feature_request, "messages": []}, config=config
        )
    except Exception as e:
        print(f"Error occurred: {e}")
