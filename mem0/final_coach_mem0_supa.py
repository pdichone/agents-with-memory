import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from mem0 import Memory
import supabase
from supabase.client import Client, ClientOptions
import uuid

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase_url = os.environ.get("SUPABASE_URL", "")
supabase_key = os.environ.get("SUPABASE_KEY", "")
supabase_client = supabase.create_client(supabase_url, supabase_key)

model = os.getenv("MODEL_CHOICE", "")
print(f"Using model===> {model}")
# Streamlit page configuration
st.set_page_config(
    page_title="Health Coach Assistant",
    page_icon="🏋️‍♀️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Cache OpenAI client and Memory instance
@st.cache_resource
def get_openai_client():
    return OpenAI()


@st.cache_resource
def get_memory():
    config = {
        "llm": {"provider": "openai", "config": {"model": model}},
        "vector_store": {
            "provider": "supabase",
            "config": {
                "connection_string": os.getenv("DATABASE_URL"),
                "collection_name": "chat_memories",
            },
        },
    }
    return Memory.from_config(config)


# Get cached resources
openai_client = get_openai_client()
memory = get_memory()


# Authentication functions
def sign_up(email, password, full_name):
    try:
        response = supabase_client.auth.sign_up(
            {
                "email": email,
                "password": password,
                "options": {"data": {"full_name": full_name}},
            }
        )
        if response and response.user:
            st.session_state.authenticated = True
            st.session_state.user = response.user
            st.rerun()
        return response
    except Exception as e:
        st.error(f"Error signing up: {str(e)}")
        return None


def sign_in(email, password):
    try:
        response = supabase_client.auth.sign_in_with_password(
            {"email": email, "password": password}
        )
        if response and response.user:
            # Store user info directly in session state
            st.session_state.authenticated = True
            st.session_state.user = response.user
            st.rerun()
        return response
    except Exception as e:
        st.error(f"Error signing in: {str(e)}")
        return None


def sign_out():
    try:
        supabase_client.auth.sign_out()
        # Clear only authentication-related session state
        st.session_state.authenticated = False
        st.session_state.user = None
        # Set a flag to trigger rerun on next render
        st.session_state.logout_requested = True
    except Exception as e:
        st.error(f"Error signing out: {str(e)}")


# Health Coach chat function with memory
def chat_with_health_coach(message, user_id):
    # Retrieve relevant memories
    relevant_memories = memory.search(query=message, user_id=user_id, limit=5)

    # Format memories for inclusion in the prompt
    formatted_memories = ""
    if relevant_memories and len(relevant_memories["results"]) > 0:
        for entry in relevant_memories["results"]:
            formatted_memories += f"- {entry['memory']}\n"

    # Health Coach system prompt
    system_prompt = """
    You are a personal health coach with access to memory. Your goal is to help users achieve their fitness goals
    by providing personalized advice and tracking their progress over time.

    You have information about the user's past interactions that you can use to provide personalized advice.
    
    User's Relevant Memories:
    {}
    
    Always be supportive, encouraging, and provide actionable advice. Don't mention the memory system directly to the user - 
    they should just experience personalized advice.
    """.format(
        formatted_memories
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message},
    ]

    with st.spinner("Your health coach is thinking..."):
        response = openai_client.chat.completions.create(model=model, messages=messages)
        assistant_response = response.choices[0].message.content

    # Create new memories from the conversation
    messages.append({"role": "assistant", "content": assistant_response})
    memory.add(messages, user_id=user_id)

    return assistant_response


# Initialize session state
if not st.session_state.get("messages", None):
    st.session_state.messages = []

if not st.session_state.get("authenticated", None):
    st.session_state.authenticated = False

if not st.session_state.get("user", None):
    st.session_state.user = None

if not st.session_state.get("health_profile", None):
    st.session_state.health_profile = {
        "goals": "",
        "fitness_level": "",
        "dietary_preferences": "",
        "medical_conditions": "",
    }

# Check for logout flag and clear it after processing
if st.session_state.get("logout_requested", False):
    st.session_state.logout_requested = False
    st.rerun()

# Sidebar for authentication and health profile
with st.sidebar:
    st.title("🏋️‍♀️ Health Coach")

    if not st.session_state.authenticated:
        tab1, tab2 = st.tabs(["Login", "Sign Up"])

        with tab1:
            st.subheader("Login")
            login_email = st.text_input("Email", key="login_email")
            login_password = st.text_input(
                "Password", type="password", key="login_password"
            )
            login_button = st.button("Login")

            if login_button:
                if login_email and login_password:
                    sign_in(login_email, login_password)
                else:
                    st.warning("Please enter both email and password.")

        with tab2:
            st.subheader("Sign Up")
            signup_email = st.text_input("Email", key="signup_email")
            signup_password = st.text_input(
                "Password", type="password", key="signup_password"
            )
            signup_name = st.text_input("Full Name", key="signup_name")
            signup_button = st.button("Sign Up")

            if signup_button:
                if signup_email and signup_password and signup_name:
                    response = sign_up(signup_email, signup_password, signup_name)
                    if response and response.user:
                        st.success(
                            "Sign up successful! Please check your email to confirm your account."
                        )
                    else:
                        st.error("Sign up failed. Please try again.")
                else:
                    st.warning("Please fill in all fields.")
    else:
        user = st.session_state.user
        if user:
            st.success(f"Logged in as: {user.email}")
            st.button("Logout", on_click=sign_out)

            # Display and edit health profile
            st.subheader("Your Health Profile")

            with st.expander("Update Your Health Profile", expanded=False):
                st.session_state.health_profile["goals"] = st.text_area(
                    "Your Fitness Goals",
                    value=st.session_state.health_profile["goals"],
                    placeholder="e.g., Lose weight, build muscle, improve endurance...",
                )

                st.session_state.health_profile["fitness_level"] = st.selectbox(
                    "Current Fitness Level",
                    options=["Beginner", "Intermediate", "Advanced"],
                    index=(
                        0
                        if not st.session_state.health_profile["fitness_level"]
                        else ["Beginner", "Intermediate", "Advanced"].index(
                            st.session_state.health_profile["fitness_level"]
                        )
                    ),
                )

                st.session_state.health_profile["dietary_preferences"] = st.text_area(
                    "Dietary Preferences/Restrictions",
                    value=st.session_state.health_profile["dietary_preferences"],
                    placeholder="e.g., Vegetarian, no dairy, low carb...",
                )

                st.session_state.health_profile["medical_conditions"] = st.text_area(
                    "Medical Conditions (if any)",
                    value=st.session_state.health_profile["medical_conditions"],
                    placeholder="e.g., Asthma, knee injury, heart condition...",
                )

                if st.button("Save Profile"):
                    # Create a memory entry for the health profile
                    profile_message = f"""
                    User updated their health profile:
                    - Goals: {st.session_state.health_profile['goals']}
                    - Fitness Level: {st.session_state.health_profile['fitness_level']}
                    - Dietary Preferences: {st.session_state.health_profile['dietary_preferences']}
                    - Medical Conditions: {st.session_state.health_profile['medical_conditions']}
                    """

                    memory.add(
                        [
                            {"role": "system", "content": "Health profile update"},
                            {"role": "user", "content": profile_message},
                        ],
                        user_id=user.id,
                    )

                    st.success("Health profile saved!")

            # Memory management options
            st.subheader("Memory Management")
            if st.button("Clear All Memories"):
                try:
                    # Check if the memory object has a clear method
                    if hasattr(memory, "clear"):
                        memory.clear(user_id=user.id)
                    else:
                        # Alternative approach if memory.clear is not available
                        # Use the search and delete approach
                        memories = memory.search(query="*", user_id=user.id, limit=1000)
                        if memories and "results" in memories:
                            # Log information for debugging
                            st.write(
                                f"Found {len(memories['results'])} memories to clear"
                            )
                            # We don't delete directly as we don't have a direct deletion API in memory object
                            # Instead, we'll clear the session and inform the user
                            st.info(
                                "Memory system doesn't support direct clearing. Session has been reset instead."
                            )

                    # Clear the session messages regardless
                    st.session_state.messages = []
                    st.success("Chat history cleared!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error clearing memories: {str(e)}")
                    st.info("Try restarting the application if the problem persists.")

# Main chat interface
if st.session_state.authenticated and st.session_state.user:
    # Use the user from session state directly
    user_id = st.session_state.user.id

    st.title("Your Personal Health Coach")
    st.write(
        "I'm your AI health coach with memory. I can provide personalized fitness advice based on your goals and progress."
    )

    # Quick action buttons
    st.subheader("Quick Actions")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("📝 Today's Workout"):
            quick_prompt = "Can you suggest a workout for me today based on my fitness level and goals?"
            st.session_state.messages.append({"role": "user", "content": quick_prompt})
            ai_response = chat_with_health_coach(quick_prompt, user_id)
            st.session_state.messages.append(
                {"role": "assistant", "content": ai_response}
            )
            st.rerun()

    with col2:
        if st.button("🍎 Nutrition Advice"):
            quick_prompt = "Can you give me some nutrition tips based on my dietary preferences and fitness goals?"
            st.session_state.messages.append({"role": "user", "content": quick_prompt})
            ai_response = chat_with_health_coach(quick_prompt, user_id)
            st.session_state.messages.append(
                {"role": "assistant", "content": ai_response}
            )
            st.rerun()

    with col3:
        if st.button("📊 Track Progress"):
            quick_prompt = "I want to track my fitness progress. What metrics should I be monitoring and how?"
            st.session_state.messages.append({"role": "user", "content": quick_prompt})
            ai_response = chat_with_health_coach(quick_prompt, user_id)
            st.session_state.messages.append(
                {"role": "assistant", "content": ai_response}
            )
            st.rerun()

    with col4:
        if st.button("🧘 Recovery Tips"):
            quick_prompt = "What are some good recovery practices I should incorporate into my fitness routine?"
            st.session_state.messages.append({"role": "user", "content": quick_prompt})
            ai_response = chat_with_health_coach(quick_prompt, user_id)
            st.session_state.messages.append(
                {"role": "assistant", "content": ai_response}
            )
            st.rerun()

    st.divider()

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    user_input = st.chat_input("Ask your health coach anything...")

    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Display user message
        with st.chat_message("user"):
            st.write(user_input)

        # Get AI response
        ai_response = chat_with_health_coach(user_input, user_id)

        # Add AI response to chat history
        st.session_state.messages.append({"role": "assistant", "content": ai_response})

        # Display AI response
        with st.chat_message("assistant"):
            st.write(ai_response)
else:
    st.title("Welcome to Your Personal Health Coach")
    st.write(
        "Please login or sign up to start chatting with your personal AI health coach."
    )
    st.write(
        "This application helps you achieve your fitness goals with personalized advice and tracking."
    )

    # Feature highlights
    st.subheader("Features")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🏋️‍♀️ Personalized Coaching")
        st.write(
            "Get custom fitness advice based on your goals, preferences, and progress."
        )

    with col2:
        st.markdown("### 🧠 Remembers Your Journey")
        st.write(
            "Your coach remembers your fitness history and adapts advice accordingly."
        )

    with col3:
        st.markdown("### 📊 Progress Tracking")
        st.write("Keep track of your fitness journey with tailored recommendations.")

    # Sample testimonial
    st.subheader("What Users Are Saying")
    st.markdown(
        """
        > "My AI health coach has transformed my fitness journey. It remembers my progress and gives me 
        personalized advice that actually works for my lifestyle." - Sarah K.
        """
    )

if __name__ == "__main__":
    # This section won't run in Streamlit
    pass
