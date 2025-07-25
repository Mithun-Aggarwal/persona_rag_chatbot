# streamlit_app.py

"""
The user-facing web application for the Persona-Aware RAG Chatbot.

This Streamlit application serves as the main interface for users to interact with
the agentic RAG system. It handles:
- User authentication (via API key input).
- Persona selection to tailor the agent's behavior.
- Chat history management using Streamlit's session state.
- Calling the MainAgent to process queries and generate responses.
- Displaying the final, formatted, and cited answer.
"""

import streamlit as st
import google.generativeai as genai
import logging

# Import the core agent logic
from src.agent import MainAgent

# --- Page Configuration ---
st.set_page_config(
    page_title="Persona RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Logging Configuration ---
# Helps in debugging, especially in a cloud environment
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("🤖 Persona RAG Chatbot")
    st.markdown("""
    This chatbot uses a sophisticated multi-agent system to answer questions based on a private knowledge base. 
    Select a persona to tailor its responses and retrieval strategy.
    """)

    # Persona Selection
    persona = st.radio(
        "**Choose your Persona:**",
        options=['Clinical Analyst', 'Medical Researcher', 'Patient Advocate'],
        captions=[
            "Focuses on clinical trial data, drug interactions, and treatment efficacy.",
            "Looks for mechanisms of action, novel research, and scientific evidence.",
            "Asks about patient outcomes, accessibility, and side effects in plain language."
        ],
        horizontal=False
    )
    st.divider()

    # API Key Management
    st.markdown("### Configuration")
    try:
        # Try to get the key from Streamlit's secrets management
        st.session_state.google_api_key = st.secrets["GOOGLE_API_KEY"]
        st.success("API key loaded from secrets!", icon="✅")
    except (KeyError, FileNotFoundError):
        st.warning("Google API Key not found in secrets.", icon="⚠️")
        st.session_state.google_api_key = st.text_input(
            "Enter your Google API Key:", type="password",
            help="Your key is not stored. It's only used for this session."
        )

# --- Main Application Logic ---

st.title(f"Persona: {persona}")
st.markdown("Ask a question about the knowledge base.")

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you today?"}]

# Initialize the agent in session state if it doesn't exist or if persona has changed
if "agent" not in st.session_state or st.session_state.get("current_persona") != persona:
    if st.session_state.get("google_api_key"):
        try:
            # Configure the generative AI library
            genai.configure(api_key=st.session_state.google_api_key)
            # Create and store a new agent instance for the selected persona
            st.session_state.agent = MainAgent(persona=persona)
            st.session_state.current_persona = persona
            st.toast(f"Agent activated for '{persona}' persona.", icon="🧠")
        except Exception as e:
            st.error(f"Failed to initialize the agent: {e}", icon="🚨")
            st.stop()
    else:
        st.session_state.agent = None


# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Your question..."):
    # Check if the API key and agent are ready
    if not st.session_state.get("google_api_key"):
        st.warning("Please enter your Google API Key in the sidebar to begin.", icon="🔑")
        st.stop()
        
    if not st.session_state.get("agent"):
        st.error("Agent is not initialized. Please check your API key and configuration.", icon="🚨")
        st.stop()

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        # Use a status spinner to show the agent is working
        with st.status("The agent is thinking...", expanded=True) as status:
            st.write("Analyzing intent and persona...")
            st.write("Querying knowledge graph and vector databases...")
            st.write("Synthesizing a grounded answer...")
            
            # Get the agent's response
            response = st.session_state.agent.run(prompt)
            
            # Update status and display the final answer
            status.update(label="Answer generated!", state="complete", expanded=False)
        
        st.markdown(response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})