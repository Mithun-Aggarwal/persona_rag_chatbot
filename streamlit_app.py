# streamlit_app.py

import streamlit as st
import google.generativeai as genai
import logging
from src.agent import MainAgent

st.set_page_config(
    page_title="Persona RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Robust Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you today?"}]
if "agent" not in st.session_state:
    st.session_state.agent = None
# Set a default persona that exists in the YAML
if "current_persona" not in st.session_state:
    st.session_state.current_persona = "Clinical Analyst"
if "google_api_key" not in st.session_state:
    st.session_state.google_api_key = None

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("🤖 Persona RAG Chatbot")
    st.markdown("This chatbot uses a sophisticated multi-agent system to answer questions based on a private knowledge base. Select a persona to tailor its responses and retrieval strategy.")

    # Persona names now match the YAML file (they will be normalized in the router)
    persona_options = ['Clinical Analyst', 'Health Economist', 'Regulatory Specialist']
    
    # Safely get the index for the radio button
    try:
        current_index = persona_options.index(st.session_state.current_persona)
    except ValueError:
        current_index = 0 # Default to the first option if the stored persona isn't in the list

    persona = st.radio(
        "**Choose your Persona:**",
        options=persona_options,
        index=current_index
    )
    st.divider()

    st.markdown("### Configuration")
    if st.secrets.get("GOOGLE_API_KEY"):
        st.session_state.google_api_key = st.secrets["GOOGLE_API_KEY"]
        st.success("API key loaded from secrets!", icon="✅")
    else:
        st.warning("Google API Key not found in secrets.", icon="⚠️")
        st.session_state.google_api_key = st.text_input(
            "Enter your Google API Key:", type="password", help="Your key is not stored."
        )

# --- Main Application Logic ---
st.title(f"Persona: {persona}")
st.markdown("Ask a question about the knowledge base.")

# Re-initialize the agent ONLY if the persona has changed or the agent doesn't exist.
if st.session_state.current_persona != persona or st.session_state.agent is None:
    if st.session_state.google_api_key:
        with st.spinner(f"Initializing agent for {persona}..."):
            try:
                genai.configure(api_key=st.session_state.google_api_key)
                st.session_state.agent = MainAgent(persona=persona)
                st.session_state.current_persona = persona
                if st.session_state.agent:
                    st.toast(f"Agent activated for '{persona}' persona.", icon="🧠")
            except Exception as e:
                # Display the actual error that stopped initialization
                st.error(f"Failed to initialize the agent: {e}", icon="🚨")
                st.session_state.agent = None
    else:
        st.session_state.agent = None

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Your question..."):
    if not st.session_state.google_api_key:
        st.warning("Please enter your Google API Key in the sidebar to begin.", icon="🔑")
        st.stop()
    if not st.session_state.agent:
        st.error("Agent is not initialized. Please verify your API key and configuration, then refresh.", icon="🚨")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("The agent is thinking...", expanded=True) as status:
            st.write("Analyzing intent and persona...")
            st.write("Querying specialist data tools...")
            st.write("Synthesizing a grounded answer...")
            response = st.session_state.agent.run(prompt)
            status.update(label="Answer generated!", state="complete", expanded=False)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})