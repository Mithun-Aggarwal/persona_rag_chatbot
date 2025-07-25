# streamlit_app.py

import streamlit as st
import google.generativeai as genai
import logging
from src.agent import MainAgent

# --- Page Configuration ---
st.set_page_config(
    page_title="Persona RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you today?"}]
if "agent" not in st.session_state:
    st.session_state.agent = None
if "current_persona" not in st.session_state:
    st.session_state.current_persona = "Clinical Analyst"
if "google_api_key" not in st.session_state:
    st.session_state.google_api_key = None

# --- NEW: Define Test Questions ---
test_questions = {
    "✅ Positive Tests (Info is Present)": [
        "What submissions were made by AstraZeneca in the December 2022 PBAC intracycle meeting agenda?",
        "What is the exact patient population for which Trastuzumab Deruxtecan is being considered?",
        "What is the submission type for Ibrutinib, and what is its trade name?",
    ],
    "❌ Negative Tests (Info is NOT Present)": [
        "Was the resubmission for Enhertu (Trastuzumab Deruxtecan) approved by the PBAC?",
        "What is the cost-effectiveness ratio or price submitted for Tixagevimab and Cilgavimab (Evusheld)?",
        "What information does this document have about Pembrolizumab (Keytruda)?",
    ]
}

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("🤖 Persona RAG Chatbot")
    st.markdown("This chatbot uses a sophisticated multi-agent system to answer questions based on a private knowledge base. Select a persona to tailor its responses and retrieval strategy.")

    persona_options = ['Clinical Analyst', 'Health Economist', 'Regulatory Specialist']
    try:
        current_index = persona_options.index(st.session_state.current_persona)
    except ValueError:
        current_index = 0

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
    st.divider()

    # --- NEW: Test Questions UI ---
    st.markdown("### 🧪 Test Questions")
    st.markdown("Click a button to run a pre-defined test query.")
    for category, questions in test_questions.items():
        with st.expander(category):
            for question in questions:
                # When a button is clicked, we set the question text to a session_state variable.
                # We use the question itself as the key to ensure uniqueness.
                if st.button(question, key=question, use_container_width=True):
                    st.session_state.run_prompt = question

# --- Main Application Logic ---

st.title(f"Persona: {persona}")
st.markdown("Ask a question about the knowledge base.")

# Initialize the agent
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
                st.error(f"Failed to initialize the agent: {e}", icon="🚨")
                st.session_state.agent = None
    else:
        st.session_state.agent = None

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- MODIFIED: Handle prompt from either button click or chat input ---
# Check if a test button was clicked
if "run_prompt" in st.session_state:
    prompt = st.session_state.run_prompt
    del st.session_state.run_prompt  # Clear the state so it doesn't run again
else:
    # Otherwise, wait for user input in the chat box
    prompt = st.chat_input("Your question...")

# Process the prompt if one exists
if prompt:
    if not st.session_state.google_api_key or not st.session_state.agent:
        st.warning("Please ensure your API key is set and the agent is initialized.", icon="🔑")
        st.stop()

    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get and display assistant response
    with st.chat_message("assistant"):
        with st.status("The agent is thinking...", expanded=True) as status:
            st.write("Analyzing intent and persona...")
            st.write("Querying specialist data tools...")
            st.write("Synthesizing a grounded answer...")
            response = st.session_state.agent.run(prompt)
            status.update(label="Answer generated!", state="complete", expanded=False)
        st.markdown(response)

    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response})
    # Rerun to show the new messages immediately, especially for button clicks
    st.rerun()