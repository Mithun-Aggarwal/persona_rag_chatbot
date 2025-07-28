# streamlit_app.py

import streamlit as st
import logging
from pathlib import Path
from dotenv import load_dotenv

# --- CRITICAL: Load environment variables at the very top ---
load_dotenv()

# Now import project modules
from src.agent import MainAgent

# --- Page Configuration ---
st.set_page_config(
    page_title="Persona RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Logging ---
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - [%(name)s:%(lineno)d] - %(message)s'
)

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you today as a Clinical Analyst?"}]
if "agent" not in st.session_state:
    st.session_state.agent = None
if "current_persona" not in st.session_state:
    st.session_state.current_persona = "clinical_analyst" # Use the key from the YAML file


# --- Helper function to re-initialize agent when persona changes ---
def initialize_agent(persona_key: str):
    persona_name = persona_key.replace('_', ' ').title()
    try:
        st.session_state.agent = MainAgent(persona=persona_key)
        st.session_state.current_persona = persona_key
        st.toast(f"Agent activated for '{persona_name}' persona.", icon="🧠")
        logger.info(f"Agent initialized for persona: {persona_key}")
    except Exception as e:
        st.error(f"Failed to initialize agent for '{persona_name}': {e}", icon="🚨")
        logger.error(f"Agent initialization failed for {persona_name}: {e}")
        st.session_state.agent = None

# --- Sidebar ---
with st.sidebar:
    st.header("🤖 Persona RAG Chatbot")

    persona_options = {
        'Clinical Analyst': 'clinical_analyst',
        'Health Economist': 'health_economist',
        'Regulatory Specialist': 'regulatory_specialist',
    }
    
    # Get the display name from the key stored in session state
    persona_display_name = [k for k, v in persona_options.items() if v == st.session_state.current_persona][0]
    
    selected_persona_name = st.radio(
        "**Choose your Persona:**",
        options=persona_options.keys(),
        index=list(persona_options.keys()).index(persona_display_name)
    )

    selected_persona_key = persona_options[selected_persona_name]

    # If persona has changed, re-initialize the agent
    if selected_persona_key != st.session_state.current_persona:
        initialize_agent(selected_persona_key)
        # Reset chat for the new persona
        st.session_state.messages = [{"role": "assistant", "content": f"How can I help you today as a {selected_persona_name}?"}]
        st.rerun()

    st.divider()
    st.markdown("### 🧪 Test Questions")

    test_questions = [
        "What company sponsors Abaloparatide?",
        "Tell me about the submission for non-small cell lung cancer.",
        "Which condition does Abaloparatide treat?",
        "Was the sponsor Janssen involved in any 2025 submissions?",
        "What is the cost-effectiveness of drugs for lung cancer?", # Good for Health Economist
    ]
    
    with st.expander("Example Questions"):
        for q in test_questions:
            if st.button(q, key=q, use_container_width=True):
                st.session_state.run_prompt = q
                st.session_state.chat_input = "" # Clear input box

# --- Main Section ---
st.title(f"Persona: {selected_persona_name}")

# Initialize agent on first run
if not st.session_state.agent:
    initialize_agent(st.session_state.current_persona)

# --- Chat History Display ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Handle User Input ---
prompt_from_button = st.session_state.pop("run_prompt", None)
prompt_from_input = st.chat_input("Your question...", key="chat_input_box")
prompt = prompt_from_button or prompt_from_input

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if st.session_state.agent:
            with st.spinner(f"Thinking as a {selected_persona_name}..."):
                try:
                    response = st.session_state.agent.run(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_message = f"An unexpected error occurred: {e}"
                    st.error(error_message)
                    logger.error(f"Error during agent run: {e}", exc_info=True)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
        else:
            st.error("The agent is not initialized. Please check the logs for errors.")

    # Rerun to clear the input box if a button was used, or to reflect the new state
    st.rerun()