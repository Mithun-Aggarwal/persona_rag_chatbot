# FILE: streamlit_app.py
# V2.4: Updated evaluation questions to align with 2024 data and added more open-ended queries.

import streamlit as st
import logging
import os
from dotenv import load_dotenv

# --- CRITICAL: Load environment variables at the very top ---
load_dotenv()

# --- Page and Logging Configuration ---
st.set_page_config(
    page_title="Persona RAG Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Configure logging for YOUR application. We want to see INFO messages.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - [%(name)s:%(lineno)d] - %(message)s'
)

# Tame the noisy loggers from the Streamlit framework and other libraries.
logging.getLogger('streamlit').setLevel(logging.WARNING)
logging.getLogger('watchdog').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Now import project modules
from src.agent import Agent
from src.tools.clients import get_google_ai_client # Used for a pre-flight check


# --- Session State Initialization ---
if "agent" not in st.session_state:
    st.session_state.agent = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_persona" not in st.session_state:
    st.session_state.current_persona = "clinical_analyst"

# --- Helper Functions ---
@st.cache_resource
def initialize_agent():
    """
    Initializes the agent once and caches it.
    Returns the agent instance or None if initialization fails.
    """
    if not get_google_ai_client():
        st.error("Google API Key is not configured. Please set the GOOGLE_API_KEY in your .env file.", icon="🚨")
        return None
    try:
        agent = Agent()
        logger.info("Unified agent initialized successfully and cached for the session.")
        return agent
    except Exception as e:
        st.error(f"Fatal error during agent initialization: {e}", icon="🚨")
        logger.error(f"Agent initialization failed: {e}", exc_info=True)
        return None

def reset_chat(persona_name: str):
    """Resets the chat history for a new conversation."""
    st.session_state.messages = [
        {"role": "assistant", "content": f"Hi! I'm now acting as a **{persona_name}**. How can I help you?"}
    ]

# --- Sidebar ---
with st.sidebar:
    st.header("🤖 Persona RAG Chatbot")
    st.markdown("Select a persona to tailor my retrieval strategy and answers to your specific role.")

    persona_options = {
        'Clinical Analyst': 'clinical_analyst',
        'Health Economist': 'health_economist',
        'Regulatory Specialist': 'regulatory_specialist',
    }
    
    current_display_name = [k for k, v in persona_options.items() if v == st.session_state.current_persona][0]
    
    selected_persona_name = st.radio(
        "**Choose your Persona:**",
        options=persona_options.keys(),
        index=list(persona_options.keys()).index(current_display_name),
        key="persona_selector"
    )
    
    selected_persona_key = persona_options[selected_persona_name]

    if selected_persona_key != st.session_state.current_persona:
        st.session_state.current_persona = selected_persona_key
        reset_chat(selected_persona_name)
        st.rerun()

    st.divider()
    if st.button("🔄 Clear Chat History", use_container_width=True):
        reset_chat(selected_persona_name)
        st.rerun()

    st.divider()
    st.header("🧪 Evaluation Questions")
    st.markdown("Use these questions to test the agent's capabilities with the 2024 data.")

    # --- START OF DEFINITIVE FIX: Updated Evaluation Questions for 2024 data ---
    with st.expander("🎯 Fact Retrieval (High Precision)", expanded=True):
        questions = {
            "Sponsor Lookup (Dec 2024)": "Who is the sponsor for Esketamine?",
            "Indication Lookup (July 2024)": "What condition is Belzutifan used to treat?",
            "Trade Name Lookup (March 2024)": "What is the trade name for Aflibercept?",
            "Dosage Form (May 2024)": "What is the dosage form of the Respiratory Syncytial Virus Vaccine?",
        }
        for name, q in questions.items():
            if st.button(f"{name}: {q}", key=q, use_container_width=True):
                st.session_state.run_prompt = q

    with st.expander("⚖️ Comparative Analysis"):
        questions = {
            "Compare drugs for same condition": "Compare Aflibercept and Adalimumab for eye-related conditions.",
            "Compare submissions across meetings": "What drugs were submitted by Bayer in the May 2024 and March 2024 meetings?",
            "Open-ended comparison": "What are the differences between a 'New PBS listing' and a 'Change to existing listing' based on the agenda documents?",
        }
        for name, q in questions.items():
            if st.button(q, key=q, use_container_width=True):
                st.session_state.run_prompt = q
    
    with st.expander("📋 Summarization"):
        questions = {
            "Summarize a meeting": "Provide a summary of the key submissions from the May 2024 intracycle meeting.",
            "Summarize by theme": "Summarize all submissions related to cancer treatment across all 2024 documents.",
            "Open-ended synthesis": "Based on the agendas, what appears to be the main focus of the PBAC's work in 2024?",
        }
        for name, q in questions.items():
            if st.button(q, key=q, use_container_width=True):
                st.session_state.run_prompt = q

    with st.expander("🤔 Challenging / Ambiguous Questions"):
        questions = {
            "Test data boundaries": "What was the final PBAC decision on Fruquintinib from the September 2024 meeting?",
            "Broad, multi-hop query": "Find sponsors who made submissions for both auto-immune diseases and cancer in 2024.",
            "Test fallback logic (no price)": "What is the price of Aflibercept?",
            "Out of scope (external knowledge)": "What are the latest EMA guidelines for vaccine submissions?",
        }
        for name, q in questions.items():
            if st.button(q, key=q, use_container_width=True):
                st.session_state.run_prompt = q
    # --- END OF DEFINITIVE FIX ---

# --- Main Chat Interface ---
st.title("Persona-Aware RAG Agent")
st.caption(f"Currently acting as: **{selected_persona_name}**")

if st.session_state.agent is None:
    st.session_state.agent = initialize_agent()
    if not st.session_state.messages:
        reset_chat(selected_persona_name)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

prompt_from_button = st.session_state.pop("run_prompt", None)
prompt_from_input = st.chat_input("Ask your question...")
prompt = prompt_from_button or prompt_from_input

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if st.session_state.agent:
            with st.spinner(f"Thinking as a {selected_persona_name}..."):
                response = st.session_state.agent.run(prompt, persona=st.session_state.current_persona)
                st.markdown(response, unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            st.error("Agent is not available due to an initialization error. Please check the terminal logs.")
            st.stop()
    
    if prompt_from_button:
        st.rerun()