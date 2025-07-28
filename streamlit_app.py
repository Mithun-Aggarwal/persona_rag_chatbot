# FILE: streamlit_app.py
# V2.2: UI updated with categorized test questions for robust evaluation.

import streamlit as st
import logging
from dotenv import load_dotenv

# --- CRITICAL: Load environment variables at the very top ---
# This ensures all modules can access them when imported.
load_dotenv()

# Now import project modules
from src.agent import Agent
from src.tools.clients import get_google_ai_client # Used for a pre-flight check

# --- Page and Logging Configuration ---
st.set_page_config(
    page_title="Persona RAG Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - [%(name)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

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
    # Pre-flight check for API key to provide a better error message.
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
    st.markdown("Use these questions to test the agent's capabilities.")

    # --- NEW: Categorized Test Questions ---
    with st.expander("🎯 Fact Retrieval (High Precision)"):
        questions = {
            "Graph-based query": "What company sponsors the drug Abaloparatide?",
            "Specific indication": "Which condition is Amivantamab intended to treat?",
            "Dosage form lookup": "What is the dosage form of Daratumumab?",
            "Sponsor for combination": "Which sponsor is associated with the combination of Dabrafenib and Trametinib?",
        }
        for name, q in questions.items():
            if st.button(f"{name}: {q}", key=q, use_container_width=True):
                st.session_state.run_prompt = q

    with st.expander("⚖️ Comparative Analysis"):
        questions = {
            "Compare two drugs for NSCLC": "Compare Amivantamab and Osimertinib for non-small cell lung cancer.",
            "Compare submissions by date": "What were the key differences in submissions between the March and July 2025 PBAC meetings?",
        }
        for name, q in questions.items():
            if st.button(q, key=q, use_container_width=True):
                st.session_state.run_prompt = q
    
    with st.expander("📋 Summarization"):
        questions = {
            "Summarize a drug's submission": "Provide a summary of the PBAC submission for Alectinib.",
            "Summarize a meeting": "Summarize all submissions related to oncology in the May 2025 meeting.",
        }
        for name, q in questions.items():
            if st.button(q, key=q, use_container_width=True):
                st.session_state.run_prompt = q

    with st.expander("🤔 Challenging / Ambiguous Questions"):
        questions = {
            "Test fallback logic": "What is the price of Abaloparatide?", # Price is explicitly excluded from consumer comments
            "Broad, multi-hop query": "Find sponsors who made submissions for both lung cancer and melanoma in 2025.",
            "Out of scope": "What are the latest FDA guidelines?", # Tests if it hallucinates or admits it doesn't know
        }
        for name, q in questions.items():
            if st.button(q, key=q, use_container_width=True):
                st.session_state.run_prompt = q

# --- Main Chat Interface ---
st.title("Persona-Aware RAG Agent")
st.caption(f"Currently acting as: **{selected_persona_name}**")

# Initialize agent on first run and handle potential failure.
if st.session_state.agent is None:
    st.session_state.agent = initialize_agent()
    # If this is the first run, set the initial message.
    if not st.session_state.messages:
        reset_chat(selected_persona_name)

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# Handle user input from both chat box and sidebar buttons
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
    
    # Rerun to clear input box if a button was used
    if prompt_from_button:
        st.rerun()