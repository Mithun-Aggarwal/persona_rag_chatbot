# FILE: streamlit_app.py
# V3.1 (Definitive Fix): Corrected the agent call to be synchronous, fixing the asyncio ValueError.

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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - [%(name)s:%(lineno)d] - %(message)s'
)
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
    st.session_state.current_persona = "automatic"

# --- Helper Functions ---
@st.cache_resource
def initialize_agent():
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
    display_name = "Automatic" if persona_name == "Automatic (Recommended)" else persona_name
    st.session_state.messages = [
        {"role": "assistant", "content": f"Hi! I'm now acting in **{display_name}** mode. How can I help you?"}
    ]

# --- Sidebar ---
with st.sidebar:
    st.header("🤖 Persona RAG Chatbot")
    st.markdown("Select a persona to tailor my retrieval strategy and answers to your specific role.")
    persona_options = {
        'Automatic (Recommended)': 'automatic',
        'Clinical Analyst': 'clinical_analyst',
        'Health Economist': 'health_economist',
        'Regulatory Specialist': 'regulatory_specialist',
    }
    current_display_name = [k for k, v in persona_options.items() if v == st.session_state.current_persona][0]
    selected_persona_name = st.radio(
        "**Choose your Mode:**",
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
    st.markdown("Use these questions to test the agent's capabilities with the 2025 data.")

    with st.expander("🎯 Fact Retrieval (2025 Data)", expanded=True):
        questions = {
            "Sponsor Lookup (July 2025)": "What company is the sponsor for Abaloparatide?",
            "Indication Lookup (March 2025)": "What is Amivantamab used to treat?",
            "Trade Name Lookup (May 2025)": "What is the trade name for Dostarlimab?",
            "Dosage Form (July 2025)": "What is the dosage form of Apomorphine?",
        }
        for name, q in questions.items():
            if st.button(f"{name}: {q}", key=q, use_container_width=True):
                st.session_state.run_prompt = q

    with st.expander("⚖️ Multi-Step Reasoning (2025 Data)"):
        questions = {
            "Intersection Query": "Which sponsors made submissions in both the March 2025 and July 2025 meetings?",
            "Comparative Query": "Compare the submission purposes for Acalabrutinib and Alectinib in the 2025 meetings.",
            "Multi-Hop Query": "What is the indication for the drug whose trade name is Cabometyx?",
        }
        for name, q in questions.items():
            if st.button(q, key=q, use_container_width=True):
                st.session_state.run_prompt = q
    
    with st.expander("📋 Summarization (2025 Data)"):
        questions = {
            "Summarize a Meeting": "Provide a summary of the key submissions from the May 2025 PBAC meeting.",
            "Summarize by Theme": "Summarize all submissions related to oncology in the March 2025 documents.",
            "Synthesize High-Level Goal": "Based on the agendas, what appears to be the main focus of the PBAC's work in 2025?",
        }
        for name, q in questions.items():
            if st.button(q, key=q, use_container_width=True):
                st.session_state.run_prompt = q

    with st.expander("🤔 Challenging / Ambiguous Questions"):
        questions = {
            "Test Data Boundaries": "What was the final PBAC decision on Ribociclib from the July 2025 meeting?",
            "Test Fallback Logic (No Price)": "What is the price of Tirzepatide?",
            "Out of Scope (External Knowledge)": "What are the latest FDA guidelines on biosimilars?",
        }
        for name, q in questions.items():
            if st.button(q, key=q, use_container_width=True):
                st.session_state.run_prompt = q

# --- Main Chat Interface ---
st.title("Persona-Aware RAG Agent")
if st.session_state.current_persona == "automatic":
    st.caption("Currently in **Automatic Mode** (selects best persona per query)")
else:
    st.caption(f"Currently acting as: **{selected_persona_name}**")
with st.container(border=True):
    st.info("""
    **Welcome! This is an advanced chatbot designed to answer questions about pharmaceutical and regulatory documents.** 
    
    Its unique feature is the ability to tailor its information retrieval strategy based on the professional role you select in the sidebar.

    **How to use this demo:**
    1.  **Choose Your Mode:** Select 'Automatic' (Recommended) or a specific persona from the sidebar.
    2.  **Ask a Question:** Use the pre-defined 'Evaluation Questions' or type your own question in the chat box below.
    """)
    st.markdown("<p style='text-align: center; color: grey;'>A not-for-profit demonstration project by <b>EVIL_MIT</b></p>", unsafe_allow_html=True)

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
            spinner_text = "Thinking..."
            if st.session_state.current_persona != "automatic":
                 spinner_text = f"Thinking as a {selected_persona_name}..."
            
            with st.spinner(spinner_text):
                history_for_rewrite = [f"{m['role']}: {m['content']}" for m in st.session_state.messages[-5:-1]]
                
                # --- START OF DEFINITIVE FIX ---
                # REMOVED asyncio.run() as agent.run is now a synchronous function.
                response = st.session_state.agent.run(
                    prompt, 
                    persona=st.session_state.current_persona,
                    chat_history=history_for_rewrite
                )
                # --- END OF DEFINITIVE FIX ---
                
                st.markdown(response, unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            st.error("Agent is not available due to an initialization error. Please check the terminal logs.")
            st.stop()
    
    if prompt_from_button:
        st.rerun()