# FILE: src/tools/clients.py
# V2.4 (Critical Error Fix): Correctly implemented the Retry policy using a `predicate` function.

import os
import logging
from functools import lru_cache

import pinecone
import neo4j
import google.generativeai as genai
from google.generativeai.client import get_default_generative_client
from google.api_core.retry import Retry
from google.api_core.client_options import ClientOptions
from google.api_core import exceptions as google_exceptions
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# --- START OF DEFINITIVE FIX: Correct Retry Implementation ---

def is_service_unavailable(exc: Exception) -> bool:
    """Predicate function to check if an exception is a ServiceUnavailable error."""
    return isinstance(exc, google_exceptions.ServiceUnavailable)

# Define a more robust retry strategy for API calls.
DEFAULT_RETRY = Retry(
    # Use the predicate function to decide if we should retry.
    predicate=is_service_unavailable,
    initial=1.0,      # Start with a 1-second delay
    maximum=10.0,     # Maximum delay of 10 seconds
    multiplier=2.0,   # Double the delay each time
    deadline=30.0,    # Total deadline for all retries, including the initial call
)

# This remains the same.
DEFAULT_REQUEST_OPTIONS = {"retry": DEFAULT_RETRY, "timeout": 15.0}

# --- END OF DEFINITIVE FIX ---


# --- Client Initializers (Cached for Performance) ---

@lru_cache(maxsize=1)
def get_google_ai_client() -> genai:
    """Initializes and returns the Google AI client."""
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        genai.configure(api_key=api_key)
        get_default_generative_client()._client_options = ClientOptions(api_endpoint="generativelanguage.googleapis.com")
        
        logger.info("Google AI client configured successfully.")
        return genai
    except Exception as e:
        logger.error(f"Failed to configure Google AI client: {e}")
        return None

# --- Model Getters (No changes needed here) ---

@lru_cache(maxsize=2)
def get_generative_model(model_name: str = 'gemini-1.5-pro-latest') -> genai.GenerativeModel:
    client = get_google_ai_client()
    if not client: return None
    logger.info(f"Requesting GenerativeModel: {model_name}")
    return client.GenerativeModel(model_name)

@lru_cache(maxsize=2)
def get_flash_model(model_name: str = 'gemini-1.5-flash-latest') -> genai.GenerativeModel:
    client = get_google_ai_client()
    if not client: return None
    logger.info(f"Requesting Flash Model: {model_name}")
    return client.GenerativeModel(model_name)


@lru_cache(maxsize=1)
def get_pinecone_index() -> pinecone.Index:
    try:
        api_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX_NAME")
        if not api_key or not index_name:
            raise ValueError("PINECONE_API_KEY or PINECONE_INDEX_NAME not set.")
        
        pc = pinecone.Pinecone(api_key=api_key)
        index = pc.Index(index_name)
        logger.info(f"Pinecone index '{index_name}' connected successfully.")
        return index
    except Exception as e:
        logger.error(f"Failed to connect to Pinecone index: {e}")
        return None

@lru_cache(maxsize=1)
def get_neo4j_driver() -> neo4j.Driver:
    try:
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD")
        if not all([uri, user, password]):
            raise ValueError("Neo4j connection details (URI, USERNAME, PASSWORD) not set.")

        driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        logger.info("Neo4j driver connected successfully.")
        return driver
    except Exception as e:
        logger.error(f"Failed to create Neo4j driver: {e}")
        return None