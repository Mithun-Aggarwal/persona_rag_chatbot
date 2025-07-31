# FILE: src/tools/clients.py
# V3.0 (Final Version): Removed all unnecessary Cohere client code.

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

# --- Resilience Configuration ---
def is_service_unavailable(exc: Exception) -> bool:
    """Predicate function to check if an exception is a ServiceUnavailable error."""
    return isinstance(exc, google_exceptions.ServiceUnavailable)

DEFAULT_RETRY = Retry(
    predicate=is_service_unavailable,
    initial=1.0,      # Start with a 1-second delay
    maximum=10.0,     # Maximum delay of 10 seconds
    multiplier=2.0,   # Double the delay each time
    deadline=30.0,    # Total deadline for all retries
)

DEFAULT_REQUEST_OPTIONS = {"retry": DEFAULT_RETRY, "timeout": 15.0}


# --- Client Initializers (Cached for Performance) ---

@lru_cache(maxsize=1)
def get_google_ai_client() -> genai:
    """Initializes and returns the Google AI client."""
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        genai.configure(api_key=api_key)
        # This line helps prevent certain region-based connection issues.
        get_default_generative_client()._client_options = ClientOptions(api_endpoint="generativelanguage.googleapis.com")
        logger.info("Google AI client configured successfully.")
        return genai
    except Exception as e:
        logger.error(f"Failed to configure Google AI client: {e}")
        return None

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
    """Initializes and returns the Pinecone index client."""
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
    """Initializes and returns the Neo4j graph database driver."""
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