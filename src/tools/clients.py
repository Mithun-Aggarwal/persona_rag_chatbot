# FILE: src/tools/clients.py
# V2.5 (Re-ranker): Added a client for the Cohere Re-rank API.

import os
import logging
from functools import lru_cache

import pinecone
import neo4j
import cohere # <-- NEW IMPORT
import google.generativeai as genai
from google.generativeai.client import get_default_generative_client
from google.api_core.retry import Retry
from google.api_core.client_options import ClientOptions
from google.api_core import exceptions as google_exceptions
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# --- Resilience Configuration (Unchanged) ---
def is_service_unavailable(exc: Exception) -> bool:
    return isinstance(exc, google_exceptions.ServiceUnavailable)

DEFAULT_RETRY = Retry(
    predicate=is_service_unavailable,
    initial=1.0, maximum=10.0, multiplier=2.0, deadline=30.0,
)
DEFAULT_REQUEST_OPTIONS = {"retry": DEFAULT_RETRY, "timeout": 15.0}


# --- Client Initializers ---

# --- NEW: Cohere Client Initializer ---
@lru_cache(maxsize=1)
def get_cohere_client() -> cohere.Client:
    """Initializes and returns the Cohere client."""
    try:
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY environment variable not set.")
        co = cohere.Client(api_key)
        logger.info("Cohere client configured successfully.")
        return co
    except Exception as e:
        logger.error(f"Failed to configure Cohere client: {e}")
        return None

@lru_cache(maxsize=1)
def get_google_ai_client() -> genai:
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key: raise ValueError("GOOGLE_API_KEY environment variable not set.")
        genai.configure(api_key=api_key)
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
    return client.GenerativeModel(model_name)

@lru_cache(maxsize=2)
def get_flash_model(model_name: str = 'gemini-1.5-flash-latest') -> genai.GenerativeModel:
    client = get_google_ai_client()
    if not client: return None
    return client.GenerativeModel(model_name)

@lru_cache(maxsize=1)
def get_pinecone_index() -> pinecone.Index:
    try:
        api_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX_NAME")
        if not api_key or not index_name: raise ValueError("PINECONE_API_KEY or PINECONE_INDEX_NAME not set.")
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
        if not all([uri, user, password]): raise ValueError("Neo4j connection details not set.")
        driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        logger.info("Neo4j driver connected successfully.")
        return driver
    except Exception as e:
        logger.error(f"Failed to create Neo4j driver: {e}")
        return None