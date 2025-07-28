# FILE: src/tools/clients.py
# V2.0: Consolidated, cached client initializers.
"""
Single source for initializing and retrieving external service clients.
Uses caching to prevent re-initialization on every call.
"""

import os
import logging
from functools import lru_cache

import pinecone
import neo4j
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# --- Client Initializers (Cached for Performance) ---

@lru_cache(maxsize=1)
def get_google_ai_client() -> genai:
    """Initializes and returns the Google AI client."""
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        genai.configure(api_key=api_key)
        logger.info("Google AI client configured successfully.")
        return genai
    except Exception as e:
        logger.error(f"Failed to configure Google AI client: {e}")
        return None

@lru_cache(maxsize=1)
def get_pinecone_index() -> pinecone.Index:
    """Initializes and returns the Pinecone index handle."""
    try:
        api_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX_NAME")
        if not api_key or not index_name:
            raise ValueError("PINECONE_API_KEY or PINECONE_INDEX_NAME not set.")
        
        # Updated initialization for latest pinecone-client versions
        pc = pinecone.Pinecone(api_key=api_key)
        index = pc.Index(index_name)
        logger.info(f"Pinecone index '{index_name}' connected successfully.")
        return index
    except Exception as e:
        logger.error(f"Failed to connect to Pinecone index: {e}")
        return None

@lru_cache(maxsize=1)
def get_neo4j_driver() -> neo4j.Driver:
    """Initializes and returns the Neo4j driver."""
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