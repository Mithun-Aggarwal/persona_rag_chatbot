# src/tools/clients.py

import os
import logging
import pinecone
import google.generativeai as genai
from neo4j import GraphDatabase, Driver

from src import config_loader

logger = logging.getLogger(__name__)

# --- Singleton instances to avoid re-initializing ---
_google_ai_client = None
_pinecone_index = None
_neo4j_driver = None

def get_google_ai_client():
    """Initializes and returns the Google Generative AI client."""
    global _google_ai_client
    if _google_ai_client is None:
        try:
            genai.configure(api_key=config_loader.GOOGLE_API_KEY)
            _google_ai_client = genai
            logger.info("Successfully initialized Google AI client.")
        except Exception as e:
            logger.error(f"Failed to initialize Google AI client: {e}")
            raise
    return _google_ai_client

def get_pinecone_index():
    """Initializes and returns the Pinecone index handle."""
    global _pinecone_index
    if _pinecone_index is None:
        try:
            pc = pinecone.Pinecone(api_key=config_loader.PINECONE_API_KEY)
            index_name = config_loader.PINECONE_INDEX_NAME
            _pinecone_index = pc.Index(index_name)
            logger.info(f"Successfully connected to Pinecone index '{index_name}'.")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone index: {e}")
            raise
    return _pinecone_index

def get_neo4j_driver() -> Driver:
    """Initializes and returns the Neo4j driver."""
    global _neo4j_driver
    if _neo4j_driver is None:
        try:
            _neo4j_driver = GraphDatabase.driver(
                config_loader.NEO4J_URI,
                auth=(config_loader.NEO4J_USERNAME, config_loader.NEO4J_PASSWORD)
            )
            _neo4j_driver.verify_connectivity()
            logger.info("Successfully connected to Neo4j database.")
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j driver: {e}")
            raise
    return _neo4j_driver

def close_neo4j_driver():
    """Closes the Neo4j driver connection if it exists."""
    global _neo4j_driver
    if _neo4j_driver:
        _neo4j_driver.close()
        _neo4j_driver = None
        logger.info("Neo4j driver closed.")