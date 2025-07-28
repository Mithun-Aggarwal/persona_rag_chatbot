# src/tools.py

"""
Low-level tools and client initializers.

This module is responsible for setting up and providing access to external
services like databases and APIs (Pinecone, Neo4j, Google AI). It reads
credentials from environment variables. It has NO dependencies on other
modules in this project to prevent circular imports.
"""

import os
import logging
from functools import lru_cache

import pinecone
import neo4j
import google.generativeai as genai

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

        pc = pinecone.Pinecone(api_key=api_key)
        index = pc.Index(index_name)
        logger.info(f"Pinecone index '{index_name}' connected successfully.")
        return index
    except Exception as e:
        logger.error(f"Failed to connect to Pinecone index: {e}")
        return None

@lru_cache(maxsize=1)
def get_neo4j_driver() -> neo4j.GraphDatabase.driver:
    """Initializes and returns the Neo4j driver."""
    try:
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USERNAME")
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

# --- Graph Schema Utility ---

@lru_cache(maxsize=1)
def get_neo4j_schema() -> str:
    """
    Retrieves the schema from Neo4j for use in LLM prompts.
    This includes node labels, properties, and relationship types.
    """
    driver = get_neo4j_driver()
    if not driver:
        return "Error: Neo4j driver not available."

    schema_query = """
    CALL db.schema.visualization()
    """
    try:
        with driver.session() as session:
            result = session.run(schema_query)
            # The result from schema visualization is complex; we need to simplify it.
            # A simpler approach for LLM prompts is often a text-based description.
            
            # Get node labels and properties
            node_schema_query = "CALL db.labels() YIELD label CALL db.propertyKeys() YIELD propertyKey WITH label, collect(propertyKey) AS properties RETURN label, properties"
            node_props = session.run(node_schema_query).data()
            
            # Get relationship schema
            rel_schema_query = """
            MATCH (n)-[r]->(m)
            RETURN DISTINCT type(r) AS rel_type, labels(n) AS from_labels, labels(m) AS to_labels
            LIMIT 50
            """
            rel_types = session.run(rel_schema_query).data()

            schema_str = "Graph Schema:\n"
            schema_str += "Node Labels and Properties:\n"
            for item in node_props:
                 # Check if the node label exists and has properties
                if item['label'] and item['properties']:
                    # This is a simplification; a more robust version would check properties per label
                    # For now, we list all possible properties under each label for prompt context
                    # A better query: "MATCH (n:{label}) UNWIND keys(n) as key RETURN distinct key"
                    schema_str += f"- Node '{item['label']}'\n"
            
            schema_str += "\nRelationship Types and Connections:\n"
            for item in rel_types:
                from_node = item['from_labels'][0] if item['from_labels'] else "Node"
                to_node = item['to_labels'][0] if item['to_labels'] else "Node"
                schema_str += f"- ({from_node})-[:{item['rel_type']}]->({to_node})\n"

            if not node_props and not rel_types:
                return "Schema not found or database is empty."

            return schema_str

    except Exception as e:
        logger.error(f"Failed to retrieve Neo4j schema: {e}")
        return f"Error retrieving schema: {e}"