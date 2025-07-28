# src/config_loader.py

import os
from dotenv import load_dotenv

# Load environment variables from a .env file if it exists
load_dotenv()

def get_env(variable_name: str, default: str = None) -> str:
    """Gets an environment variable or returns a default."""
    value = os.getenv(variable_name)
    if value is None and default is None:
        raise ValueError(f"Required environment variable '{variable_name}' is not set.")
    return value if value is not None else default

# --- API Keys and Environment ---
GOOGLE_API_KEY = get_env("GOOGLE_API_KEY")
PINECONE_API_KEY = get_env("PINECONE_API_KEY")
PINECONE_INDEX_NAME = get_env("PINECONE_INDEX_NAME", "pbac")

# --- Neo4j Database Credentials ---
NEO4J_URI = get_env("NEO4J_URI")
NEO4J_USERNAME = get_env("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = get_env("NEO4J_PASSWORD")
NEO4J_DATABASE = get_env("NEO4J_DATABASE", "neo4j")