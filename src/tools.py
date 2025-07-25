# src/tools.py

"""
Specialized Data Retrieval Tools for the RAG Agent.

This module contains the functions that form the "second brain" of the agent.
Each function is a "tool" that the main agent can call to retrieve specific
information from a backend data source (Pinecone Vector DB, Neo4j Graph DB).

Key Principles:
1.  **Modularity**: Each tool is self-contained and handles one specific data source.
2.  **Efficiency**: Connections to databases are cached using Streamlit's resource caching
    to avoid re-establishing connections on every run.
3.  **Provenance**: Every piece of context retrieved is bundled with its source
    metadata (e.g., document ID, page number, URL). This is crucial for citation
    and explainability.
4.  **Security**: All API keys and credentials are securely accessed via st.secrets.
"""

import streamlit as st
import pinecone
import neo4j
import google.generativeai as genai
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONNECTION INITIALIZATION (CACHED) ---

@st.cache_resource
def get_google_ai_client():
    """Initializes and returns a Google Generative AI client."""
    try:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        # Model for embedding generation
        model = genai.GenerativeModel('models/embedding-001')
        return model
    except Exception as e:
        logging.error(f"Failed to initialize Google AI client: {e}")
        st.error("Could not connect to Google AI. Please check your API key.", icon="🚨")
        return None

@st.cache_resource
def get_pinecone_index():
    """Initializes and returns a connection to the Pinecone index."""
    try:
        pc = pinecone.Pinecone(
            api_key=st.secrets["PINECONE_API_KEY"]
        )
        # Note: Your index name should be stored in secrets or config.
        # For this example, we'll hardcode it but in a real app, externalize it.
        index_name = "pbac-main-index" 
        return pc.Index(index_name)
    except Exception as e:
        logging.error(f"Failed to initialize Pinecone: {e}")
        st.error("Could not connect to Pinecone. Please check your credentials.", icon="🌲")
        return None

@st.cache_resource
def get_neo4j_driver():
    """Initializes and returns a Neo4j driver instance."""
    try:
        driver = neo4j.GraphDatabase.driver(
            st.secrets["NEO4J_URI"],
            auth=(st.secrets["NEO4J_USERNAME"], st.secrets["NEO4J_PASSWORD"])
        )
        # Verify connection
        driver.verify_connectivity()
        logging.info("Neo4j driver initialized successfully.")
        return driver
    except Exception as e:
        logging.error(f"Failed to initialize Neo4j driver: {e}")
        st.error("Could not connect to the Neo4j database. Please check your credentials.", icon="🕸️")
        return None

# --- SPECIALIST TOOLS ---

def pinecone_search_tool(query: str, namespace: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Performs a semantic search on a specific Pinecone namespace.

    Args:
        query (str): The user's natural language query.
        namespace (str): The Pinecone namespace to search within (e.g., 'pbac-text').
        top_k (int): The number of results to return.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                              contains 'content' and 'source' metadata.
    """
    embedding_client = get_google_ai_client()
    pinecone_index = get_pinecone_index()

    if not all([embedding_client, pinecone_index]):
        logging.error("Search failed: Pinecone or Google AI client not available.")
        return []

    try:
        # 1. Create embedding for the user's query
        query_embedding_result = genai.embed_content(
            model='models/embedding-001',
            content=query,
            task_type="retrieval_query"
        )
        query_embedding = query_embedding_result['embedding']

        # 2. Query Pinecone
        results = pinecone_index.query(
            namespace=namespace,
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

        # 3. Process and format results with provenance
        processed_results = []
        for match in results.get('matches', []):
            metadata = match.get('metadata', {})
            content = metadata.pop('text', 'No content available.') # Extract text, remove from metadata dict
            
            # Ensure a clean source dictionary
            source_info = {
                "document_id": metadata.get("document_id", "N/A"),
                "page_number": metadata.get("page_number", "N/A"),
                "source_url": metadata.get("source_url", "N/A"),
                "retrieval_score": match.get('score', 0.0)
            }
            
            processed_results.append({
                "content": content,
                "source": source_info
            })
            
        logging.info(f"Pinecone search in namespace '{namespace}' returned {len(processed_results)} results.")
        return processed_results

    except Exception as e:
        logging.error(f"An error occurred during Pinecone search: {e}")
        return []

def neo4j_graph_tool(cypher_query: str) -> List[Dict[str, Any]]:
    """
    Executes a read-only Cypher query against the Neo4j database.

    Note: This tool expects a valid Cypher query. The main agent is responsible
    for generating this query from the user's natural language input.

    Args:
        cypher_query (str): The Cypher query to execute.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each represents a
                              result from the graph, formatted for an LLM.
    """
    driver = get_neo4j_driver()
    if not driver:
        logging.error("Graph query failed: Neo4j driver not available.")
        return []

    results = []
    try:
        with driver.session(database="neo4j") as session:
            # Using a read transaction for safety
            records, summary, keys = session.run(cypher_query)
            
            for record in records:
                # We expect the query to return a path or nodes/relationships
                # that can be serialized into a text format.
                # This part is highly dependent on your graph schema and query design.
                # Example: a query might `RETURN path`
                path = record.get("path")
                if path:
                    # Serialize the path into a human-readable string
                    content_str, sources = _serialize_path(path)
                    results.append({"content": content_str, "source": sources})
                else:
                    # Fallback for other types of record
                    record_dict = record.data()
                    results.append({
                        "content": str(record_dict),
                        "source": {"type": "graph_record", "query": cypher_query}
                    })

        logging.info(f"Neo4j query returned {len(results)} results.")
        return results
    except Exception as e:
        logging.error(f"An error occurred during Neo4j query: {e}")
        # Return the error message as context so the agent knows the query failed.
        return [{"content": f"Failed to execute Cypher query due to an error: {e}", "source": {"type": "error"}}]

def _serialize_path(path: neo4j.graph.Path) -> (str, Dict):
    """
    Helper function to convert a Neo4j Path object into a text representation
    and extract source metadata.
    """
    nodes_str = []
    rels_str = []
    sources = {"nodes": [], "relationships": []}

    for node in path.nodes:
        node_props = dict(node)
        node_label = next(iter(node.labels), "Node")
        nodes_str.append(f"({node_label} {{name: '{node_props.get('name', 'Unknown')}'}})")
        sources["nodes"].append(node_props)

    for rel in path.relationships:
        rel_props = dict(rel)
        # Critical: Extracting provenance from the relationship properties
        source_preview = rel_props.pop('source_text_preview', 'N/A')
        page_number = rel_props.pop('page_number', 'N/A')
        
        rel_str = f"-[{rel.type} {rel_props}]->"
        rels_str.append(rel_str)
        sources["relationships"].append({
            "type": rel.type,
            "page": page_number,
            "preview": source_preview
        })

    # Weave nodes and relationships together
    full_path_str = nodes_str[0]
    for i, rel_s in enumerate(rels_str):
        full_path_str += rel_s + nodes_str[i+1]

    # Combine sources into a single representative source for this path
    combined_source = {
        "type": "graph_path",
        "primary_source": sources["relationships"][0] if sources["relationships"] else "N/A"
    }
    return full_path_str, combined_source