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
        # --- FIX: Read the index name from secrets instead of hardcoding it ---
        index_name = st.secrets["PINECONE_INDEX_NAME"] 
        
        if index_name not in pc.list_indexes().names():
            logging.error(f"Pinecone index '{index_name}' not found in your project.")
            st.error(f"Pinecone index '{index_name}' does not exist. Please check your configuration.", icon="🌲")
            return None

        return pc.Index(index_name)
    except Exception as e:
        logging.error(f"Failed to initialize Pinecone: {e}")
        st.error(f"Could not connect to Pinecone. Please check your credentials and configuration in secrets.toml.", icon="🌲")
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

# ... (keep all other functions and imports as they are) ...

def pinecone_search_tool(query: str, namespace: str, top_k: int = 15) -> List[Dict[str, Any]]:
    """
    Performs a semantic search on a specific Pinecone namespace.
    V1.3: FINAL version. Includes all variable definitions.
    """
    # --- FIX: Re-instating the client and index definitions ---
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
            content = metadata.get('source_text_preview') or metadata.get('text', 'No content available.')
            
            source_info = {
                "document_id": metadata.get("doc_id", "N/A"),
                "page_numbers": metadata.get("page_numbers", "N/A"),
                "source_url": metadata.get("source_pdf_url", "N/A"),
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

# ... (the rest of the file remains the same) ...

# In src/tools.py

# ... (keep other functions as they are) ...

def neo4j_graph_tool(cypher_query: str) -> List[Dict[str, Any]]:
    """
    Executes a read-only Cypher query against the Neo4j database.
    V2.0: Corrected to properly handle the Neo4j driver's Result object.
    """
    driver = get_neo4j_driver()
    if not driver:
        logging.error("Graph query failed: Neo4j driver not available.")
        return []

    results = []
    try:
        # The official way to run a query and get records
        records, _, _ = driver.execute_query(cypher_query)

        for record in records:
            path = record.get("p") # We expect the query to return a path named 'p'
            if path:
                content_str, sources = _serialize_path(path)
                results.append({"content": content_str, "source": sources})
            else:
                record_dict = record.data()
                results.append({
                    "content": str(record_dict),
                    "source": {"type": "graph_record", "query": cypher_query}
                })

        logging.info(f"Neo4j query returned {len(results)} results.")
        return results
    except Exception as e:
        logging.error(f"An error occurred during Neo4j query: {e}")
        return [{"content": f"Failed to execute Cypher query due to an error: {e}", "source": {"type": "error"}}]

# ... (the _serialize_path helper function remains the same) ...
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