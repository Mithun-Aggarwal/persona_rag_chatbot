# src/retrievers.py

"""
Modular data retrieval functions ("retrievers").

Each function here is responsible for fetching data from a single source
(e.g., Pinecone, Neo4j) and returning it in a standardized format.
This module imports low-level clients from `src.tools`.
"""
import logging
from typing import List, Dict, Any

import neo4j
import google.generativeai as genai

from src import tools  # Correct: Import the module itself
from src.models import ContextItem, Source

logger = logging.getLogger(__name__)


def vector_search(query: str, namespace: str, top_k: int) -> List[ContextItem]:
    """Performs semantic search on a Pinecone namespace and returns standardized context."""
    embedding_client = tools.get_google_ai_client()
    pinecone_index = tools.get_pinecone_index()

    if not all([embedding_client, pinecone_index]):
        logger.error("Vector search failed: Pinecone or Google AI client not available.")
        return []

    try:
        query_embedding_result = genai.embed_content(
            model='models/embedding-001',
            content=query,
            task_type="retrieval_query"
        )
        query_embedding = query_embedding_result['embedding']

        results = pinecone_index.query(
            namespace=namespace,
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

        processed_results = []
        for match in results.get('matches', []):
            metadata = match.get('metadata', {})
            content = metadata.get('source_text_preview') or metadata.get('text', 'No content available.')

            page_numbers_raw = metadata.get("page_numbers")
            if isinstance(page_numbers_raw, str):
                try:
                    page_numbers = eval(page_numbers_raw)
                except:
                    page_numbers = []
            elif isinstance(page_numbers_raw, list):
                page_numbers = page_numbers_raw
            else:
                page_numbers = []

            source = Source(
                document_id=metadata.get("doc_id"),
                page_numbers=page_numbers,
                source_url=metadata.get("source_pdf_url"),
                retrieval_score=match.get('score', 0.0),
                type="vector_chunk"
            )
            processed_results.append(ContextItem(content=content, source=source))

        logger.info(f"Vector search in '{namespace}' with top_k={top_k} found {len(processed_results)} results.")
        return processed_results

    except Exception as e:
        logger.error(f"An error occurred during vector search in namespace '{namespace}': {e}")
        return []


# --- THIS IS THE CRITICAL FIX ---
# The helper function is defined LOCALLY within this file.
# It is NOT imported from src.tools.
def _serialize_path(path: neo4j.graph.Path) -> (str, Dict[str, Any]):
    """Helper to convert a Neo4j Path into a text representation and source dict."""
    nodes_str = []
    all_pages = set()
    all_docs = set()

    for i, node in enumerate(path.nodes):
        node_label = next(iter(node.labels), "Node")
        name = node.get('name', node.get('id', 'Unknown'))
        nodes_str.append(f"({name}:{node_label})")
        if i < len(path.relationships):
            rel = path.relationships[i]
            nodes_str.append(f"-[{rel.type}]->")

            doc_id = rel.get("doc_id")
            page_num = rel.get("page_number")
            if doc_id:
                all_docs.add(doc_id)
            if page_num:
                all_pages.add(str(page_num))

    path_repr = "".join(nodes_str)
    
    source_info = {
        "type": "graph_path",
        "document_id": ", ".join(sorted(list(all_docs))) if all_docs else "N/A",
        "page_numbers": sorted(list(all_pages))
    }
    return path_repr, source_info


def graph_search(cypher_query: str) -> List[ContextItem]:
    """Executes a Cypher query and returns standardized context."""
    driver = tools.get_neo4j_driver() # This import is correct.
    if not driver:
        logger.error("Graph search failed: Neo4j driver not available.")
        return []

    results = []
    try:
        with driver.session() as session:
            records, _, _ = session.execute_query(cypher_query)

            for record in records:
                path = record.get("p")
                if path and isinstance(path, neo4j.graph.Path):
                    # It now calls the LOCAL _serialize_path function.
                    content_str, source_info = _serialize_path(path)
                    source_info['query'] = cypher_query
                    source = Source(**source_info)
                    results.append(ContextItem(content=content_str, source=source))
                else:
                    content_str = str(record.data())
                    source = Source(type="graph_record", query=cypher_query)
                    results.append(ContextItem(content=content_str, source=source))

            logger.info(f"Graph search with Cypher query returned {len(results)} results.")
            return results
    except Exception as e:
        logger.error(f"An error occurred during graph search: {e}")
        return [ContextItem(
            content=f"Failed to execute Cypher query due to an error: {e}",
            source=Source(type="error", query=cypher_query)
        )]