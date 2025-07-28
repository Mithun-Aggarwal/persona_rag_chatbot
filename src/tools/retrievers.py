# FILE: src/tools/retrievers.py
# V3.5 (Definitive Fix 2): Uses explicit relationship properties map from Cypher for robustness.

import logging
from typing import List, Dict, Any
import neo4j

from src.tools.clients import get_google_ai_client, get_pinecone_index, get_neo4j_driver
from src.models import ToolResult, QueryMetadata
from src.prompts import CYPHER_GENERATION_PROMPT

logger = logging.getLogger(__name__)

# --- Helper Functions ---

def _format_pinecone_results(matches: List[dict]) -> List[str]:
    """Formats Pinecone results into a clean string with an HTML citation."""
    contents = []
    for match in matches:
        metadata = match.get('metadata', {})
        text = metadata.get('text', 'No content available.')
        doc_id = metadata.get('doc_id', 'Unknown Document')
        page_num = metadata.get('page_number')
        url = metadata.get('source_pdf_url', '#')
        link_url = f"{url}#page={page_num}" if page_num else url
        citation_text = f"{doc_id} (Page {page_num or 'N/A'})"
        citation = f'<a href="{link_url}" target="_blank">{citation_text}</a>'
        contents.append(f"Evidence from document: {text}\nCitation: {citation}")
    return contents

# --- START OF DEFINITIVE FIX ---
def _serialize_graph_record(record: Dict[str, Any]) -> str:
    """
    Robustly serializes a single graph record into a citable string.
    It expects the record to contain a 'p' key for the path and a 'rel_props' key
    for the relationship properties map.
    """
    path_data = record.get("p")
    rel_props = record.get("rel_props") # This is our new, reliable properties map
    
    subject, predicate, object_val = None, None, None
    
    if isinstance(path_data, neo4j.graph.Path):
        if path_data.start_node and path_data.relationships and path_data.end_node:
            subject = path_data.start_node.get('name', 'Unknown')
            predicate = path_data.relationships[0].type.replace('_', ' ').lower()
            object_val = path_data.end_node.get('name', 'Unknown')
    elif isinstance(path_data, list) and len(path_data) >= 3:
        if isinstance(path_data[0], dict) and isinstance(path_data[2], dict):
            subject = path_data[0].get('name', 'Unknown')
            predicate = str(path_data[1]).replace('_', ' ').lower()
            object_val = path_data[2].get('name', 'Unknown')

    if not all([subject, predicate, object_val]):
        logger.warning(f"Could not parse path structure from record: {record}")
        return None

    text_representation = f"{subject} {predicate} {object_val}."
    
    # Use the reliable properties map for citation
    if isinstance(rel_props, dict):
        doc_id = rel_props.get('doc_id')
        url = rel_props.get('source_pdf_url')
        page_num_str = rel_props.get('page_numbers') # This is already a string from the uploader

        if url and doc_id:
            link_url = f"{url}#page={page_num_str}" if page_num_str and page_num_str != "N/A" else url
            citation_text = f"{doc_id} (Page {page_num_str or 'N/A'})"
            citation = f'<a href="{link_url}" target="_blank">{citation_text}</a>'
            return f"Evidence from graph: {text_representation}\nCitation: {citation}"

    logger.warning(f"Missing citation metadata in rel_props for '{text_representation}'. Falling back.")
    citation = '<a href="#references" target="_blank">Knowledge Graph</a>'
    return f"Evidence from graph: {text_representation}\nCitation: {citation}"
# --- END OF DEFINITIVE FIX ---

def _vector_search_tool(query: str, namespace: str, tool_name: str, top_k: int = 7) -> ToolResult:
    # No changes needed here
    # ... (code is identical to previous correct version)
    pinecone_index = get_pinecone_index()
    embedding_client = get_google_ai_client()
    if not pinecone_index or not embedding_client: return ToolResult(tool_name=tool_name, success=False, content="Vector search client not available.")
    try:
        embedding_model = 'models/text-embedding-004'
        query_embedding = embedding_client.embed_content(model=embedding_model, content=query, task_type="retrieval_query")['embedding']
        response = pinecone_index.query(namespace=namespace, vector=query_embedding, top_k=top_k, include_metadata=True)
        if not response or not response.get('matches'):
            logger.info(f"[Tool: {tool_name}] Vector search in '{namespace}' found 0 matches.")
            return ToolResult(tool_name=tool_name, success=True, content="")
        content_list = _format_pinecone_results(response['matches'])
        return ToolResult(tool_name=tool_name, success=True, content="\n---\n".join(content_list))
    except Exception as e:
        logger.error(f"[Tool: {tool_name}] Error during vector search: {e}", exc_info=True)
        return ToolResult(tool_name=tool_name, success=False, content=f"An error occurred in the tool: {e}")


def retrieve_clinical_data(query: str, query_meta: QueryMetadata) -> ToolResult:
    return _vector_search_tool(query, "pbac-clinical", "retrieve_clinical_data")
def retrieve_summary_data(query: str, query_meta: QueryMetadata) -> ToolResult:
    return _vector_search_tool(query, "pbac-summary", "retrieve_summary_data")
def retrieve_general_text(query: str, query_meta: QueryMetadata) -> ToolResult:
    return _vector_search_tool(query, "pbac-text", "retrieve_general_text")

def query_knowledge_graph(query: str, query_meta: QueryMetadata) -> ToolResult:
    tool_name = "query_knowledge_graph"
    if not query_meta.question_is_graph_suitable: return ToolResult(tool_name=tool_name, success=True, content="")
    llm = get_google_ai_client().GenerativeModel('gemini-1.5-flash-latest')
    driver = get_neo4j_driver()
    if not llm or not driver: return ToolResult(tool_name=tool_name, success=False, content="LLM or Neo4j client is not available.")
    
    try:
        with driver.session(database="neo4j") as session:
            schema_data = session.run("CALL db.schema.visualization()").data()
        schema_str = f"Node labels and properties: {schema_data[0]['nodes']}\nRelationship types: {schema_data[0]['relationships']}"
        prompt = CYPHER_GENERATION_PROMPT.format(schema=schema_str, question=query)
        response = llm.generate_content(prompt)
        cypher_query = response.text.strip().replace("```cypher", "").replace("```", "")
        if "none" in cypher_query.lower() or "match" not in cypher_query.lower():
            return ToolResult(tool_name=tool_name, success=True, content="")
        logger.info(f"[Tool: {tool_name}] Generated Cypher: {cypher_query}")
    except Exception as e:
        logger.error(f"[Tool: {tool_name}] Cypher generation failed: {e}", exc_info=True)
        return ToolResult(tool_name=tool_name, success=False, content=f"Cypher generation failed: {e}")
        
    try:
        with driver.session(database="neo4j") as session:
            records = session.run(cypher_query).data()
        
        if not records:
            logger.warning(f"[Tool: {tool_name}] Cypher query returned 0 records.")
            return ToolResult(tool_name=tool_name, success=True, content="")

        logger.info(f"[Tool: {tool_name}] Cypher query returned {len(records)} record(s).")
        
        results = [_serialize_graph_record(record) for record in records]
        valid_results = [res for res in results if res is not None]

        if not valid_results:
             logger.warning(f"[Tool: {tool_name}] All {len(records)} records failed serialization.")
             return ToolResult(tool_name=tool_name, success=True, content="")

        return ToolResult(tool_name=tool_name, success=True, content="\n".join(valid_results))
    except Exception as e:
        logger.error(f"[Tool: {tool_name}] Cypher execution/serialization failed: {e}", exc_info=True)
        return ToolResult(tool_name=tool_name, success=False, content=f"Cypher execution failed: {e}")