# FILE: src/tools/retrievers.py
# V5.0 (Definitive Sync): Reverted to a stable, synchronous architecture with timing logs.

import logging
import time
from typing import List, Dict, Any
import neo4j

from src.tools.clients import get_google_ai_client, get_pinecone_index, get_neo4j_driver
from src.models import ToolResult, QueryMetadata
from src.prompts import CYPHER_GENERATION_PROMPT

logger = logging.getLogger(__name__)

class Timer:
    """A simple context manager for timing code blocks."""
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    def __exit__(self, *args):
        self.end = time.perf_counter()
        duration = (self.end - self.start) * 1000
        logger.info(f"[TIMER] {self.name} took {duration:.2f} ms")

# --- Helper Functions (Unchanged) ---
def _format_pinecone_results(matches: List[dict]) -> List[str]:
    # ... (code is identical to previous correct version)
    contents = []
    for match in matches:
        metadata = match.get('metadata', {})
        text = metadata.get('text', 'No content available.')
        doc_id = metadata.get('doc_id', 'Unknown Document')
        page_numbers = metadata.get('page_numbers', [])
        url = metadata.get('source_pdf_url', '#')
        page_str = ", ".join(map(str, sorted(list(set(page_numbers))))) if page_numbers else "N/A"
        link_url = f"{url}#page={page_numbers[0]}" if page_numbers else url
        citation_text = f"{doc_id} (Page {page_str})"
        citation = f'<a href="{link_url}" target="_blank">{citation_text}</a>'
        contents.append(f"Evidence from document: {text}\nCitation: {citation}")
    return contents

def _serialize_neo4j_path(record: Dict[str, Any]) -> str:
    # ... (code is identical to previous correct version)
    path_data, rel_props = record.get("p"), record.get("rel_props")
    subject, predicate, object_val = None, None, None
    if isinstance(path_data, neo4j.graph.Path):
        subject, predicate, object_val = path_data.start_node.get('name'), path_data.relationships[0].type, path_data.end_node.get('name')
    elif isinstance(path_data, list) and len(path_data) >= 3:
        subject, predicate, object_val = path_data[0].get('name'), str(path_data[1]), path_data[2].get('name')
    if not all([subject, predicate, object_val]):
        return f"A complex relationship found. <a href='#' target='_blank'>Knowledge Graph</a>"
    text_representation = f"{subject} {predicate.replace('_', ' ').lower()} {object_val}."
    if isinstance(rel_props, dict):
        doc_id, url, page_num_str = rel_props.get('doc_id'), rel_props.get('source_pdf_url'), rel_props.get('page_numbers', 'N/A')
        if url and doc_id:
            first_page = page_num_str.split(',')[0].strip()
            link_url = f"{url}#page={first_page}" if first_page.isdigit() else url
            citation_text = f"{doc_id} (Page {page_num_str})"
            citation = f'<a href="{link_url}" target="_blank">{citation_text}</a>'
            return f"Evidence from graph: {text_representation}\nCitation: {citation}"
    return f"Evidence from graph: {text_representation}\nCitation: <a href='#' target='_blank'>Knowledge Graph</a>"


# --- Tool Functions (DEFINITIVE SYNC FIX) ---

def _vector_search_tool(query: str, namespace: str, tool_name: str, top_k: int = 7) -> ToolResult:
    with Timer(f"Tool: {tool_name}"):
        pinecone_index = get_pinecone_index()
        embedding_client = get_google_ai_client()
        if not pinecone_index or not embedding_client: 
            return ToolResult(tool_name=tool_name, success=False, content="Vector search client not available.")
        try:
            with Timer("Embedding Generation"):
                query_embedding = embedding_client.embed_content(model='models/text-embedding-004', content=query, task_type="retrieval_query")
            with Timer("Pinecone Query"):
                response = pinecone_index.query(namespace=namespace, vector=query_embedding['embedding'], top_k=top_k, include_metadata=True)
            if not response.get('matches'): return ToolResult(tool_name=tool_name, success=True, content="")
            content_list = _format_pinecone_results(response['matches'])
            return ToolResult(tool_name=tool_name, success=True, content="\n---\n".join(content_list))
        except Exception as e:
            logger.error(f"Error in vector search tool '{tool_name}': {e}", exc_info=True)
            return ToolResult(tool_name=tool_name, success=False, content=f"An error occurred: {e}")

def retrieve_clinical_data(query: str, query_meta: QueryMetadata) -> ToolResult:
    return _vector_search_tool(query, "pbac-clinical", "retrieve_clinical_data")

def retrieve_summary_data(query: str, query_meta: QueryMetadata) -> ToolResult:
    return _vector_search_tool(query, "pbac-summary", "retrieve_summary_data")

def retrieve_general_text(query: str, query_meta: QueryMetadata) -> ToolResult:
    return _vector_search_tool(query, "pbac-text", "retrieve_general_text")

def query_knowledge_graph(query: str, query_meta: QueryMetadata) -> ToolResult:
    with Timer("Tool: query_knowledge_graph"):
        tool_name = "query_knowledge_graph"
        if not query_meta.question_is_graph_suitable: 
            return ToolResult(tool_name=tool_name, success=True, content="")
        llm = get_google_ai_client().GenerativeModel('gemini-1.5-flash-latest')
        driver = get_neo4j_driver()
        if not llm or not driver: return ToolResult(tool_name=tool_name, success=False, content="LLM or Neo4j client not available.")
        try:
            with Timer("KG Schema Fetch"):
                with driver.session() as session: schema_data = session.run("CALL db.schema.visualization()").data()
            schema_str = f"Node labels: {schema_data[0]['nodes']}\nRelationships: {schema_data[0]['relationships']}"
            prompt = CYPHER_GENERATION_PROMPT.format(schema=schema_str, question=query)
            with Timer("Cypher Generation LLM Call"):
                response = llm.generate_content(prompt)
            cypher_query = response.text.strip().replace("```cypher", "").replace("```", "")
            if "none" in cypher_query.lower() or "match" not in cypher_query.lower(): 
                return ToolResult(tool_name=tool_name, success=True, content="")
            logger.info(f"Generated Cypher: {cypher_query}")
        except Exception as e:
            return ToolResult(tool_name=tool_name, success=False, content=f"Cypher generation failed: {e}")
        try:
            with Timer("KG Query Execution"):
                with driver.session() as session: records = session.run(cypher_query).data()
            if not records: return ToolResult(tool_name=tool_name, success=True, content="")
            results = [_serialize_neo4j_path(record) for record in records if record.get("p")]
            return ToolResult(tool_name=tool_name, success=True, content="\n".join(results))
        except Exception as e:
            return ToolResult(tool_name=tool_name, success=False, content=f"Cypher execution failed: {e}")