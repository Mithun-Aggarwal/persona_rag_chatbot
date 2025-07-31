# FILE: src/tools/retrievers.py
# V5.4 (Definitive Serializer Fix): Added logic to handle list-based path results from Neo4j.

import logging
import time
from typing import List, Dict, Any
import neo4j
import google.generativeai as genai

from src.tools.clients import get_flash_model, get_pinecone_index, get_neo4j_driver, DEFAULT_REQUEST_OPTIONS
from src.models import ToolResult, QueryMetadata
from src.prompts import CYPHER_GENERATION_PROMPT

logger = logging.getLogger(__name__)

class Timer:
    def __init__(self, name): self.name = name
    def __enter__(self): self.start = time.perf_counter(); return self
    def __exit__(self, *args): self.end = time.perf_counter(); logger.info(f"[TIMER] {self.name} took {(self.end - self.start) * 1000:.2f} ms")

def _format_pinecone_results(matches: List[dict]) -> List[str]:
    contents = []
    MAX_PAGES_TO_SHOW = 4
    for match in matches:
        metadata = match.get('metadata', {})
        text = metadata.get('text', 'No content available.')
        doc_id = metadata.get('doc_id', 'Unknown Document')
        page_numbers_raw = metadata.get('page_numbers', [])
        url = metadata.get('source_pdf_url', '#')
        page_str, link_url = "N/A", url
        if page_numbers_raw:
            unique_pages = sorted(list(set(page_numbers_raw)))
            if len(unique_pages) > MAX_PAGES_TO_SHOW:
                page_str = ", ".join(map(str, unique_pages[:MAX_PAGES_TO_SHOW])) + ", ..."
            else:
                page_str = ", ".join(map(str, unique_pages))
            if unique_pages:
                link_url = f"{url}#page={unique_pages[0]}"
        citation_text = f"{doc_id} (Page {page_str})"
        citation = f'<a href="{link_url}" target="_blank">{citation_text}</a>'
        contents.append(f"Evidence from document: {text}\nCitation: {citation}")
    return contents

def _serialize_neo4j_path(record: Dict[str, Any]) -> str:
    path_data = record.get("p")
    rel_props = record.get("rel_props")

    if not path_data:
        return ""

    subject_name, predicate_type, object_name = None, None, None

    try:
        if isinstance(path_data, neo4j.graph.Path):
            subject_name = path_data.start_node.get('name')
            predicate_type = path_data.relationships[0].type
            object_name = path_data.end_node.get('name')
        
        elif isinstance(path_data, list) and len(path_data) == 3 and all(isinstance(i, (dict, str)) for i in path_data):
            subject_name = path_data[0].get('name')
            predicate_type = path_data[1]
            object_name = path_data[2].get('name')
        
        elif isinstance(path_data, dict) and 'start' in path_data and 'end' in path_data:
            subject_name = path_data['start'].get('properties', {}).get('name')
            predicate_type = path_data.get('segments', [{}])[0].get('relationship', {}).get('type')
            object_name = path_data['end'].get('properties', {}).get('name')

        if not all([subject_name, predicate_type, object_name]):
            logger.warning(f"Could not fully parse Neo4j path data with any known method: {path_data}")
            return ""

        predicate_str = predicate_type.replace('_', ' ').lower()
        text_representation = f"{subject_name} {predicate_str} {object_name}."
        
        citation_text, link_url = "Knowledge Graph", "#"
        if isinstance(rel_props, dict):
            doc_id = rel_props.get('doc_id')
            url = rel_props.get('source_pdf_url')
            page_num_str = rel_props.get('page_numbers', 'N/A')
            if doc_id:
                citation_text = f"{doc_id} (Page {page_num_str})"
            if url:
                first_page = page_num_str.split(',')[0].strip()
                link_url = f"{url}#page={first_page}" if first_page.isdigit() else url
        
        citation = f'<a href="{link_url}" target="_blank">{citation_text}</a>'
        return f"Evidence from graph: {text_representation}\nCitation: {citation}"

    except (AttributeError, IndexError, TypeError) as e:
        logger.error(f"Failed to serialize Neo4j record due to parsing error: {e}", exc_info=True)
        return ""

def _vector_search_tool(query: str, namespace: str, tool_name: str, top_k: int = 7) -> ToolResult:
    with Timer(f"Tool: {tool_name}"):
        pinecone_index = get_pinecone_index()
        if not pinecone_index or not genai:
            return ToolResult(tool_name=tool_name, success=False, content="Vector search client not available.")
        try:
            with Timer("Embedding Generation"):
                query_embedding = genai.embed_content(model='models/text-embedding-004', content=query, task_type="retrieval_query")
            with Timer("Pinecone Query"):
                response = pinecone_index.query(namespace=namespace, vector=query_embedding['embedding'], top_k=top_k, include_metadata=True)
            if not response.get('matches'):
                return ToolResult(tool_name=tool_name, success=True, content="")
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
        llm, driver = get_flash_model(), get_neo4j_driver()
        if not llm or not driver:
            return ToolResult(tool_name=tool_name, success=False, content="LLM or Neo4j client not available.")
        try:
            with Timer("KG Schema Fetch"):
                with driver.session() as session:
                    schema_data = session.run("CALL db.schema.visualization()").data()
            schema_str = f"Node labels: {schema_data[0]['nodes']}\nRelationships: {schema_data[0]['relationships']}"
            prompt = CYPHER_GENERATION_PROMPT.format(schema=schema_str, question=query)
            with Timer("Cypher Generation LLM Call"):
                response = llm.generate_content(prompt, request_options=DEFAULT_REQUEST_OPTIONS)
            cypher_query = response.text.strip().replace("```cypher", "").replace("```", "")
            if "none" in cypher_query.lower() or "match" not in cypher_query.lower():
                return ToolResult(tool_name=tool_name, success=True, content="")
            logger.info(f"Generated Cypher: {cypher_query}")
        except Exception as e:
            return ToolResult(tool_name=tool_name, success=False, content=f"Cypher generation failed: {e}")
        try:
            with Timer("KG Query Execution"):
                with driver.session() as session:
                    records = session.run(cypher_query).data()
            if not records:
                return ToolResult(tool_name=tool_name, success=True, content="")
            results = [_serialize_neo4j_path(record) for record in records if record.get("p")]
            results = [res for res in results if res]
            return ToolResult(tool_name=tool_name, success=True, content="\n".join(results))
        except Exception as e:
            return ToolResult(tool_name=tool_name, success=False, content=f"Cypher execution failed: {e}")