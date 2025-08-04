# FILE: src/tools/retrievers.py
# V6.4 (Enhanced Schema Introspection): Upgrades the schema generation for the
# Cypher prompt. It now dynamically fetches and includes relationship properties,
# giving the LLM the necessary context to write correct queries that filter on
# relationship attributes like date and source.

import logging
import time
from typing import List, Dict, Any
import neo4j
import google.generativeai as genai

from src.tools.clients import get_flash_model, get_pinecone_index, get_neo4j_driver, DEFAULT_REQUEST_OPTIONS
from src.models import ToolResult, QueryMetadata
from src.prompts import CYPHER_GENERATION_PROMPT

logger = logging.getLogger(__name__)

# ... (Timer class and _format_pinecone_results are unchanged) ...
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

        if page_numbers_raw and all(isinstance(p, (str, int, float)) for p in page_numbers_raw):
            try:
                unique_pages = sorted(list(set(map(int, page_numbers_raw))))
                
                if len(unique_pages) > MAX_PAGES_TO_SHOW:
                    page_str = f"Pages {', '.join(map(str, unique_pages[:MAX_PAGES_TO_SHOW]))}, ..."
                elif len(unique_pages) > 1:
                    page_str = f"Pages {', '.join(map(str, unique_pages))}"
                else:
                    page_str = f"Page {unique_pages[0]}"
                link_url = f"{url}#page={unique_pages[0]}"
            except (ValueError, TypeError):
                 page_str, link_url = ", ".join(map(str, page_numbers_raw)), url
        
        citation = f'<a href="{link_url}" target="_blank">{doc_id} ({page_str})</a>'
        contents.append(f"Evidence from document: {text}\nCitation: {citation}")
    return contents

def _serialize_neo4j_path(record: Dict[str, Any]) -> str:
    # ... (This function is unchanged) ...
    path_data, rel_props = record.get("p"), record.get("rel_props")
    if not path_data: return ""
    subject_name, predicate_type, object_name = None, None, None
    try:
        if isinstance(path_data, neo4j.graph.Path):
            subject_name, predicate_type, object_name = path_data.start_node.get('name'), path_data.relationships[0].type, path_data.end_node.get('name')
        elif isinstance(path_data, list) and len(path_data) == 3:
            subject_name, predicate_type, object_name = path_data[0].get('name'), path_data[1], path_data[2].get('name')
        if not all([subject_name, predicate_type, object_name]): return ""
        predicate_str = predicate_type.replace('_', ' ').lower()
        text_representation = f"{subject_name} {predicate_str} {object_name}."
        citation_text, link_url = "Knowledge Graph", "#"
        if isinstance(rel_props, dict):
            doc_id, url, page_num = rel_props.get('doc_id'), rel_props.get('source_pdf_url'), rel_props.get('page_numbers', 'N/A')
            if doc_id: citation_text = f"{doc_id} (Page {page_num})"
            if url: link_url = f"{url}#page={str(page_num).split(',')[0]}"
        citation = f'<a href="{link_url}" target="_blank">{citation_text}</a>'
        return f"Evidence from graph: {text_representation}\nCitation: {citation}"
    except Exception as e:
        logger.warning(f"Could not serialize Neo4j path: {e}")
        return ""

def vector_search(query: str, query_meta: QueryMetadata) -> ToolResult:
    # ... (This function is unchanged) ...
    tool_name = "vector_search"; namespace = "pbac-text"
    with Timer(f"Tool: {tool_name}"):
        pinecone_index = get_pinecone_index()
        if not pinecone_index: return ToolResult(tool_name=tool_name, success=False, content="Pinecone not available.")
        metadata_filter = {}
        if query_meta and query_meta.themes:
            metadata_filter["semantic_purpose"] = {"$in": query_meta.themes}
            logger.info(f"Applying metadata filter: {metadata_filter}")
        try:
            query_embedding = genai.embed_content(model='models/text-embedding-004', content=query, task_type="retrieval_query")
            response = pinecone_index.query(namespace=namespace, vector=query_embedding['embedding'], top_k=10, include_metadata=True, filter=metadata_filter or None)
            if not response.get('matches'): return ToolResult(tool_name=tool_name, success=True, content="")
            content_list = _format_pinecone_results(response['matches'])
            return ToolResult(tool_name=tool_name, success=True, content="\n---\n".join(content_list))
        except Exception as e:
            logger.error(f"Error in vector search: {e}", exc_info=True)
            return ToolResult(tool_name=tool_name, success=False, content=f"An error occurred: {e}")

def query_knowledge_graph(query: str, query_meta: QueryMetadata) -> ToolResult:
    tool_name = "query_knowledge_graph"
    with Timer(f"Tool: {tool_name}"):
        llm, driver = get_flash_model(), get_neo4j_driver()
        if not llm or not driver: return ToolResult(tool_name=tool_name, success=False, content="Clients not available.")
        try:
            with driver.session() as session:
                nodes_schema = session.run("CALL db.schema.nodeTypeProperties()").data()
                rels_schema = session.run("CALL db.schema.relTypeProperties()").data()

            schema_str = "Node Properties:\n"
            for node in nodes_schema:
                # --- START OF DEFINITIVE FIX ---
                # The correct key for the property name is 'propertyName'.
                prop_name = node['propertyName']
                prop_type = node['propertyTypes'][0] # It's a list, take the first
                schema_str += f"- Label: {node['nodeLabels'][0]}, Properties: {prop_name}: {prop_type}\n"
                # --- END OF DEFINITIVE FIX ---
            
            schema_str += "\nRelationship Properties:\n"
            for rel in rels_schema:
                rel_type = rel.get('relType', '').strip('`')
                properties = rel.get('properties', [])
                props_list = [f"{p['property']}: {p['type']}" for p in properties]
                props_str = ", ".join(props_list)
                schema_str += f"- (:Entity)-[:{rel_type} {{{props_str}}}]->(:Entity)\n"

            prompt = CYPHER_GENERATION_PROMPT.format(schema=schema_str, question=query)
            response = llm.generate_content(prompt, request_options=DEFAULT_REQUEST_OPTIONS)
            cypher_query = response.text.strip().replace("```cypher", "").replace("```", "")
            
            if "none" in cypher_query.lower() or "match" not in cypher_query.lower(): 
                return ToolResult(tool_name=tool_name, success=True, content="")
                
            logger.info(f"Generated Cypher: {cypher_query}")
            with driver.session() as session: records = session.run(cypher_query).data()
            if not records: return ToolResult(tool_name=tool_name, success=True, content="")
            
            results = [_serialize_neo4j_path(record) for record in records if record.get("p")]
            return ToolResult(tool_name=tool_name, success=True, content="\n".join(filter(None, results)))
        except Exception as e:
            logger.error(f"Error in KG tool: {e}", exc_info=True)
            return ToolResult(tool_name=tool_name, success=False, content=f"An error occurred: {e}")