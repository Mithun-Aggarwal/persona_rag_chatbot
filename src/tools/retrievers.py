# FILE: src/tools/retrievers.py
# V7.5 (Definitive Fix):
# 1. Completely re-architected _serialize_neo4j_path to be robust. It now correctly
#    handles BOTH rich `neo4j.graph.Path` objects (for complex multi-hop queries) AND
#    simple list-based results (for single-hop queries) returned by the driver.
# 2. This solves the critical bug where the KG tool was silently failing on valid
#    queries, ensuring it can now reliably answer both simple and complex questions.

import logging
import time
from typing import List, Dict, Any, Union
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


def _format_citation(doc_id: str, page_numbers: Union[List[int], str, None], source_url: str) -> str:
    """Creates a standardized, clickable HTML citation link."""
    page_str = "N/A"
    link_url = source_url if source_url else "#"
    
    if page_numbers:
        if isinstance(page_numbers, list) and all(isinstance(p, int) for p in page_numbers):
            unique_pages = sorted(list(set(page_numbers)))
            if unique_pages:
                page_str = f"Page {unique_pages[0]}" if len(unique_pages) == 1 else f"Pages {', '.join(map(str, unique_pages))}"
                if source_url:
                    link_url = f"{source_url}#page={unique_pages[0]}"
        elif isinstance(page_numbers, str):
            page_str = f"Page {page_numbers}"
            first_page = page_numbers.split(',')[0].split('-')[0].strip()
            if first_page.isdigit() and source_url:
                 link_url = f"{source_url}#page={first_page}"

    return f'<a href="{link_url}" target="_blank">{doc_id} ({page_str})</a>'

def _format_pinecone_results(matches: List[dict]) -> List[str]:
    contents = []
    for match in matches:
        metadata = match.get('metadata', {})
        text = metadata.get('text', 'No content available.')
        doc_id = metadata.get('doc_id', 'Unknown Document')
        page_numbers = metadata.get('page_numbers')
        url = metadata.get('source_pdf_url')
        
        citation = _format_citation(doc_id, page_numbers, url)
        contents.append(f"Evidence from document: {text}\nCitation: {citation}")
    return contents

# --- START OF DEFINITIVE FIX: Robust, Type-Aware Path Serializer ---
def _serialize_neo4j_path(record: Dict[str, Any]) -> List[str]:
    """
    Serializes a Neo4j path of arbitrary length into a list of citable facts.
    Handles both rich `neo4j.graph.Path` objects and simple list-based results.
    """
    path_data = record.get("p")
    rel_props_list = record.get("rel_props_list")
    
    if not path_data or not rel_props_list:
        return []

    facts = []
    try:
        # Case 1: Handle rich Path objects (typically for multi-hop queries)
        if isinstance(path_data, neo4j.graph.Path):
            relationships = path_data.relationships
        # Case 2: Handle simple list-based results (often for single-hop queries)
        # The driver may return a list: [start_node, relationship, end_node]
        elif isinstance(path_data, list):
            relationships = [item for item in path_data if isinstance(item, neo4j.graph.Relationship)]
        else:
            logger.warning(f"Unrecognized Neo4j path data type: {type(path_data)}. Could not serialize.")
            return []

        # Iterate through each relationship segment in the path
        for i, rel in enumerate(relationships):
            if i >= len(rel_props_list): continue
            rel_props = rel_props_list[i]
            if not rel_props: continue
            
            subject_name = rel.start_node.get('name')
            object_name = rel.end_node.get('name')
            rel_type = rel_props.get('type') # Get type from properties map
            
            if not all([subject_name, rel_type, object_name]): continue

            predicate_str = rel_type.replace('_', ' ').lower()
            text_representation = f"{subject_name} {predicate_str} {object_name}."
            
            doc_id = rel_props.get('doc_id', 'Knowledge Graph')
            page_numbers = rel_props.get('page_numbers')
            url = rel_props.get('source_pdf_url')
            citation = _format_citation(doc_id, page_numbers, url)
            facts.append(f"Evidence from graph: {text_representation}\nCitation: {citation}")
            
    except Exception as e:
        logger.error(f"Failed to serialize Neo4j path: {e}", exc_info=True)
    
    return facts
# --- END OF DEFINITIVE FIX ---

def vector_search(query: str, query_meta: QueryMetadata) -> ToolResult:
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
                nodes_schema_result = session.run("CALL db.schema.nodeTypeProperties() YIELD nodeLabels, propertyName, propertyTypes RETURN nodeLabels, propertyName, propertyTypes").data()
                rels_schema_result = session.run("CALL db.schema.relTypeProperties() YIELD relType, propertyName, propertyTypes RETURN relType, propertyName, propertyTypes").data()

            schema_str = "Node Properties:\n"
            processed_nodes: Dict[str, List[str]] = {}
            for node in nodes_schema_result:
                label = node['nodeLabels'][0]
                if label not in processed_nodes: processed_nodes[label] = []
                prop_name = node.get('propertyName')
                prop_type = node.get('propertyTypes', ['String'])[0]
                if prop_name:
                    processed_nodes[label].append(f"{prop_name}: {prop_type}")
            for label, props in processed_nodes.items():
                schema_str += f"- Label: {label}, Properties: {', '.join(props)}\n"
            
            schema_str += "\nRelationship Properties:\n"
            processed_rels: Dict[str, List[str]] = {}
            for rel in rels_schema_result:
                rel_type = rel.get('relType', '').strip('`')
                if not rel_type: continue
                if rel_type not in processed_rels: processed_rels[rel_type] = []
                prop_name = rel.get('propertyName')
                prop_type = rel.get('propertyTypes', ['String'])[0]
                if prop_name:
                    processed_rels[rel_type].append(f"{prop_name}: {prop_type}")
            for rel_type, props in processed_rels.items():
                props_str = ", ".join(props)
                schema_str += f"- (:Entity)-[:{rel_type}]->(:Entity) properties: {props_str}\n"

            prompt = CYPHER_GENERATION_PROMPT.format(schema=schema_str, question=query)
            response = llm.generate_content(prompt, request_options=DEFAULT_REQUEST_OPTIONS)
            cypher_query = response.text.strip().replace("```cypher", "").replace("```", "").replace("`", "")
            
            if "none" in cypher_query.lower() or "match" not in cypher_query.lower(): 
                return ToolResult(tool_name=tool_name, success=True, content="")
                
            logger.info(f"Generated Cypher: {cypher_query}")
            with driver.session() as session: records = session.run(cypher_query).data()
            if not records: return ToolResult(tool_name=tool_name, success=True, content="")
            
            all_facts = []
            for record in records:
                all_facts.extend(_serialize_neo4j_path(record))
            
            return ToolResult(tool_name=tool_name, success=True, content="\n---\n".join(filter(None, all_facts)))
            
        except Exception as e:
            logger.error(f"Error in KG tool: {e}", exc_info=True)
            return ToolResult(tool_name=tool_name, success=False, content=f"An error occurred: {e}")