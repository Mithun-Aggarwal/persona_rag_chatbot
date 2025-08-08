# FILE: src/tools/retrievers.py
# V17.0 (The Unbreakable Fix): This is the definitive version based on conclusive log analysis.
#
# THE FLAW: All previous attempts incorrectly assumed the neo4j driver's `.data()` method
# returned rich graph objects. The logs proved it returns simple Python dictionaries.
#
# THE FIX:
# 1. The serialization logic has been completely rewritten to treat the contents of the
#    record (`start_node`, `end_node`) as simple dictionaries.
# 2. It now accesses values using dictionary keys (e.g., `start_node.get('name')`) instead
#    of trying to call object methods.
#
# This approach is simple, direct, aligns perfectly with the observed log data, and
# will definitively resolve all previous KG tool failures.

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

# --- START OF DEFINITIVE FIX V17.0 ---
def _serialize_kg_results(records: List[Dict[str, Any]]) -> List[str]:
    """
    Serializes records from the simple Cypher query pattern.
    It correctly treats all returned values as simple Python dictionaries and primitives.
    """
    facts = []
    for record in records:
        # The .data() method returns dictionaries, not rich objects. Access accordingly.
        start_node_dict = record.get("start_node")
        end_node_dict = record.get("end_node")
        rel_type = record.get("rel_type")
        rel_props = record.get("r_props")

        # Check for the presence of the dictionaries/strings themselves
        if not all([start_node_dict, end_node_dict, rel_type, rel_props]):
            logger.warning(f"Skipping record due to missing core data keys: {record}")
            continue
        
        try:
            # Access values using dictionary .get() method for safety
            subject_name = start_node_dict.get('name', 'Unknown')
            object_name = end_node_dict.get('name', 'Unknown')
            
            predicate_str = str(rel_type).replace('_', ' ').lower()
            text_representation = f"{subject_name.upper()} {predicate_str} {object_name.upper()}."

            citation = _format_citation(
                rel_props.get('doc_id', 'KG'),
                rel_props.get('page_numbers'),
                rel_props.get('source_pdf_url') # Match key from r_props
            )
            facts.append(f"Evidence from graph: {text_representation}\nCitation: {citation}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during KG serialization: {e}. Record: {record}", exc_info=True)
            continue

    return facts
# --- END OF DEFINITIVE FIX V17.0 ---

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
                if prop_name:
                    prop_type = node.get('propertyTypes', ['String'])[0]
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
                if prop_name:
                    prop_type = rel.get('propertyTypes', ['String'])[0]
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
            # The .data() method here is the source of the dictionaries
            with driver.session() as session: records = session.run(cypher_query).data()
            if not records: return ToolResult(tool_name=tool_name, success=True, content="")
            
            all_facts = _serialize_kg_results(records)
            
            final_content = "\n---\n".join(filter(None, all_facts))
            if final_content:
                logger.info(f"KG tool successfully retrieved and serialized {len(all_facts)} facts.")
            else:
                logger.warning("KG tool ran and found records, but serialization produced no content.")
            return ToolResult(tool_name=tool_name, success=True, content=final_content)
            
        except neo4j.exceptions.CypherSyntaxError as e:
            logger.error(f"A Cypher syntax error occurred in the KG tool. Query: '{cypher_query}'. Error: {e}", exc_info=False)
            return ToolResult(tool_name=tool_name, success=False, content=f"A database syntax error occurred.")
        except Exception as e:
            logger.error(f"An unexpected error occurred in the KG tool: {e}", exc_info=True)
            return ToolResult(tool_name=tool_name, success=False, content=f"An unexpected error occurred.")