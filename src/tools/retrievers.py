# FILE: src/tools/retrievers.py
# V3.0 (Production Ready): Final, confirmed version.

import logging
from typing import List
import neo4j

from src.tools.clients import get_google_ai_client, get_pinecone_index, get_neo4j_driver
from src.models import ToolResult, QueryMetadata
from src.prompts import CYPHER_GENERATION_PROMPT

logger = logging.getLogger(__name__)

# --- Helper Functions ---

def _format_pinecone_results(matches: List[dict]) -> List[str]:
    contents = []
    for match in matches:
        metadata = match.get('metadata', {})
        text = metadata.get('text', 'No content available.')
        doc_id = metadata.get('doc_id', 'Unknown Document')
        page_num = metadata.get('page_number', 'N/A')
        url = metadata.get('source_pdf_url', '#')
        citation = f"[Source: {doc_id}, Page: {page_num}]({url})"
        contents.append(f"{text}\n{citation}")
    return contents

def _serialize_neo4j_path(path: neo4j.graph.Path) -> str:
    citations = set()
    if len(path.nodes) == 2 and len(path.relationships) == 1:
        subject = path.start_node.get('name', 'Unknown')
        predicate = path.relationships[0].type.replace('_', ' ').lower()
        object_val = path.end_node.get('name', 'Unknown')
        rel = path.relationships[0]
        
        doc_id = rel.get('doc_id', 'Unknown Document')
        url = rel.get('source_pdf_url', '#')
        page_num = rel.get('page_numbers', 'N/A')
        citation = f"[Source: {doc_id}, Page: {page_num}]({url})"
        citations.add(citation)
        
        text_representation = f"{subject} {predicate} {object_val}."
        citation_str = " ".join(sorted(list(citations)))
        return f"{text_representation}\n{citation_str}"

    return f"A complex relationship was found in the knowledge graph.\n[Source: Knowledge Graph, Page: N/A](#)"

def _vector_search_tool(query: str, namespace: str, tool_name: str, top_k: int = 7) -> ToolResult:
    pinecone_index = get_pinecone_index()
    embedding_client = get_google_ai_client()
    if not pinecone_index or not embedding_client: return ToolResult(tool_name=tool_name, success=False, content="Vector search client not available.")
    try:
        query_embedding = embedding_client.embed_content(model='models/text-embedding-004', content=query, task_type="retrieval_query")['embedding']
        response = pinecone_index.query(namespace=namespace, vector=query_embedding, top_k=top_k, include_metadata=True)
        if not response.get('matches'): return ToolResult(tool_name=tool_name, success=True, content="")
        content_list = _format_pinecone_results(response['matches'])
        return ToolResult(tool_name=tool_name, success=True, content="\n---\n".join(content_list))
    except Exception as e:
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
        with driver.session() as session:
            schema_data = session.run("CALL db.schema.visualization()").data()
        schema_str = f"Node labels and properties: {schema_data[0]['nodes']}\nRelationship types: {schema_data[0]['relationships']}"
        prompt = CYPHER_GENERATION_PROMPT.format(schema=schema_str, question=query)
        response = llm.generate_content(prompt)
        cypher_query = response.text.strip().replace("```cypher", "").replace("```", "")
        if "none" in cypher_query.lower() or "match" not in cypher_query.lower(): return ToolResult(tool_name=tool_name, success=True, content="")
        logger.info(f"Generated Cypher: {cypher_query}")
    except Exception as e:
        return ToolResult(tool_name=tool_name, success=False, content=f"Cypher generation failed: {e}")
    try:
        with driver.session() as session:
            records = session.run(cypher_query).data()
        if not records: return ToolResult(tool_name=tool_name, success=True, content="")
        results = [
            _serialize_neo4j_path(record["p"]) if "p" in record and isinstance(record["p"], neo4j.graph.Path) else f"{str(record)}\n[Source: Knowledge Graph, Page: N/A](#)"
            for record in records
        ]
        return ToolResult(tool_name=tool_name, success=True, content="\n".join(results))
    except Exception as e:
        return ToolResult(tool_name=tool_name, success=False, content=f"Cypher execution failed: {e}")