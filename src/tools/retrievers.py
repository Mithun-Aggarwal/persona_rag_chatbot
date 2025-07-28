# FILE: src/tools/retrievers.py
# V2.1 (Citation Fix): Upgrades the knowledge graph tool to extract and format PDF citations.

import logging
from typing import List
import neo4j
import ast

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
        score = match.get('score', 0.0)
        # Format a clean, clickable citation string
        citation = f"[Source: {doc_id}, Page: {page_num}]({url})"
        contents.append(f"{text}\n{citation}")
    return contents

# --- START OF CITATION FIX ---
def _serialize_neo4j_path(path: neo4j.graph.Path) -> str:
    """
    Helper to convert a Neo4j Path into a readable text representation
    WITH a formatted, clickable Markdown citation.
    """
    path_parts = []
    citations = set()

    for i, node in enumerate(path.nodes):
        node_label = next(iter(node.labels), "Node")
        name = node.get('name', 'Unknown')
        path_parts.append(f"({name}:{node_label})")
        
        if i < len(path.relationships):
            rel = path.relationships[i]
            path_parts.append(f"-[{rel.type}]->")
            
            # Extract metadata from the relationship properties
            doc_id = rel.get('doc_id', 'Unknown Document')
            url = rel.get('source_pdf_url', '#')
            try:
                # Page numbers are stored as a string representation of a list
                pages_str = rel.get('page_numbers', '[]')
                pages_list = ast.literal_eval(pages_str)
                page_num = ", ".join(map(str, sorted(pages_list))) if pages_list else "N/A"
            except (ValueError, SyntaxError):
                page_num = "N/A"

            # Create a unique, clickable markdown citation and add to our set
            citation = f"[Source: {doc_id}, Page: {page_num}]({url})"
            citations.add(citation)
    
    # Join the path and the unique citations
    text_representation = "".join(path_parts)
    citation_str = " ".join(sorted(list(citations)))
    return f"{text_representation}\n{citation_str}"
# --- END OF CITATION FIX ---

# Vector Search Tools (no changes needed here)
def _vector_search_tool(query: str, namespace: str, tool_name: str, top_k: int = 7) -> ToolResult:
    # ... (code is correct, but let's add the clickable URL to pinecone results too)
    pinecone_index = get_pinecone_index()
    embedding_client = get_google_ai_client()
    if not pinecone_index or not embedding_client:
        return ToolResult(tool_name=tool_name, success=False, content="Vector search client not available.")

    try:
        query_embedding = embedding_client.embed_content(
            model='models/text-embedding-004',
            content=query,
            task_type="retrieval_query"
        )['embedding']
        
        response = pinecone_index.query(
            namespace=namespace,
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        if not response.get('matches'):
            return ToolResult(tool_name=tool_name, success=True, content="No relevant information was found for this query.")
        
        # This now calls the updated formatter which creates clickable links
        content_list = _format_pinecone_results(response['matches'])
        return ToolResult(tool_name=tool_name, success=True, content="\n---\n".join(content_list))

    except Exception as e:
        logger.error(f"Error in '{tool_name}' tool: {e}", exc_info=True)
        return ToolResult(tool_name=tool_name, success=False, content=f"An error occurred while executing the tool: {e}")

def retrieve_clinical_data(query: str, query_meta: QueryMetadata) -> ToolResult:
    return _vector_search_tool(query, "pbac-clinical", "retrieve_clinical_data")

def retrieve_summary_data(query: str, query_meta: QueryMetadata) -> ToolResult:
    return _vector_search_tool(query, "pbac-summary", "retrieve_summary_data")

def retrieve_general_text(query: str, query_meta: QueryMetadata) -> ToolResult:
    return _vector_search_tool(query, "pbac-text", "retrieve_general_text")

# --- Graph Database Tool (with changes) ---
def query_knowledge_graph(query: str, query_meta: QueryMetadata) -> ToolResult:
    tool_name = "query_knowledge_graph"
    if not query_meta.question_is_graph_suitable:
        return ToolResult(tool_name=tool_name, success=True, content="This question was determined to be unsuitable for the knowledge graph.")

    llm = get_google_ai_client().GenerativeModel('gemini-1.5-flash-latest')
    driver = get_neo4j_driver()
    if not llm or not driver:
        return ToolResult(tool_name=tool_name, success=False, content="LLM or Neo4j client is not available.")

    try:
        with driver.session() as session:
            schema_data = session.run("CALL db.schema.visualization()").data()
        schema_str = f"Node labels and properties: {schema_data[0]['nodes']}\nRelationship types: {schema_data[0]['relationships']}"
        prompt = CYPHER_GENERATION_PROMPT.format(schema=schema_str, question=query)
        response = llm.generate_content(prompt)
        cypher_query = response.text.strip().replace("```cypher", "").replace("```", "")
        
        if "none" in cypher_query.lower() or "match" not in cypher_query.lower():
            return ToolResult(tool_name=tool_name, success=True, content="Could not generate a suitable graph query for this question.")
        logger.info(f"Generated Cypher: {cypher_query}")
    except Exception as e:
        logger.error(f"Error during Cypher generation: {e}", exc_info=True)
        return ToolResult(tool_name=tool_name, success=False, content=f"Cypher generation failed: {e}")

    try:
        with driver.session() as session:
            # We are now executing the query and getting back the raw records
            records = session.run(cypher_query).data()

        if not records:
            return ToolResult(tool_name=tool_name, success=True, content="The graph query executed successfully but found no results.")
        
        results = []
        for record in records:
            if "p" in record and isinstance(record["p"], neo4j.graph.Path):
                # --- START OF CITATION FIX ---
                # Call the NEW _serialize_neo4j_path function which includes citations
                results.append(_serialize_neo4j_path(record["p"]))
                # --- END OF CITATION FIX ---
            else:
                results.append(str(record))
        
        return ToolResult(tool_name=tool_name, success=True, content="\n".join(results))
    
    except Exception as e:
        logger.error(f"Error during Cypher execution for query '{cypher_query}': {e}", exc_info=True)
        return ToolResult(tool_name=tool_name, success=False, content=f"The generated Cypher query failed during execution: {e}")