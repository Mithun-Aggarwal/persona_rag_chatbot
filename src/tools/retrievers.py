# FILE: src/tools/retrievers.py
# V2.0: Centralized, tool-based retrieval functions.
"""
This module contains the actual "tools" the agent can execute.
Each function corresponds to a tool name, fetches data from a source,
and returns a standardized ToolResult.
"""

import logging
from typing import List

import neo4j

from src.tools.clients import get_google_ai_client, get_pinecone_index, get_neo4j_driver
from src.models import ToolResult, QueryMetadata
from src.prompts import CYPHER_GENERATION_PROMPT

logger = logging.getLogger(__name__)

# --- Helper Functions ---

def _format_pinecone_results(matches: List[dict]) -> List[str]:
    """Standardizes Pinecone matches into a list of content strings with clear citations."""
    contents = []
    for match in matches:
        metadata = match.get('metadata', {})
        text = metadata.get('text', 'No content available.')
        doc_id = metadata.get('doc_id', 'Unknown Document')
        page_num = metadata.get('page_number') or metadata.get('page', 'N/A')
        score = match.get('score', 0.0)
        
        # Format a clean citation string for the LLM to use
        citation = f"[Source: {doc_id}, Page: {page_num}, Similarity: {score:.2f}]"
        contents.append(f"{text}\n{citation}")
        
    return contents

def _serialize_neo4j_path(path: neo4j.graph.Path) -> str:
    """Helper to convert a Neo4j Path into a readable text representation."""
    nodes_str = []
    for i, node in enumerate(path.nodes):
        node_label = next(iter(node.labels), "Node")
        name = node.get('name', node.get('id', 'Unknown'))
        nodes_str.append(f"({name}:{node_label})")
        if i < len(path.relationships):
            rel = path.relationships[i]
            nodes_str.append(f"-[{rel.type}]->")
    return "".join(nodes_str)

# --- Vector Search Tools ---

def _vector_search_tool(query: str, namespace: str, tool_name: str, top_k: int = 7) -> ToolResult:
    """Generic, reusable vector search tool for a specific Pinecone namespace."""
    pinecone_index = get_pinecone_index()
    embedding_client = get_google_ai_client()
    if not pinecone_index or not embedding_client:
        return ToolResult(tool_name=tool_name, success=False, content="Vector search client not available.")

    try:
        query_embedding = embedding_client.embed_content(
            model='models/text-embedding-004',  # Using a modern, recommended model
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

        content_list = _format_pinecone_results(response['matches'])
        return ToolResult(tool_name=tool_name, success=True, content="\n---\n".join(content_list))

    except Exception as e:
        logger.error(f"Error in '{tool_name}' tool: {e}", exc_info=True)
        return ToolResult(tool_name=tool_name, success=False, content=f"An error occurred while executing the tool: {e}")

# These are the concrete tool functions the ToolRouter will call.
# Each one calls the generic helper with its specific namespace.
def retrieve_clinical_data(query: str, query_meta: QueryMetadata) -> ToolResult:
    return _vector_search_tool(query, "pbac-clinical", "retrieve_clinical_data")

def retrieve_summary_data(query: str, query_meta: QueryMetadata) -> ToolResult:
    return _vector_search_tool(query, "pbac-summary", "retrieve_summary_data")

def retrieve_general_text(query: str, query_meta: QueryMetadata) -> ToolResult:
    return _vector_search_tool(query, "pbac-text", "retrieve_general_text")

# --- Graph Database Tool ---

def query_knowledge_graph(query: str, query_meta: QueryMetadata) -> ToolResult:
    """Generates and executes a Cypher query against the knowledge graph."""
    tool_name = "query_knowledge_graph"
    if not query_meta.question_is_graph_suitable:
        return ToolResult(
            tool_name=tool_name,
            success=True, # Succeeded in its decision to not run
            content="This question was determined to be unsuitable for the knowledge graph."
        )

    llm = get_google_ai_client().GenerativeModel('gemini-1.5-flash-latest')
    driver = get_neo4j_driver()
    if not llm or not driver:
        return ToolResult(tool_name=tool_name, success=False, content="LLM or Neo4j client is not available.")

    # Step 1: Generate the Cypher query
    try:
        # A more efficient schema retrieval for LLM prompts
        with driver.session() as session:
            schema_data = session.run("CALL db.schema.visualization()").data()
        schema_str = f"Node labels and properties: {schema_data[0]['nodes']}\nRelationship types: {schema_data[0]['relationships']}"

        prompt = CYPHER_GENERATION_PROMPT.format(schema=schema_str, question=query)
        response = llm.generate_content(prompt)
        cypher_query = response.text.strip().replace("```cypher", "").replace("```", "")
        
        if "none" in cypher_query.lower() or "match" not in cypher_query.lower():
            logger.warning(f"LLM decided not to generate a Cypher query for: '{query}'")
            return ToolResult(tool_name=tool_name, success=True, content="Could not generate a suitable graph query for this question.")

        logger.info(f"Generated Cypher: {cypher_query}")
    except Exception as e:
        logger.error(f"Error during Cypher generation: {e}", exc_info=True)
        return ToolResult(tool_name=tool_name, success=False, content=f"Cypher generation failed: {e}")

    # Step 2: Execute the generated Cypher query
    try:
        with driver.session() as session:
            records = session.run(cypher_query).data()

        if not records:
            return ToolResult(tool_name=tool_name, success=True, content="The graph query executed successfully but found no results.")
        
        results = []
        for record in records:
            if "p" in record and isinstance(record["p"], neo4j.graph.Path):
                results.append(_serialize_neo4j_path(record["p"]))
            else:
                # Fallback for non-path results (e.g., RETURN count(n))
                results.append(str(record))
        
        return ToolResult(tool_name=tool_name, success=True, content="\n".join(results))
    
    except Exception as e:
        logger.error(f"Error during Cypher execution for query '{cypher_query}': {e}", exc_info=True)
        return ToolResult(tool_name=tool_name, success=False, content=f"The generated Cypher query failed during execution: {e}")